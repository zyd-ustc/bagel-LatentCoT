"""Image-conditioned native teacher: build the target state from the source scene.

Scheme 2 of the coupled construction. The target branch must see the source
scene, not only a rewritten sentence, so the teacher runs the native *image*
conditioned path from the same initial noise up to the read time and hands over
its state; the deployment tail then continues that state under ``c_tail``. The
whole target branch is forward-only: no inversion, no external editor.

The teacher borrows the native IT2I prefixes exactly as ``it2i_generate`` builds
them (text+image for the conditional branch, image-only for the CFG reference),
so it inherits the native editing ability without any new network.
"""

from .conditions import compact
from .diagnosis import semantic_plan
from .runtime import Prefix, autocast, native_pixels

TEACHER_SYSTEM = (
    "Rewrite the user's image request as one precise scene description.\n"
    "Rules:\n"
    "Make every count exact, bind each attribute to its subject, and state the "
    "spatial relations explicitly.\n"
    "Keep the same scene: do not add objects, styles, viewpoints, textures or "
    "details the request does not already imply.\n"
    "Do not give an instruction. Output only the description, at most 60 words."
)


def _query(rt, instruction, image_only):
    from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN

    m, tok = rt.model, rt.tokenizer
    m.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    m.img_start_token_id = tok.convert_tokens_to_ids("<img>")
    prompt = "<image>" if image_only else "<image>\n" + instruction
    if image_only:
        query = m._build_t2i_query(prompt, append_text="<img>")
    else:
        query = m._build_t2i_query(
            prompt, system_message=SYSTEM_MESSAGE_FOR_GEN,
            append_text="<think>\n\n</think>\n\n<img>",
        )
    count = int(rt.grid[0] * rt.grid[1])
    tokens = "<img>" + "<IMG_CONTEXT>" * count + "</img>"
    return query.replace("<image>", tokens, 1)


def prefixes(rt, instruction, image):
    """The two native IT2I prefixes used by the image-conditioned branch."""
    from sensenova_u1.models.neo_unify.modeling_qwen3 import create_block_causal_mask

    m, tok = rt.model, rt.tokenizer
    pixels, grid = native_pixels(rt, image)
    built = {}
    for name, image_only in (("positive", False), ("image", True)):
        query = _query(rt, "" if image_only else instruction, image_only)
        encoded = tok(query, return_tensors="pt")
        input_ids = encoded["input_ids"].to(rt.device)
        indexes = m.get_thw_indexes(input_ids[0], grid)
        mask = {"full_attention": create_block_causal_mask(indexes[0])}
        embeds = m.language_model.get_input_embeddings()(input_ids)
        batch, length, channels = embeds.shape
        flat = input_ids.reshape(batch * length)
        selected = flat == m.img_context_token_id
        with autocast(rt.device):
            visual = m.extract_feature(pixels, grid_hw=grid).reshape(-1, channels)
        if int(selected.sum()) != visual.shape[0]:
            raise ValueError("Condition image tokens differ from the generation canvas")
        embeds = embeds.reshape(batch * length, channels).clone()
        embeds[selected] = visual.to(embeds.dtype)
        embeds = embeds.reshape(batch, length, channels)
        with autocast(rt.device):
            cache, _ = m._it2i_prefix_forward(embeds, indexes, mask)
            gen_index = m._build_t2i_image_indexes(
                *rt.grid, int(indexes[0].max()) + 1, device=rt.device
            )
        built[name] = Prefix(cache, gen_index)
    return built


class NativeIT2I:
    """Frozen image-conditioned teacher. It never loads the editor LoRA."""

    def __init__(self, rt, instruction, image, cfg):
        self.rt, self.cfg = rt, cfg
        self.calls = 0
        built = prefixes(rt, instruction, image)
        self.positive = built["positive"]
        self.image = built["image"] if float(cfg["cfg_scale"]) > 1 else None

    def __call__(self, z, t):
        self.calls += 1
        positive = self.rt.predict(z, t, self.positive).float()
        lo, hi = [float(v) for v in self.cfg["cfg_interval"]]
        if self.image is None or not lo <= float(t) <= hi:
            return positive
        self.calls += 1
        reference = self.rt.predict(z, t, self.image).float()
        return self.rt.guided(positive, reference, float(self.cfg["cfg_scale"])).float()


def teacher_state(rt, cfg, instruction, image, noise, grid, read_time):
    """Run the teacher from ``noise`` up to ``read_time`` and return that state."""
    from .field import euler, segment_times

    velocity = NativeIT2I(rt, instruction, image, cfg)
    state = euler(noise.float(), segment_times(grid, 0.0, float(read_time)), velocity)
    return state, velocity.calls


def propose_teacher(rt, raw, cfg):
    """Offline-only target prefix for self-correction."""
    result = semantic_plan(rt, raw, cfg, system_message=TEACHER_SYSTEM,
                           question="USER REQUEST:\n" + compact(raw) + "\n\nOutput the description only.")
    text = compact(result["plan"])
    return dict(condition=text, raw_output=result["plan"],
                error=None if 6 <= len(text.split()) <= 80 else "teacher_length")
