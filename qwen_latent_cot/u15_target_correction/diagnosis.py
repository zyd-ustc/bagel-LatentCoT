"""Three-step UND reflection over the actual preview image."""

import torch
from .runtime import autocast

def _image_inputs(rt, image):
    """The image tag and understanding-side pixels for one protocol canvas."""
    m, tok = rt.model, rt.tokenizer
    m.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    m.img_start_token_id = tok.convert_tokens_to_ids("<img>")
    mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    rgb = ((image.detach().float().clamp(-1, 1) + 1) / 2 - mean) / std
    gh, gw = rt.height // m.patch_size, rt.width // m.patch_size
    grid = torch.tensor([[gh, gw]], device=rt.device)
    pixels = (
        m.patchify(rgb, m.patch_size, channel_first=True)
        .reshape(gh * gw, -1)
        .to(m.vision_model.embeddings.patch_embedding.weight.dtype)
    )
    tag = "<img>" + "<IMG_CONTEXT>" * rt.tokens + "</img>"
    return grid, pixels, tag


@torch.no_grad()
def ask_with_image(rt, image, question, cfg, system_message=None, max_new_tokens=None):
    """Text answer conditioned on one image; same canvas and normalization as scoring."""
    from sensenova_u1.models.neo_unify.conversation import get_conv_template

    m, tok = rt.model, rt.tokenizer
    grid, pixels, img_tag = _image_inputs(rt, image)
    query = m._build_t2i_query(
        img_tag + "\n" + question,
        system_message=system_message,
        append_text="<think>\n\n</think>\n\n",
    )
    encoded = tok(query, return_tensors="pt")
    separator = get_conv_template(m.template).sep.strip()
    eos = tok.convert_tokens_to_ids(separator)
    with autocast(rt.device):
        vit = m.extract_feature(pixels, gen_model=False, grid_hw=grid)
        output = m.generate(
            pixel_values=pixels,
            input_ids=encoded["input_ids"].to(rt.device),
            grid_hw=grid,
            attention_mask=encoded["attention_mask"].to(rt.device),
            visual_features=vit,
            max_new_tokens=int(max_new_tokens or cfg["und_max_new_tokens"]),
            do_sample=False,
            eos_token_id=eos,
            pad_token_id=eos,
        )
    text = tok.batch_decode(output, skip_special_tokens=True)[0].split(separator)[0].strip()
    return text.rsplit("</think>", 1)[-1].strip()


REFLECT_SYSTEM = (
    "You are a careful picture inspector. Do three things in order and stop as "
    "soon as one of them settles the matter.\n"
    "1. Describe the picture completely: every object, how many of each, their "
    "colours and materials, and where each group sits. Count the instances one by "
    "one; do not estimate. State the count you reached.\n"
    "2. Compare that description with the target prompt and answer with a single "
    "word, yes or no, to the question: does the picture already satisfy every "
    "requirement in the prompt?\n"
    "3. Only if the answer is no, write ONE executable instruction in the form a "
    "retoucher can follow literally. It must name the operation (add, remove, move, "
    "recolour, replace, rearrange), the exact location ('the two rightmost in the "
    "middle row'), the quantity that changes, and the count before and after "
    "('remove 2 so that 9 becomes 7'). Never write only the final count: name which "
    "instances change. Keep the wording plain and specific."
)


REFLECT_TEMPLATE = (
    "<figcaption>picture</figcaption>\n"
    "Target prompt: {prompt}\n\n"
    "Reply with exactly these four lines and nothing else:\n"
    "DESCRIPTION: <complete description with counts>\n"
    "MATCH: <yes or no>\n"
    "INSTRUCTION: <one executable instruction, or NONE when MATCH is yes>\n"
    "PRESERVE: <short phrase naming what must stay unchanged>"
)


def parse_reflection(text):
    """Three labelled lines into a record; tolerant of stray formatting."""
    fields = {}
    for key in ("description", "match", "instruction", "preserve"):
        marker = key.upper() + ":"
        start = text.upper().find(marker)
        if start < 0:
            continue
        rest = text[start + len(marker):]
        stop = len(rest)
        for other in ("DESCRIPTION:", "MATCH:", "INSTRUCTION:", "PRESERVE:"):
            position = rest.upper().find(other)
            if 0 <= position < stop:
                stop = position
        fields[key] = rest[:stop].strip()
    if fields.get("match", "").strip().lower() not in ("yes", "no"):
        return dict(description=fields.get("description", ""), match=None,
                    instruction="", preserve=fields.get("preserve", ""),
                    raw=text, outcome="failed_parse")
    verdict = fields["match"].strip().lower()
    matched = verdict.startswith("yes")
    instruction = fields.get("instruction", "").strip()
    if matched:
        instruction = ""
    elif instruction.upper().startswith("NONE"):
        instruction = ""
    return dict(description=fields.get("description", ""), match=matched,
                instruction=instruction, preserve=fields.get("preserve", ""),
                raw=text, outcome=("satisfied" if matched else
                                   ("edit" if instruction else "failed_unconverted")))


def review_reflection(rt, image, raw, cfg):
    """Describe, then judge the prompt, then write one executable instruction."""
    response = ask_with_image(rt, image, REFLECT_TEMPLATE.format(prompt=raw), cfg,
                              system_message=REFLECT_SYSTEM,
                              max_new_tokens=int(cfg.get("reflect_max_new_tokens", 512)))
    parsed = parse_reflection(response)
    return dict(questions=[], decomposition={"response": response}, preview_score={"questions": []},
                failed=[], rejected=[], edit_request=parsed["instruction"],
                preserve=parsed["preserve"], atomic_edit=None,
                reflection=parsed, diagnosis_outcome=parsed["outcome"])
