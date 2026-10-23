"""Native BAGEL ViT-UND reader: image + question -> short answer."""
import torch
from PIL import Image

from ..correction.numerics import autocast_for
from ..bagel.modeling.image_transforms import ImageTransform


def _to_device(value, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in value.items()}


class UNDReader:
    """Wraps the frozen understanding path (ViT prefill + text prefill + greedy decode)."""

    def __init__(self, bundle, template='{question}\nAnswer:', max_answer_tokens=8,
                 max_image_size=840, min_image_size=224, image_stride=14):
        self.bundle = bundle
        self.model = bundle.model
        self.tokenizer = bundle.tokenizer
        self.ids = bundle.token_ids
        self.device = next(self.model.parameters()).device
        self.template = template
        self.max_answer_tokens = int(max_answer_tokens)
        self.transform = ImageTransform(max_image_size=max_image_size,
                                        min_image_size=min_image_size,
                                        image_stride=image_stride)

    @torch.no_grad()
    def ask(self, image: Image.Image, question: str) -> str:
        from ..bagel.modeling.bagel.qwen2_navit import NaiveCache
        prompt = self.template.format(question=question)
        cache = NaiveCache(self.model.config.llm_config.num_hidden_layers)
        newlens, newrope = [0], [0]

        inputs, newlens, newrope = self.model.prepare_vit_images(
            curr_kvlens=newlens, curr_rope=newrope, images=[image],
            transforms=self.transform, new_token_ids=self.ids)
        with autocast_for(self.device):
            cache = self.model.forward_cache_update_vit(cache, **_to_device(inputs, self.device))

        inputs, newlens, newrope = self.model.prepare_prompts(
            curr_kvlens=newlens, curr_rope=newrope, prompts=[prompt],
            tokenizer=self.tokenizer, new_token_ids=self.ids)
        with autocast_for(self.device):
            cache = self.model.forward_cache_update_text(cache, **_to_device(inputs, self.device))

        inputs = self.model.prepare_start_tokens(newlens, newrope, self.ids)
        with autocast_for(self.device):
            tokens = self.model.generate_text(
                past_key_values=cache, max_length=self.max_answer_tokens,
                do_sample=False, end_token_id=self.ids['eos_token_id'],
                **_to_device(inputs, self.device))
        text = self.tokenizer.decode(tokens[:, 0])
        return text.split('<|im_end|>')[0].split('<|im_start|>')[-1].strip()

    def score(self, image, vqa_list):
        return [self.ask(image, question) for question, _ in vqa_list]
