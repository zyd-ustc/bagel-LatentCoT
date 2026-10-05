"""Strict loading of the original BAGEL checkpoint, with no historical modules."""
from dataclasses import dataclass
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import Qwen2Tokenizer
from accelerate import init_empty_weights
from .modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from .modeling.autoencoder import load_ae


@dataclass
class NativeBundle:
    model: object
    vae: object
    tokenizer: object
    token_ids: dict
    model_path: str


def load_native(model_path, device='cuda:0', timestep_shift=3.0):
    path = Path(model_path)
    llm_cfg = Qwen2Config.from_json_file(path/'llm_config.json')
    llm_cfg.pad_token_id = getattr(llm_cfg, 'pad_token_id', None)
    llm_cfg.qk_norm = True; llm_cfg.tie_word_embeddings = False
    llm_cfg.layer_module = 'Qwen2MoTDecoderLayer'
    vit_cfg = SiglipVisionConfig.from_json_file(path/'vit_config.json')
    vit_cfg.rope = False; vit_cfg.num_hidden_layers -= 1
    vae, vae_cfg = load_ae(str(path/'ae.safetensors'))
    cfg = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm_cfg,
        vit_config=vit_cfg, vae_config=vae_cfg, vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh', latent_patch_size=2, max_latent_size=64,
        timestep_shift=timestep_shift)
    print('Constructing native architecture on meta parameters...', flush=True)
    with init_empty_weights(include_buffers=False):
        model = Bagel(Qwen2ForCausalLM(llm_cfg), SiglipVisionModel(vit_cfg), cfg)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_cfg, meta=False)
    model.language_model.model.enable_taylorseer = False
    if (path/'ema.safetensors').exists():
        state = load_file(str(path/'ema.safetensors'))
    else:
        state = {}
        for shard in sorted(path.glob('*.safetensors')):
            if shard.name == 'ae.safetensors': continue
            chunk = load_file(str(shard))
            if state.keys() & chunk.keys(): raise ValueError('duplicate native checkpoint keys')
            state.update(chunk)
    print('Loading every native checkpoint tensor strictly...', flush=True)
    vocab = state['language_model.model.embed_tokens.weight'].shape[0]
    if model.language_model.model.embed_tokens.num_embeddings != vocab:
        model.language_model.resize_token_embeddings(vocab, mean_resizing=False)
    model.load_state_dict(state, strict=True, assign=True)
    model.to(torch.bfloat16)
    del state
    tokenizer = Qwen2Tokenizer.from_pretrained(path)
    ids = {}
    for name, token in {'bos_token_id':'<|im_start|>', 'eos_token_id':'<|im_end|>',
                        'start_of_image':'<|vision_start|>', 'end_of_image':'<|vision_end|>'}.items():
        values = tokenizer.encode(token, add_special_tokens=False)
        if len(values) != 1: raise ValueError(f'non-native tokenizer token: {token}')
        ids[name] = values[0]
    model.eval().requires_grad_(False).to(device)
    vae.eval().requires_grad_(False).to(device)
    return NativeBundle(model, vae, tokenizer, ids, str(path.resolve()))
