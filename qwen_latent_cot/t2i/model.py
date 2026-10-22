"""Pure native GEN training. No corrector, CFG or modified regression target."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.func import functional_call
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def gen_parameter(name):
    return '_moe_gen' in name or name.startswith(('vae2llm.', 'llm2vae.', 'time_embedder.'))


def configure(model):
    model.requires_grad_(False)
    model.config.visual_und = False  # Pure T2I has no target-image understanding input.
    model.language_model.model.config.freeze_und = True
    for layer in model.language_model.model.layers:
        layer.freeze_und = True
        layer.self_attn.config.freeze_und = True
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(gen_parameter(name))
    return {name: tuple(p.shape) for name, p in model.named_parameters() if p.requires_grad}


def sequence(model, tokenizer, token_ids, prompt, height, width, device, max_length):
    """Same tokens and image positions as native prepare_prompts/prepare_vae_latent."""
    text = [] if prompt is None else [token_ids['bos_token_id']] + tokenizer.encode(prompt) + [token_ids['eos_token_id']]
    n = (height // model.latent_downsample) * (width // model.latent_downsample)
    prefix = len(text); length = prefix + n + 2
    if length > max_length:
        raise ValueError(f'Sequence has {length} tokens, exceeds {max_length}; no silent truncation')
    ids = torch.tensor(text + [token_ids['start_of_image'], token_ids['end_of_image']], device=device)
    text_indexes = torch.tensor(list(range(prefix + 1)) + [length - 1], device=device)
    image_indexes = torch.arange(prefix + 1, length - 1, device=device)
    positions = torch.tensor(list(range(prefix)) + [prefix] * (n + 2), device=device)
    # Text causal; complete image split (including boundary tokens) bidirectional.
    mask = torch.ones(length, length, device=device, dtype=torch.bool).tril()
    mask[prefix:, :] = True
    image_positions = model.get_flattened_position_ids(height, width, model.latent_downsample,
                                                      max_num_patches_per_side=model.max_latent_size).to(device)
    return dict(ids=ids, text_indexes=text_indexes, image_indexes=image_indexes,
                positions=positions, image_positions=image_positions,
                mask=torch.zeros(length,length,device=device).masked_fill(~mask,float('-inf')))


class NativeT2I(nn.Module):
    """Call original MoT blocks with a device-native differentiable SDPA mask."""
    def __init__(self, model, activation_checkpointing=True):
        super().__init__()
        self.model = model
        self.parameter_shapes = configure(model)
        model.train()
        self.activation_checkpointing = activation_checkpointing

    def forward(self, latent, inputs, logit_time, noise=None):
        model = self.model; decoder = model.language_model.model
        p = model.latent_patch_size; channels, h, w = latent.shape
        clean = latent.reshape(channels, h//p, p, w//p, p)
        clean = torch.einsum('chpwq->hwpqc', clean).reshape(-1, channels*p*p)
        epsilon = torch.randn_like(clean) if noise is None else noise
        s = torch.sigmoid(logit_time.float())
        t = model.timestep_shift*s/(1+(model.timestep_shift-1)*s)
        noisy = (1-t)*clean + t*epsilon
        hidden = model.language_model.model.embed_tokens(inputs['ids']).to(torch.bfloat16)
        packed = hidden.new_zeros(len(inputs['positions']), model.hidden_size)
        packed = packed.index_copy(0, inputs['text_indexes'], hidden)
        image_hidden = model.vae2llm(noisy.to(hidden.dtype))
        image_hidden = image_hidden + model.time_embedder(t.expand(len(noisy)))
        image_hidden = image_hidden + model.latent_pos_embed(inputs['image_positions']).to(image_hidden.dtype)
        packed = packed.index_copy(0, inputs['image_indexes'], image_hidden.to(packed.dtype))
        cos, sin = decoder.rotary_emb(packed, inputs['positions'][None])
        kwargs = dict(sample_lens=[len(packed)], attention_mask=[inputs['mask']],
                      packed_position_embeddings=(cos[0], sin[0]),
                      packed_und_token_indexes=inputs['text_indexes'],
                      packed_gen_token_indexes=inputs['image_indexes'])
        for layer in decoder.layers:
            if self.activation_checkpointing and torch.is_grad_enabled():
                packed = checkpoint(self.run_layer, layer, packed, kwargs, use_reentrant=False)
            else:
                packed = self.run_layer(layer, packed, kwargs)
        prediction = model.llm2vae(decoder.norm_moe_gen(packed[inputs['image_indexes']]))
        loss = (prediction.float() - (epsilon.float()-clean.float())).square().mean()
        return loss, prediction, t

    @staticmethod
    def run_layer(layer, packed, kwargs):
        if isinstance(layer, FSDP):
            return layer(packed_sequence=packed, **kwargs)
        # Unsharded execution retains FP32 master weights. Native indexed MoT
        # assignments require BF16 norms/projections, not only autocast linears.
        weights={n:p.to(torch.bfloat16) for n,p in layer.named_parameters()}
        return functional_call(layer, weights, (), dict(packed_sequence=packed, **kwargs))
