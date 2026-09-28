"""Frozen, prompt-manifold initialization of Phase 1A memory slots."""

import torch


@torch.no_grad()
def initialize_memory_from_prompt_hidden(
    *, hidden_cache, prompt_mask, layer_index, num_slots,
    content_mask=None, strategy="uniform_content",
):
    """Select real prompt-content layer-entry rows, returning [B*K, D].

    ``hidden_cache`` is either the tensor captured at ``layer_index`` or a
    layer-indexed mapping. BOS/EOS/template/punctuation exclusion is supplied
    via ``content_mask``; silently falling back to such rows is forbidden.
    """
    if strategy != "uniform_content":
        raise ValueError("Phase 1A T0 requires uniform_content memory initialization")
    if isinstance(hidden_cache, dict):
        if layer_index not in hidden_cache:
            raise ValueError(f"no captured prompt hidden at layer {layer_index}")
        hidden = hidden_cache[layer_index]
    else:
        hidden = hidden_cache
    if hidden.ndim != 3 or num_slots < 1:
        raise ValueError("expected [B,L,D] hidden and positive num_slots")
    mask = torch.as_tensor(prompt_mask, device=hidden.device, dtype=torch.bool)
    if mask.shape != hidden.shape[:2]:
        raise ValueError("prompt_mask must match [B,L]")
    if content_mask is not None:
        content = torch.as_tensor(content_mask, device=hidden.device, dtype=torch.bool)
        if content.shape != mask.shape:
            raise ValueError("content_mask must match [B,L]")
        mask = mask & content
    rows = []
    for batch in range(hidden.shape[0]):
        eligible = torch.where(mask[batch])[0]
        length = int(eligible.numel())
        if length == 0:
            raise ValueError("prompt has no eligible content tokens")
        if length >= num_slots:
            locations = torch.linspace(0, length - 1, num_slots, device=hidden.device)
            selected = eligible[locations.round().long()]
        else:
            selected = eligible[torch.arange(num_slots, device=hidden.device) % length]
        slots = hidden[batch, selected].detach().clone()
        if length < num_slots:
            # Deterministic, tiny symmetry breaker. At BF16 precision it can
            # round away; it is never allowed to change the native hidden scale.
            position = torch.arange(num_slots * hidden.shape[-1], device=hidden.device,
                                    dtype=torch.float32).reshape(num_slots, -1)
            slots = (slots.float() + 1e-5 * torch.sin(position + batch)).to(hidden.dtype)
        rows.append(slots)
    return torch.stack(rows).reshape(-1, hidden.shape[-1]).detach()


def prompt_content_mask(token_ids, tokenizer, *, special_ids=()):
    """Reject BAGEL boundary IDs and tokens consisting only of punctuation."""
    specials = set(int(value) for value in special_ids)
    return [int(token) not in specials and any(char.isalnum() for char in
            tokenizer.decode([int(token)])) for token in token_ids]
