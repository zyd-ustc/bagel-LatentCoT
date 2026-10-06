"""Frozen denoiser loops: native layerwise KV feedback and legacy hidden control.

R counts extra body passes. State is local to a denoiser call and CFG branch.
Layerwise feedback preserves native layer depth and resets GEN to its entrance.
The explicit MEMORY_LOOP mode retains the frozen parent hidden recurrence.
"""
from dataclasses import dataclass
from types import MethodType
import torch
from .memory_attention import blocked_memory_attention
from .modeling.bagel.qwen2_navit import BaseNavitOutputWithPast

MODES = ('BASE', 'MEMORY_LOOP', 'MEMORY_NO_READ', 'LAYERWISE_MEMORY_KV', 'LAYERWISE_KV_NO_READ',
         'LAYERWISE_MEMORY_REPLACE', 'LAYERWISE_SEED_REPLACE')


@dataclass(frozen=True)
class LoopConfig:
    mode: str = 'LAYERWISE_MEMORY_KV'
    extra_rounds: int = 1
    start_layer: int = 0
    end_layer: int = 8
    memory_slots: int = 8
    progress_start: float = 0.0
    progress_end: float = 1.0
    memory_seed: int = 0

    def __post_init__(self):
        if self.mode not in MODES or self.extra_rounds < 0 or self.memory_slots < 0:
            raise ValueError('invalid mode, extra round count or memory slot count')
        if not 0 <= self.start_layer < self.end_layer:
            raise ValueError('layer window must be nonempty and half-open')
        if self.mode.startswith('LAYERWISE') and self.extra_rounds>0 and self.memory_slots>0 and self.end_layer-self.start_layer<2:
            raise ValueError('layerwise feedback needs at least two body layers')
        if not 0 <= self.progress_start <= self.progress_end <= 1:
            raise ValueError('sampling progress must be in [0,1]')


@dataclass
class MemoryLayout:
    native_indexes: torch.Tensor
    memory_indexes: torch.Tensor
    positions: torch.Tensor
    query_lengths: torch.Tensor
    query_indexes: torch.Tensor
    cache_indexes: torch.Tensor
    text_indexes: torch.Tensor
    image_indexes: torch.Tensor
    lengths: list


def memory_layout(kwargs, slots):
    """Per sample [SOI, M*K, GEN, EOI]; original GEN/UND RoPE is unchanged."""
    lengths = kwargs['query_lens'].tolist()
    past_lengths = kwargs['key_values_lens'].tolist()
    positions = kwargs['packed_query_position_ids']
    native, memory, query, cached, pos = [], [], [], [], []
    old_offset = query_offset = merged_offset = 0
    for length, past in zip(lengths, past_lengths):
        if length < 2:
            raise ValueError('native image query requires SOI/EOI boundaries')
        native.append(query_offset)
        memory.extend(range(query_offset + 1, query_offset + 1 + slots))
        native.extend(range(query_offset + 1 + slots, query_offset + length + slots))
        pos.extend([positions[old_offset:old_offset+1],
                    positions[old_offset:old_offset+1].expand(slots),
                    positions[old_offset+1:old_offset+length]])
        cached.extend(range(merged_offset, merged_offset + past))
        query.extend(range(merged_offset + past, merged_offset + past + length + slots))
        old_offset += length
        query_offset += length + slots
        merged_offset += past + length + slots
    if old_offset != len(kwargs['packed_query_sequence']):
        raise ValueError('query lengths do not cover the packed sequence')
    ids = lambda values: positions.new_tensor(values, dtype=torch.long)
    native, memory = ids(native), ids(memory)
    return MemoryLayout(native, memory, torch.cat(pos), kwargs['query_lens'] + slots,
                        ids(query), ids(cached),
                        torch.cat([native[kwargs['packed_text_indexes']], memory]),
                        native[kwargs['packed_vae_token_indexes']], [slots]*len(lengths))


def initial_memory(hidden, lengths, slots, seed=0):
    """Legacy boundary mean + 1e-4 slot noise, initialized separately per sample."""
    noise = torch.randn((slots, hidden.shape[-1]), generator=torch.Generator(device='cpu').manual_seed(seed))
    # Parent adds noise in the native hidden dtype. No parameters are created.
    noise = (1e-4 * noise.to(hidden))
    states = []
    offset = 0
    for length in lengths:
        boundary = hidden[[offset, offset + length - 1]].float().mean(0).to(hidden.dtype)
        states.append(boundary.unsqueeze(0) + noise)
        offset += length
    return torch.cat(states)


class InternalLoopRuntime:
    """Wrap the native decoder entry; restore it on close. No weight mutation."""
    def __init__(self, model, config=LoopConfig(), diagnostics=False, probe_capture=None):
        self.model, self.config = model, config
        self.decoder = model.language_model.model
        if config.end_layer > len(self.decoder.layers):
            raise ValueError('loop window exceeds native decoder depth')
        if not self.decoder.use_moe or getattr(self.decoder, 'enable_taylorseer', False):
            raise ValueError('Memory loop requires native MoT without TaylorSeer')
        if self.decoder.training:
            raise ValueError('this runtime implements frozen training-free inference only')
        if getattr(self.decoder, '_memory_loop_runtime', None) is not None:
            raise ValueError('a Memory runtime is already installed')
        self.progress = 0.0
        self.step_index = 0
        self.probe_capture = probe_capture
        self.diagnostics_enabled = diagnostics
        self.diagnostics = []
        from .layerwise_memory import LayerwiseMemoryLoop
        self.layerwise = LayerwiseMemoryLoop(self.decoder)
        self.original = self.decoder.forward_inference
        self.decoder._memory_loop_runtime = self
        def wrapped(this, **kwargs):
            return self._forward(kwargs)
        self.decoder.forward_inference = MethodType(wrapped, self.decoder)

    def begin_prefill(self, cache, token_ids, lengths, special_ids):
        if self.config.mode.startswith('LAYERWISE') and self.config.extra_rounds>0 and self.config.memory_slots>0:
            self.layerwise.begin_prefill(cache, token_ids, lengths, special_ids, self.config)

    def end_prefill(self):
        self.layerwise.end_prefill()

    def clear_prompt_state(self):
        self.layerwise.clear()

    def _forward(self, kwargs):
        cfg = self.config
        if kwargs.get('mode','und')=='und' and self.layerwise.prefill is not None:
            return self.layerwise.capture_prefill(self.original, kwargs, cfg)
        active = (kwargs.get('mode', 'und') == 'gen' and cfg.mode != 'BASE'
                  and cfg.extra_rounds > 0 and cfg.memory_slots > 0
                  and cfg.progress_start <= self.progress <= cfg.progress_end)
        if not active:
            return self.original(**kwargs)
        if kwargs.get('update_past_key_values', True) or kwargs.get('is_causal', True):
            raise ValueError('GEN loop requires immutable prompt KV and noncausal attention')
        if cfg.mode.startswith('LAYERWISE'):
            return self.layerwise.run(kwargs, self)
        return self._memory_forward(kwargs)

    def _memory_forward(self, kwargs):
        cfg, decoder = self.config, self.decoder
        layout = memory_layout(kwargs, cfg.memory_slots)
        native_hidden = kwargs['packed_query_sequence']
        hidden = native_hidden.new_empty((len(native_hidden) + len(layout.memory_indexes), native_hidden.shape[-1]))
        hidden[layout.native_indexes] = native_hidden
        hidden[layout.memory_indexes] = initial_memory(native_hidden, kwargs['query_lens'].tolist(), cfg.memory_slots, cfg.memory_seed)
        cos, sin = decoder.rotary_emb(hidden, layout.positions.unsqueeze(0))
        rope = (cos.squeeze(0), sin.squeeze(0))
        layer_kwargs = dict(query_lens=layout.query_lengths, packed_query_position_embeddings=rope,
                            packed_query_indexes=layout.query_indexes, past_key_values=kwargs['past_key_values'],
                            key_values_lens=kwargs['key_values_lens'], packed_key_value_indexes=layout.cache_indexes,
                            update_past_key_values=False, is_causal=False, mode='gen',
                            packed_vae_token_indexes=layout.image_indexes, packed_text_indexes=layout.text_indexes)
        native_layer_kwargs = {k:v for k,v in kwargs.items() if k != 'packed_query_position_ids'}
        native_layer_kwargs['packed_query_position_embeddings'] = tuple(r[layout.native_indexes] for r in rope)
        initial_kv = {}
        capturing = (self.probe_capture is not None and self.step_index in self.probe_capture.steps
                     and int(kwargs['key_values_lens'].sum()) > 0)
        if capturing and len(layout.lengths) != 1:
            raise ValueError('probe export is batch=1; generation supports packed variable lengths')
        from .memory_stats import project_memory, memory_slot_stats

        def run_layer(index, state, blocked, round_index=None):
            layer = decoder.layers[index]
            if capturing and round_index in (0, cfg.extra_rounds):
                _, mk, mv = project_memory(layer, state[layout.memory_indexes],
                                           tuple(r[layout.memory_indexes] for r in rope), query=False)
                if round_index == 0:
                    initial_kv[index] = (mk.detach().clone(), mv.detach().clone())
                else:
                    ik, iv = initial_kv[index]
                    self.probe_capture.record(self.step_index, index, {
                        'dynamic_k':mk, 'dynamic_v':mv, 'seed_k':ik, 'seed_v':iv,
                        'lengths':layout.lengths, 'question_position_start':int(layout.positions.max()) + 1,
                        'source_indexes':layout.memory_indexes, 'read_round':round_index})
            native_result = None
            if cfg.mode == 'MEMORY_NO_READ':
                # Use the exact native non-Memory calculation. This closes all
                # relay paths and removes masked-kernel drift from this control.
                native_result, _ = layer.forward_inference(**{
                    **native_layer_kwargs, 'packed_query_sequence':state[layout.native_indexes]})
            original_attention = layer.self_attn.forward_inference
            if blocked:
                def masked(this, **attention_kwargs):
                    return blocked_memory_attention(this, **attention_kwargs,
                        packed_memory_token_indexes=layout.memory_indexes, block_gen_reads_memory=True)
                layer.self_attn.forward_inference = MethodType(masked, layer.self_attn)
            try:
                result, _ = layer.forward_inference(packed_query_sequence=state, **layer_kwargs)
            finally:
                if blocked:
                    layer.self_attn.forward_inference = original_attention
            if native_result is not None:
                result[layout.native_indexes] = native_result
            if self.diagnostics_enabled:
                before = state[layout.memory_indexes].split(layout.lengths)
                after = result[layout.memory_indexes].split(layout.lengths)
                branch = 'conditional' if int(kwargs['key_values_lens'].sum()) else 'text_removed'
                for sample, (a, b) in enumerate(zip(before, after)):
                    self.diagnostics.append({'layer':index, 'round':round_index, 'sample':sample,
                        'phase':'prefix' if index < cfg.start_layer else 'suffix' if index >= cfg.end_layer else 'body',
                        'branch':branch, 'progress':self.progress, 'gen_reads_memory':not blocked,
                        'update_ratio':float((b-a).float().norm()/a.float().norm().clamp_min(1e-12)),
                        **memory_slot_stats(b)})
            return result

        # Strict Read also covers prefix; Memory traverses its native depth.
        for index in range(cfg.start_layer):
            hidden = run_layer(index, hidden, True)
        entry = hidden.clone()
        memory = entry[layout.memory_indexes]
        for round_index in range(cfg.extra_rounds + 1):
            if round_index:
                hidden = entry.clone()
                hidden[layout.memory_indexes] = memory
            blocked = round_index == 0 or cfg.mode == 'MEMORY_NO_READ'
            for index in range(cfg.start_layer, cfg.end_layer):
                hidden = run_layer(index, hidden, blocked, round_index)
            memory = hidden[layout.memory_indexes]
        # Parent semantics: suffix reads final Memory and executes once.
        for index in range(cfg.end_layer, len(decoder.layers)):
            hidden = run_layer(index, hidden, cfg.mode == 'MEMORY_NO_READ')
        result = hidden[layout.native_indexes]
        normalized = torch.zeros_like(result)
        text, image = kwargs['packed_text_indexes'], kwargs['packed_vae_token_indexes']
        normalized[text] = decoder.norm(result[text])
        normalized[image] = decoder.norm_moe_gen(result[image])
        return BaseNavitOutputWithPast(packed_query_sequence=normalized, past_key_values=kwargs['past_key_values'])

    def close(self):
        if self.original is not None:
            self.decoder.forward_inference = self.original
            del self.decoder._memory_loop_runtime
            self.original = None
            self.layerwise.clear()
