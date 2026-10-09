"""Frozen BAGEL: persistent UND state and full prompt-KV replacement only."""
from dataclasses import dataclass
from types import MethodType

MODE = 'LAYERWISE_UND_STATE_REPLACE'
MODES = ('BASE', MODE)


@dataclass(frozen=True)
class LoopConfig:
    mode: str = MODE
    extra_rounds: int = 2
    start_layer: int = 0
    end_layer: int = 8
    memory_update: str = "legacy_layerwise"
    progress_start: float = 0.0
    progress_end: float = 1.0

    def __post_init__(self):
        if self.mode not in MODES or self.extra_rounds < 0:
            raise ValueError('only Base and persistent UND Memory loop are supported')
        if self.memory_update not in ('legacy_layerwise','full_depth','full_depth_restart'):
            raise ValueError('unknown Memory update topology')
        if not 0 <= self.start_layer < self.end_layer:
            raise ValueError('layer window must be nonempty and half-open')
        if not 0 <= self.progress_start <= self.progress_end <= 1:
            raise ValueError('sampling progress must be in [0,1]')


class InternalLoopRuntime:
    """Install an inference-only decoder wrapper; never change model weights."""
    def __init__(self, model, config=LoopConfig(), diagnostics=False, kv_observer=None):
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
        self.kv_observer = kv_observer
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
        if self.config.mode == MODE and self.config.extra_rounds > 0:
            self.layerwise.begin_prefill(cache, token_ids, lengths, special_ids, self.config)

    def end_prefill(self):
        self.layerwise.end_prefill()

    def clear_prompt_state(self):
        self.layerwise.clear()

    def _forward(self, kwargs):
        if kwargs.get('mode', 'und') == 'und' and self.layerwise.prefill is not None:
            return self.layerwise.capture_prefill(self.original, kwargs, self.config)
        cfg = self.config
        active = (kwargs.get('mode', 'und') == 'gen' and cfg.mode == MODE
                  and cfg.extra_rounds > 0
                  and cfg.progress_start <= self.progress <= cfg.progress_end)
        if not active:
            return self.original(**kwargs)
        if kwargs.get('update_past_key_values', True) or kwargs.get('is_causal', True):
            raise ValueError('GEN loop requires immutable prompt KV and noncausal attention')
        return self.layerwise.run(kwargs, self)

    def close(self):
        if self.original is not None:
            self.decoder.forward_inference = self.original
            del self.decoder._memory_loop_runtime
            self.original = None
            self.layerwise.clear()
