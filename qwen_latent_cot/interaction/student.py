"""Frozen Observe/C2 parent, trainable writer and consumer, shared native loop."""
from contextlib import contextmanager
import torch
from torch import nn
from .adapters import Adapters
from ..diagnostics.feedback_source import compose_source
from .loop import ReentryBridge, full_loop
from ..diagnostics.numerics import autocast_for


PARENT_KEYS = ('format_version', 'und', 'config', 'stage', 'step', 'topology', 'slots', 'depth', 'seed_ids', 'native_files')


class InteractionStudent(nn.Module):
    def __init__(self, runtime, parent, config):
        super().__init__()
        from ..diagnostics.record import validate_checkpoint
        validate_checkpoint(parent, runtime, ('C2-observe', 'C2-feedback'))
        if parent.get('format_version') != 4:
            raise ValueError('Interaction requires a native UND format-4 parent')
        object.__setattr__(self, 'runtime', runtime)
        object.__setattr__(self, 'parent', {key: parent[key] for key in PARENT_KEYS})
        if runtime.writer.slots != 16:
            raise ValueError('Full-loop Memory requires 16 slots')
        self.config = dict(config)
        runtime.model.eval().requires_grad_(False)
        runtime.bundle.vae.eval().requires_grad_(False)
        object.__setattr__(self, 'original_und', {name: self.native_tensor(name).detach().cpu().clone() for name in parent['und']})
        with torch.no_grad():
            for name, value in parent['und'].items():
                parameter = self.native_tensor(name)
                parameter.data = value.to(device=parameter.device).clone()
        runtime._neutral_context = None
        self.writer = runtime.writer
        self.writer.activation_checkpointing = config.get('activation_checkpointing', True)
        self.adapters = Adapters(runtime.decoder, config['writer_rank'], config['consumer_rank'], consumer_k=config.get('stage_a', False))
        self.bridge = ReentryBridge(runtime.decoder.config.hidden_size, runtime.device)
        if config.get('stage_a'):
            self.bridge.vel_scale = nn.Parameter(torch.full((config['feedback_updates'],), config['velocity_scale_init'], device=runtime.device))
            for layer in runtime.decoder.layers:
                layer._und_checkpointing = config.get('activation_checkpointing', True)
        if config['feedback_updates'] < 1 or len(config['round_loss_weights']) != config['feedback_updates']:
            raise ValueError('feedback_updates must match nonempty round_loss_weights')
        if any(w <= 0 for w in config['round_loss_weights']):
            raise ValueError('Every feedback round must receive positive supervision')

    def native_tensor(self, name):
        if name.startswith('layers.'):
            return self.runtime.decoder.get_parameter(name)
        group, suffix = name.split('.', 1)
        module = {'final_norm': self.runtime.decoder.norm, 'embeddings': self.runtime.decoder.embed_tokens,
                  'lm_head': self.runtime.model.language_model.lm_head}[group]
        return module.get_parameter(suffix)

    @contextmanager
    def native_context(self):
        """Original BAGEL prompt encoding, restored after native comparison."""
        try:
            for name, value in self.original_und.items():
                parameter = self.native_tensor(name)
                parameter.data = value.to(device=parameter.device).clone()
            yield
        finally:
            for name, value in self.parent['und'].items():
                parameter = self.native_tensor(name)
                parameter.data = value.to(device=parameter.device).clone()
            self.runtime._neutral_context = None

    def write(self, source, t, *, prompt):
        runtime = self.runtime
        source = compose_source(prompt, source)
        positions = torch.arange(runtime.config['writer_position_start'], runtime.config['writer_position_start']+self.writer.slots,
                                 device=runtime.device)
        with autocast_for(runtime.device):
            return self.writer(runtime.seeds, source, positions, source_kind='visual', t=t,
                               detach_source=False, adapter_scope=self.adapters.context('writer'))

    def forward(self, state, *, guidance=False, feedback_updates=None, memory_masked=False, reentry=True):
        return full_loop(self, state, guidance=guidance, feedback_updates=feedback_updates, memory_masked=memory_masked, reentry=reentry)

    def parameter_groups(self):
        return {name: list(module.parameters()) for name, module in (
            ('writer', self.adapters.writer), ('consumer', self.adapters.consumer), ('bridge', self.bridge))}

    def optimizer_groups(self):
        return [dict(params=parameters, lr=self.config[f'{name}_lr'], name=name)
                for name, parameters in self.parameter_groups().items()]
