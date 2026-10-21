"""Frozen Observe/C2 parent, trainable writer and consumer, shared native loop."""
from contextlib import contextmanager
import torch
from torch import nn
from .adapters import Adapters
from ..diagnostics.feedback_source import compose_source
from .loop import ReentryBridge, CanvasReentryBridge, full_loop
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
        self.adapters = Adapters(runtime.decoder, config['writer_rank'], config['consumer_rank'], stage_a=config.get('stage_a', False))
        self.bridge = (CanvasReentryBridge(runtime.decoder.config.hidden_size, config['feedback_updates'],
                                           len(runtime.decoder.layers), runtime.device) if config.get('stage_a')
                       else ReentryBridge(runtime.decoder.config.hidden_size, runtime.device))
        if config.get('stage_a'):
            if config['feedback_updates'] != 1:
                raise ValueError('Stage A v2 trains one correction; override updates at inference')
            if not 0 <= config['teacher_mix'] <= 1 or config['cos_weight'] < 0 or config['raw_correction_weight'] <= 0:
                raise ValueError('Invalid Stage A v2 correction loss coefficients')
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
        """Bind original UND parameters without changing tensors saved for backward."""
        bindings = []
        try:
            for name, value in self.original_und.items():
                if name.startswith('layers.'):
                    module_name, parameter_name = name.rsplit('.', 1)
                    module = self.runtime.decoder.get_submodule(module_name)
                    # ScopedLinear stores the original Parameter in base.
                    if hasattr(module, 'base'):
                        module = module.base
                else:
                    group, suffix = name.split('.', 1)
                    module = {'final_norm': self.runtime.decoder.norm,
                              'embeddings': self.runtime.decoder.embed_tokens,
                              'lm_head': self.runtime.model.language_model.lm_head}[group]
                    module_name, _, parameter_name = suffix.rpartition('.')
                    if module_name:
                        module = module.get_submodule(module_name)
                current = module._parameters[parameter_name]
                bindings.append((module, parameter_name, current))
                module._parameters[parameter_name] = nn.Parameter(value.to(current.device), requires_grad=False)
            yield
        finally:
            for module, name, current in reversed(bindings):
                module._parameters[name] = current
            self.runtime._neutral_context = None

    def write(self, source, t, *, prompt):
        runtime = self.runtime
        source = compose_source(prompt, source)
        positions = torch.arange(runtime.config['writer_position_start'], runtime.config['writer_position_start']+self.writer.slots,
                                 device=runtime.device)
        with autocast_for(runtime.device):
            return self.writer(runtime.seeds, source, positions, source_kind='visual', t=t,
                               detach_source=False, adapter_scope=self.adapters.context('writer'))

    def forward(self, state, *, guidance=False, feedback_updates=None, memory_masked=False, reentry=True, consumer_enabled=True):
        return full_loop(self, state, guidance=guidance, feedback_updates=feedback_updates, memory_masked=memory_masked, reentry=reentry, consumer_enabled=consumer_enabled)

    def parameter_groups(self):
        return {name: list(module.parameters()) for name, module in (
            ('writer', self.adapters.writer), ('consumer', self.adapters.consumer), ('bridge', self.bridge))}

    def optimizer_groups(self):
        return [dict(params=parameters, lr=self.config[f'{name}_lr'], name=name)
                for name, parameters in self.parameter_groups().items()]
