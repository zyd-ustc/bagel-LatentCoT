"""Frozen Observe/C2 parent, trainable writer and consumer, shared native loop."""
import torch
from torch import nn
from .adapters import Adapters
from ..diagnostics.feedback_source import compose_source, prompt_source
from ..diagnostics.flow_training import conditional_gen
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
        self.config = dict(config)
        runtime.model.eval().requires_grad_(False)
        runtime.bundle.vae.eval().requires_grad_(False)
        with torch.no_grad():
            for name, value in parent['und'].items():
                parameter = self.native_tensor(name)
                parameter.data = value.to(device=parameter.device).clone()
        runtime._neutral_context = None
        self.writer = runtime.writer
        self.writer.activation_checkpointing = config.get('activation_checkpointing', True)
        self.adapters = Adapters(runtime.decoder, config['rank'], config['alpha'])
        self.adapters_enabled = True
        self.consumer_mode = 'policy'

    def native_tensor(self, name):
        if name.startswith('layers.'):
            return self.runtime.decoder.get_parameter(name)
        group, suffix = name.split('.', 1)
        module = {'final_norm': self.runtime.decoder.norm, 'embeddings': self.runtime.decoder.embed_tokens,
                  'lm_head': self.runtime.model.language_model.lm_head}[group]
        return module.get_parameter(suffix)

    def consumer_context(self, mode=None):
        return self.adapters.context('consumer', (mode or self.consumer_mode) if self.adapters_enabled else 'off')

    def write(self, source, t, *, prompt=None, source_off=False, detach_source=True, source_layers=None):
        runtime = self.runtime
        if prompt is not None:
            source = prompt if source_off else compose_source(prompt, source, source_layers)
            source_off, source_layers = False, None
        positions = torch.arange(runtime.config['writer_position_start'], runtime.config['writer_position_start']+self.writer.slots,
                                 device=runtime.device)
        with autocast_for(runtime.device):
            return self.writer(runtime.seeds, source, positions, source_off, source_kind='visual', t=t,
                               detach_source=detach_source, source_layers=source_layers,
                               adapter_scope=self.adapters.context('writer', 'policy' if self.adapters_enabled else 'off'))

    def proposal(self, state):
        with torch.no_grad():
            return conditional_gen(self.runtime, state)

    def forward(self, state, *, guidance=False, mode='policy', memory_mode='correct', donor_bank=None):
        proposal = self.proposal(state)
        prompt = state.prompt_kv if state.prompt_kv is not None else prompt_source(state.flow, state.lengths, state.positions)
        bank = self.write(proposal.source, state.t, prompt=prompt)
        if memory_mode == 'swap':
            if donor_bank is None:
                raise ValueError('Swap requires a same-prompt/time donor bank')
            bank = donor_bank
        elif memory_mode not in ('correct', 'masked'):
            raise ValueError(memory_mode)
        output = conditional_gen(self.runtime, state, bank, capture=False,
                                 masked=memory_mode == 'masked', guidance=guidance,
                                 checkpoint_activations=self.config.get('activation_checkpointing', True),
                                 adapter_scope=self.consumer_context(mode))
        return output, proposal, bank

    def parameter_groups(self):
        return {name: [p for p in getattr(self.adapters, name).parameters() if p.requires_grad]
                for name in ('writer', 'consumer')}

    def optimizer_groups(self, stage='sft'):
        return [dict(params=parameters, lr=self.config[f'{name}_lr'], name=name)
                for name, parameters in self.parameter_groups().items() if parameters]

    def begin_rl(self):
        self.adapters.freeze_writer()
        self.adapters.snapshot_reference()
