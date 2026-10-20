"""Native UND parameter selection and full-depth memory writing."""
import torch
from torch import nn
from .memory import MemoryWriter
from .numerics import autocast_for


UND_GROUPS = {
    'q_proj': 'self_attn.q_proj', 'k_proj': 'self_attn.k_proj',
    'v_proj': 'self_attn.v_proj', 'o_proj': 'self_attn.o_proj',
    'q_norm': 'self_attn.q_norm', 'k_norm': 'self_attn.k_norm',
    'input_layernorm': 'input_layernorm',
    'post_attention_layernorm': 'post_attention_layernorm',
    'mlp_gate_proj': 'mlp.gate_proj', 'mlp_up_proj': 'mlp.up_proj',
    'mlp_down_proj': 'mlp.down_proj',
}


class VisualStudent(nn.Module):
    """Register selected original UND parameters; keep the native model shared."""
    architecture = 'native_und_v1'

    def __init__(self, runtime, text_teacher=None):
        super().__init__()
        object.__setattr__(self, 'runtime', runtime)
        self.writer = runtime.writer
        self.writer.activation_checkpointing = runtime.config.get('activation_checkpointing', True)
        # The readout uses the same original UND weights as the writer.
        for layer in runtime.decoder.layers:
            object.__setattr__(layer, '_und_checkpointing', self.writer.activation_checkpointing)
        controls = runtime.config['und_train']
        layers = controls.get('layers', 'all')
        selected = range(len(runtime.decoder.layers)) if layers == 'all' else layers
        parameters = {}; groups = {}
        for index in selected:
            layer = runtime.decoder.layers[index]
            for group, path in UND_GROUPS.items():
                if controls.get(group, False):
                    for name, parameter in layer.get_submodule(path).named_parameters():
                        key = f'layers.{index}.{path}.{name}'
                        parameters[key] = parameter; groups[key] = group
        for group, module in (('final_norm', runtime.decoder.norm),
                              ('embeddings', runtime.decoder.embed_tokens),
                              ('lm_head', runtime.model.language_model.lm_head)):
            if controls.get(group, False):
                for name, parameter in module.named_parameters():
                    key = f'{group}.{name}'
                    parameters[key] = parameter; groups[key] = group
        if not parameters:
            raise ValueError('und_train must select original UND parameters')
        self.native_names = tuple(parameters)
        self.checkpoint_names = list(parameters)
        self.parameter_group_names = groups
        # Optimize original tensors in float32; native mixed precision handles forward.
        for parameter in parameters.values():
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
        self.und = nn.ParameterList(parameters.values())
        object.__setattr__(self, 'text_teacher', text_teacher)
        self.text_teacher_hash = None

    def native_parameters(self):
        return dict(zip(self.native_names, self.und))

    def native_tensor(self, name):
        decoder = self.runtime.decoder
        if name.startswith('layers.'):
            return decoder.get_parameter(name)
        group, suffix = name.split('.', 1)
        module = {'final_norm': decoder.norm, 'embeddings': decoder.embed_tokens,
                  'lm_head': self.runtime.model.language_model.lm_head}[group]
        return module.get_parameter(suffix)

    def native_state_dict(self):
        return {name: self.native_tensor(name).detach().cpu() for name in self.checkpoint_names}

    def load_native_state_dict(self, state):
        with torch.no_grad():
            for name, tensor in state.items():
                parameter = self.native_tensor(name)
                parameter.copy_(tensor.to(parameter.device, parameter.dtype))
                if name not in self.checkpoint_names:
                    self.checkpoint_names.append(name)
        self.runtime._neutral_context = None

    def write(self, source, t, *, prompt=None, source_off=False, visual=True,
              detach_source=True, source_layers=None):
        runtime = self.runtime
        from .feedback_source import compose_source
        if prompt is not None:
            source = prompt if source_off else compose_source(prompt, source, source_layers)
            source_off = False
            source_layers = None
        positions = torch.arange(runtime.config['writer_position_start'],
                                 runtime.config['writer_position_start']+self.writer.slots,
                                 device=runtime.device)
        with autocast_for(runtime.device):
            return self.writer(runtime.seeds, source, positions, source_off,
                               source_kind='visual' if visual else 'text', t=t,
                               detach_source=detach_source, source_layers=source_layers)

    @torch.no_grad()
    def teacher_bank(self, caption, state=None):
        runtime = self.runtime
        if self.text_teacher is None:
            raise ValueError('text teacher is not configured')
        if state is not None:
            from .feedback_source import prompt_source, write_feedback
            from .conditioning import correction_source
            flow, lengths, positions = runtime.flow_inputs(state.metadata['prompt'])
            correction = correction_source(runtime, flow, lengths, positions, caption)[0]
            return write_feedback(runtime, prompt_source(flow, lengths, positions),
                                  correction, writer=self.text_teacher)
        source = runtime.text(caption)[0]
        positions = torch.arange(runtime.config['writer_position_start'],
                                 runtime.config['writer_position_start']+self.writer.slots, device=runtime.device)
        with autocast_for(runtime.device):
            return self.text_teacher(runtime.seeds, source, positions)

    def parameter_groups(self):
        groups = {}
        for name, parameter in self.native_parameters().items():
            groups.setdefault(self.parameter_group_names[name], []).append(parameter)
        return groups


def validate_config(runtime, stage):
    cfg = runtime.config
    if stage not in ('C2-observe', 'C2-feedback'):
        raise ValueError('select C2-observe/C2-feedback')
    if cfg['slots'] != 16 or runtime.writer.topology != 'C':
        raise ValueError('feedback uses continuous UND and 16 slots')
    if cfg.get('writer_architecture') != VisualStudent.architecture:
        raise ValueError('select native_und_v1')
    if cfg.get('source_layers') != list(range(len(runtime.decoder.layers))):
        raise ValueError('all UND layers must read same-layer GEN KV')
    controls = cfg['und_train']
    allowed = set(UND_GROUPS) | {'layers', 'final_norm', 'embeddings', 'lm_head'}
    if set(controls)-allowed:
        raise ValueError('unknown und_train controls: '+str(set(controls)-allowed))
    layers = controls.get('layers', 'all')
    if layers != 'all' and (not layers or len(set(layers)) != len(layers) or
                           any(not 0 <= i < len(runtime.decoder.layers) for i in layers)):
        raise ValueError('invalid UND training layer selection')
    if set(cfg['loss_weights']) != {'sem','teacher','flow'} or any(w<0 for w in cfg['loss_weights'].values()):
        raise ValueError('declare sem/teacher/flow weights')
    if stage in ('C2-observe', 'C2-feedback') and (cfg['loss_weights']['teacher'] or cfg['loss_weights']['flow']):
        raise ValueError('C2 Observe/Feedback use text supervision')
    if not cfg.get('noise_curriculum') or not cfg.get('noise_calibration'):
        raise ValueError('declare noise curriculum and calibration')
    if any(cfg[k] <= 0 for k in ('effective_batch','steps','eval_every')):
        raise ValueError('positive training budgets required')
    if cfg.get('feedback_updates',1) not in (1,2):
        raise ValueError('one or two feedback updates required')
