"""Time-conditioned UND queries. GEN keys, RoPE and all native weights stay native."""
import math
import torch
from torch import nn
from .memory import MemoryWriter
from .numerics import autocast_for


class VisualQueryResidual(nn.Module):
    def __init__(self, width, rank, time_width=8):
        super().__init__()
        if time_width % 2:
            raise ValueError('time features require an even width')
        self.register_buffer('frequencies', 2.**torch.arange(time_width//2), persistent=True)
        self.down = nn.Linear(width, rank, bias=False)
        self.time = nn.Linear(time_width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden, t):
        # No zero gate: up learns on step one, down/time can learn from step two.
        phase = self.frequencies * (math.pi*float(t))
        features = torch.cat((phase.sin(), phase.cos())).to(self.down.weight.dtype)
        value = self.down(hidden.to(self.down.weight.dtype))+self.time(features)
        return self.up(value).to(hidden.dtype)


class VisualReadAdapter(nn.Module):
    architecture = 'und_query_time_low_rank_v1'

    def __init__(self, decoder, rank=32):
        super().__init__()
        self.layers = nn.ModuleList([VisualQueryResidual(decoder.embed_tokens.weight.shape[1], rank)
                                    for _ in decoder.layers])
        self.to(decoder.embed_tokens.weight.device)

    def forward(self, layer, hidden, t):
        return self.layers[layer](hidden, t)


class VisualStudent(nn.Module):
    """Register only newly trained modules; independently freeze the text teacher."""
    def __init__(self, runtime, text_teacher=None):
        super().__init__()
        if runtime.writer.topology != 'C':
            raise ValueError('visual/loop training requires continuous C writer')
        object.__setattr__(self, 'runtime', runtime)
        self.writer = runtime.writer
        self.visual = VisualReadAdapter(runtime.decoder, runtime.config.get('visual_rank', self.writer.rank))
        self.writer.requires_grad_(runtime.config.get('train_writer', False))
        teacher=None
        if text_teacher is not None:
            teacher=MemoryWriter(runtime.decoder,'C',self.writer.slots,self.writer.rank)
            teacher.load_state_dict(text_teacher.state_dict(),strict=True)
            teacher.eval().requires_grad_(False)
        object.__setattr__(self,'text_teacher',teacher)
        self.text_teacher_hash=None

    def write(self, source, t, *, prompt=None, source_off=False, visual=True,
              detach_source=True, source_layers=None):
        runtime = self.runtime
        from .feedback_source import compose_source
        if visual and source_layers is None:
            source_layers = runtime.config.get('source_layers')
        if prompt is not None:
            # Source-off removes GEN evidence while preserving the target prompt.
            source = prompt if source_off else compose_source(prompt, source, source_layers)
            source_layers = None
            source_off = False
        positions = torch.arange(runtime.config['writer_position_start'],
                                 runtime.config['writer_position_start']+self.writer.slots,
                                 device=runtime.device)
        with autocast_for(runtime.device):
            return self.writer(runtime.seeds, source, positions, source_off,
                               source_kind='visual' if visual else 'text', t=t,
                               visual_adapter=self.visual if visual else None,
                               detach_source=detach_source, source_layers=source_layers)

    @torch.no_grad()
    def teacher_bank(self, caption, state=None):
        runtime = self.runtime
        if self.text_teacher is None: raise ValueError('text teacher not loaded')
        if state is not None:
            from .feedback_source import prompt_source, write_feedback
            from .explicit_feedback import correction_source
            flow, lengths, positions = runtime.flow_inputs(state.metadata['prompt'])
            correction = correction_source(runtime, flow, lengths, positions, caption)[0]
            return write_feedback(runtime, prompt_source(flow, lengths, positions),
                                  correction, writer=self.text_teacher)
        source = runtime.text(caption)[0]
        positions = torch.arange(runtime.config['writer_position_start'],
                                 runtime.config['writer_position_start']+self.writer.slots,
                                 device=runtime.device)
        with autocast_for(runtime.device):
            return self.text_teacher(runtime.seeds, source, positions)

    def parameter_groups(self):
        return {name: [p for p in module.parameters() if p.requires_grad]
                for name, module in (('visual', self.visual), ('writer', self.writer))}


def validate_config(runtime, stage):
    cfg = runtime.config
    if stage not in ('C2-observe', 'C2-feedback', 'C3-loop'):
        raise ValueError('select C2-observe/C2-feedback/C3-loop')
    if cfg['slots'] != 16 or runtime.writer.topology != 'C':
        raise ValueError('feedback requires continuous UND and 16 slots')
    if cfg.get('visual_architecture') != VisualReadAdapter.architecture:
        raise ValueError('unknown visual query residual architecture')
    layers = cfg.get('source_layers')
    if not layers or len(set(layers)) != len(layers) or any(not 0 <= i < len(runtime.decoder.layers) for i in layers):
        raise ValueError('declare a valid GEN source layer window')
    if set(cfg['loss_weights']) != {'sem','teacher','flow'} or any(w<0 for w in cfg['loss_weights'].values()):
        raise ValueError('declare sem/teacher/flow weights; text replay is no longer a training objective')
    if stage=='C2-observe' and (cfg['loss_weights']['teacher'] or cfg['loss_weights']['flow']):
        raise ValueError('Observe trains GEN reading through text supervision')
    if not cfg.get('noise_curriculum') or not cfg.get('noise_calibration'):
        raise ValueError('declare curriculum and calibrated noise limits')
    if any(cfg[k]<=0 for k in ('effective_batch','steps','eval_every')):raise ValueError('positive training budgets required')
    if cfg.get('feedback_updates',1) not in (1,2):raise ValueError('one or two feedback updates required')
