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
    def __init__(self, runtime):
        super().__init__()
        if runtime.writer.topology != 'C':
            raise ValueError('visual/loop training requires continuous C writer')
        object.__setattr__(self, 'runtime', runtime)
        self.writer = runtime.writer
        self.visual = VisualReadAdapter(runtime.decoder, runtime.config.get('visual_rank', self.writer.rank))
        self.writer.requires_grad_(runtime.config.get('train_writer', False))
        teacher = MemoryWriter(runtime.decoder, 'C', self.writer.slots, self.writer.rank)
        teacher.load_state_dict(self.writer.state_dict(), strict=True)
        teacher.eval().requires_grad_(False)
        # Not part of optimizer, student state or broadcast parameter collection.
        object.__setattr__(self, 'text_teacher', teacher)

    def write(self, source, t, *, source_off=False, visual=True, detach_source=True):
        runtime = self.runtime
        positions = torch.arange(runtime.config['writer_position_start'],
                                 runtime.config['writer_position_start']+self.writer.slots,
                                 device=runtime.device)
        with autocast_for(runtime.device):
            return self.writer(runtime.seeds, source, positions, source_off,
                               source_kind='visual' if visual else 'text', t=t,
                               visual_adapter=self.visual if visual else None,
                               detach_source=detach_source)

    @torch.no_grad()
    def teacher_bank(self, caption):
        runtime = self.runtime
        source = runtime.text(caption)[0]
        positions = torch.arange(runtime.config['writer_position_start'],
                                 runtime.config['writer_position_start']+self.writer.slots,
                                 device=runtime.device)
        with autocast_for(runtime.device):
            return self.text_teacher(runtime.seeds, source, positions)

    def parameter_groups(self):
        return {name: [p for p in module.parameters() if p.requires_grad]
                for name, module in (('visual', self.visual), ('writer', self.writer))}
