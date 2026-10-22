"""One goal-aware spatial RGB velocity head, outside all native GEN blocks."""
import torch
from torch import nn
from .reader import SemanticReader


class CorrectionReadout(nn.Module):
    def __init__(self,width,patch,channels=256,heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(width,heads,batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.project = nn.Conv2d(width,channels,1)
        self.output = nn.Conv2d(channels,3*patch*patch,3,padding=1)
        self.shuffle = nn.PixelShuffle(patch)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,hidden,prompt,grid):
        hidden = hidden.float(); prompt = prompt.detach().float()
        attention,_ = self.attention(hidden,prompt,prompt,need_weights=False)
        value = self.norm(hidden+attention).transpose(1,2).reshape(len(hidden),-1,*grid)
        return self.shuffle(self.output(torch.nn.functional.silu(self.project(value))))


class Corrector(nn.Module):
    def __init__(self,runtime,config):
        super().__init__();object.__setattr__(self,'runtime',runtime)
        self.reader = SemanticReader(runtime,config)
        self.head = CorrectionReadout(runtime.decoder.config.hidden_size,runtime.patch,
            config['head_channels'],config['readout_heads']).to(runtime.device)

    def forward(self,observation,t,prompt):
        spatial,_ = self.reader(observation,t)
        with torch.autocast(self.runtime.device.type,enabled=False):
            return self.head(spatial,prompt,self.runtime.grid)

    def initialize_global(self,steps):
        self.register_buffer('global_sums',torch.zeros(steps,3,device=self.runtime.device))
        self.register_buffer('global_counts',torch.zeros(steps,device=self.runtime.device))

    def global_delta(self,index):
        if self.global_counts[index] <= 0: raise ValueError('No training calibration observations at this time')
        return (self.global_sums[index]/self.global_counts[index])[None,:,None,None]
