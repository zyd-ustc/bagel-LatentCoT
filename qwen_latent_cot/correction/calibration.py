"""Train-only least-squares constant residual field for each active time node."""
import torch
from torch import nn
import torch.distributed as dist


class GlobalCalibration(nn.Module):
    def __init__(self, nodes, channels, device):
        super().__init__()
        self.nodes=tuple(nodes)
        self.register_buffer('sums',torch.zeros(len(nodes),channels,device=device))
        self.register_buffer('counts',torch.zeros(len(nodes),device=device))

    @torch.no_grad()
    def observe(self, pending, t, residual):
        index=min(range(len(self.nodes)),key=lambda i:abs(self.nodes[i]-t))
        if abs(self.nodes[index]-t)>1e-6:raise ValueError('Training time is not a calibration node')
        pending[index,:-1]+=residual.detach().float().mean(0)
        pending[index,-1]+=1

    @torch.no_grad()
    def update(self, pending):
        if dist.is_initialized():dist.all_reduce(pending,op=dist.ReduceOp.SUM)
        self.sums.add_(pending[:,:-1]);self.counts.add_(pending[:,-1])

    def forward(self,t):
        index=min(range(len(self.nodes)),key=lambda i:abs(self.nodes[i]-t))
        if abs(self.nodes[index]-t)>1e-6:raise ValueError('Sampling time is not a calibration node')
        if self.counts[index]==0:raise ValueError('Calibration time has no training observations')
        return self.sums[index]/self.counts[index]
