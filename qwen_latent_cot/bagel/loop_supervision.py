"""Full-suffix velocity supervision; Read is never a supervised Write round."""
from dataclasses import dataclass
from typing import Tuple, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LoopSupervisionOutput:
    final_velocity: torch.Tensor
    write_round_velocities: Tuple[torch.Tensor, ...]
    write_round_memories: Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class LoopSupervisionLoss:
    loss: torch.Tensor
    deep_supervision: torch.Tensor
    distillation: torch.Tensor
    monotonic: torch.Tensor
    round_errors: torch.Tensor


def loop_supervision_loss(velocities: Sequence[torch.Tensor], target: torch.Tensor,
                          *, weights: Sequence[float], loop_distill=False,
                          final_round_validated=False, lambda_loop_distill=0.2,
                          lambda_loop_monotonic=0.1, monotonic_margin=0.0):
    if not velocities or len(weights) != len(velocities):
        raise ValueError("one weight is required for every Write round")
    if any(v.shape != target.shape for v in velocities):
        raise ValueError("all Write velocities must match the target")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Write weights must be nonnegative with positive sum")
    if min(lambda_loop_distill, lambda_loop_monotonic, monotonic_margin) < 0:
        raise ValueError("loss coefficients and margin must be nonnegative")
    if loop_distill and (not final_round_validated or len(velocities) < 2):
        raise ValueError("distillation requires held-out final-round improvement and >=2 Writes")
    errors = torch.stack([(v.float() - target.detach().float()).square().mean() for v in velocities])
    ds = (errors * errors.new_tensor(weights)).sum()
    zero = errors.sum() * 0.0
    # Teacher detach is intentional: shallow students cannot move the target.
    ld = sum(F.mse_loss(v.float(), velocities[-1].detach().float())
             for v in velocities[:-1]) if loop_distill else zero
    mono = F.relu(monotonic_margin + errors[1:] - errors[:-1]).sum()
    return LoopSupervisionLoss(ds + lambda_loop_distill * ld + lambda_loop_monotonic * mono,
                               ds, ld, mono, errors.detach())
