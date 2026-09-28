"""Plain same-state on-policy velocity-field distillation; no ranking terms."""

from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class OPDLossOutput:
    loss: torch.Tensor
    cosine: torch.Tensor
    relative_error: torch.Tensor
    student_rms: torch.Tensor
    teacher_rms: torch.Tensor


def opd_velocity_loss(student_velocity, teacher_velocity):
    if student_velocity.shape != teacher_velocity.shape:
        raise ValueError("student and teacher velocities must have identical shape")
    student = student_velocity.float()
    teacher = teacher_velocity.detach().float()
    loss = F.mse_loss(student, teacher)
    flat_s, flat_t = student.reshape(-1), teacher.reshape(-1)
    return OPDLossOutput(
        loss=loss,
        cosine=F.cosine_similarity(flat_s, flat_t, dim=0).detach(),
        relative_error=(flat_s - flat_t).norm().div(flat_t.norm().clamp_min(1e-8)).detach(),
        student_rms=student.square().mean().sqrt().detach(),
        teacher_rms=teacher.square().mean().sqrt().detach(),
    )
