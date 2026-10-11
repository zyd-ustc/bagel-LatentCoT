"""Keep native mixed-precision modules usable in the CPU engineering harness."""
import torch
from ..bagel.accelerator import autocast_for as native_autocast


def autocast_for(device):
    return torch.autocast('cpu', dtype=torch.bfloat16) if device.type == 'cpu' else native_autocast(device)
