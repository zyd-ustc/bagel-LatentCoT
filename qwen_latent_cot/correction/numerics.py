"""BAGEL mixed-precision execution on accelerator and the tiny CPU harness."""
import torch


def autocast_for(device):
    return torch.autocast(device.type,dtype=torch.bfloat16)
