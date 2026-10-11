"""Native Euler nodes shared by trajectory capture and continuation."""
import torch


def native_times(config):
    steps, shift = config['sampler_steps'], config['timestep_shift']
    if steps < 2 or shift <= 0:
        raise ValueError('native sampler needs at least two nodes and positive shift')
    times = torch.linspace(1, 0, steps)
    return (shift*times/(1+(shift-1)*times)).tolist()


def nearest_node(config, requested):
    times = native_times(config)
    index = min(range(len(times)-1), key=lambda i: abs(times[i]-requested))
    return index, times[index]


def native_suffix(config, t):
    times = native_times(config)
    matches = [i for i, node in enumerate(times[:-1]) if abs(t-node) <= 1e-7]
    if not matches:
        raise ValueError('continuation must start at a native sampler node')
    return times[matches[0]+1:]
