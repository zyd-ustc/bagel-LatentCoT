"""Fixed-generation-time state flow, native T2I, and numerical inversion."""

import torch
from torch.utils.checkpoint import checkpoint

from . import KIND, PROTOCOL_VERSION


def protocol(rt, cfg):
    return dict(
        kind=KIND,
        protocol_version=PROTOCOL_VERSION,
        native=rt.identity,
        width=rt.width,
        height=rt.height,
        noise_scale=rt.noise_scale(),
        times=[float(t) for t in rt.times()],
        sampler="euler",
        cfg_scale=float(cfg["cfg_scale"]),
        image_cfg_scale=float(cfg["image_cfg_scale"]),
        cfg_norm=cfg["cfg_norm"],
        cfg_interval=[float(t) for t in cfg["cfg_interval"]],
        t_eps=float(cfg["t_eps"]),
        read_time=float(cfg["read_time"]),
        state_times=[float(t) for t in cfg["state_times"]],
        state_time_probs=[float(p) for p in cfg["state_time_probs"]],
        edit_steps=int(cfg["edit_steps"]),
        lookahead_steps=int(cfg["lookahead_steps"]),
        condition_protocol=cfg["condition_protocol"],
        preview_protocol="heun_then_euler_from_state_v1",
        tail_solver="euler",
        keep_instruction=cfg["keep_instruction"],
    )


def phase_grid(rt, cfg):
    """The native grid plus every supported state time and the read time."""
    values = [float(t) for t in rt.times()]
    extras = [float(cfg["read_time"]), *[float(t) for t in cfg["state_times"]]]
    for value in extras:
        values = [t for t in values if abs(t - value) > 1e-7] + [value]
    return sorted(values)


def segment_times(grid, start, end):
    start, end = float(start), float(end)
    if end < start:
        raise ValueError("segment_times expects start <= end")
    return [start] + [float(t) for t in grid if start + 1e-7 < t < end - 1e-7] + [end]


def time_key(value):
    return f"{float(value):.6f}"


def weighted_choice(rng, values, weights, total=1.0):
    """One weighted draw in declaration order; ``total`` lets callers skip renormalizing."""
    draw = rng.random() * float(total)
    cumulative = 0.0
    for value, weight in zip(values, weights):
        cumulative += float(weight)
        if draw <= cumulative + 1e-12:
            return value
    return values[-1]


def sample_state_time(cfg, rng):
    """One supported native generation time; edit progress s is sampled later."""
    times = [float(t) for t in cfg["state_times"]]
    probs = [float(p) for p in cfg["state_time_probs"]]
    return weighted_choice(rng, times, probs)


def native_shift_time(u, shift=3.0):
    """Map a uniform schedule coordinate to the native shifted time."""
    sigma = 1 - float(u)
    sigma = float(shift) * sigma / (1 + (float(shift) - 1) * sigma)
    return 1 - sigma


def native_unshift_time(t, shift=3.0):
    """Inverse of native_shift_time, used only for accounting."""
    sigma = 1 - float(t)
    if sigma <= 0:
        return 1.0
    u = sigma / (shift + (1 - shift) * sigma)
    return 1 - u


def euler(z, times, velocity):
    """Discrete Euler with every state connected to the velocity parameters."""
    z = z.float()
    for left, right in zip(times[:-1], times[1:]):
        step = z.new_tensor(float(right) - float(left))
        moment = z.new_tensor(float(left))

        def call(state, moment=moment):
            return velocity(state, moment)

        if torch.is_grad_enabled():
            speed = checkpoint(call, z, use_reentrant=False)
        else:
            speed = call(z)
        z = z + step * speed
    return z


@torch.no_grad()
def preview_image(state, start_time, grid, velocity):
    """Look ahead on a copy: one Heun interval then one Euler interval."""
    original = state.detach().clone()
    tail = segment_times(grid, start_time, 1.0)
    middle = tail[len(tail) // 2]
    if not float(start_time) < middle < 1:
        raise ValueError("Native grid has no interior preview node after the state time")
    z = original.clone()
    left, mid, right = z.new_tensor(float(start_time)), z.new_tensor(middle), z.new_tensor(1.0)
    first = velocity(z, left)
    proposal = z + (mid - float(start_time)) * first
    z = z + (mid - float(start_time)) * 0.5 * (first + velocity(proposal, mid))
    z = z + (right - mid) * velocity(z, mid)
    if not torch.equal(state.detach(), original):
        raise RuntimeError("Preview mutated the main trajectory")
    return z


class NativeT2I:
    """Frozen text-to-image CFG. It never loads the editor LoRA."""

    def __init__(self, rt, raw, cfg):
        self.rt, self.cfg = rt, cfg
        self.positive = rt.prefix(raw)
        self.negative = rt.prefix(None) if float(cfg["cfg_scale"]) > 1 else None
        self.calls = 0

    def __call__(self, z, t):
        self.calls += 1
        positive = self.rt.predict(z, t, self.positive).float()
        lo, hi = [float(v) for v in self.cfg["cfg_interval"]]
        if self.negative is None or not lo <= float(t) <= hi:
            return positive
        self.calls += 1
        negative = self.rt.predict(z, t, self.negative).float()
        return self.rt.guided(positive, negative, float(self.cfg["cfg_scale"])).float()


def forward_native(state, start_time, grid, velocity):
    """The frozen native continuation used by reconstruction checks."""
    if not float(start_time) < 1:
        raise ValueError("A native continuation needs a state before the endpoint")
    with torch.no_grad():
        return euler(state.float(), segment_times(grid, start_time, 1.0), velocity).float()


def edit_state(editor, anchor, time, prefixes, steps, base_prediction=None):
    """Euler in s at fixed native time t. Returns the edited state and b0."""
    steps = int(steps)
    if steps < 1:
        raise ValueError("The state edit needs at least one Euler step")
    state = anchor.float()
    if base_prediction is None:
        base = editor.image_clean(state, float(time), prefixes["image"])
    else:
        base = base_prediction.float()
    with torch.no_grad():
        for index in range(steps):
            progress = index / steps
            speed = editor.state_velocity(state, float(time), progress, prefixes["joint"], base)
            state = state + speed / steps
    return state, base


def _descending_subgrid(grid, start, end, substeps):
    ascending = segment_times(grid, end, start)
    nodes = [ascending[0]]
    for left, right in zip(ascending[:-1], ascending[1:]):
        for index in range(1, substeps + 1):
            nodes.append(left + (right - left) * index / substeps)
    return list(reversed(nodes))


def reverse_heun_states(state, start_time, targets, grid, velocity, substeps=2):
    """One reverse Heun sweep from t=start_time down to the earliest target.

    Every native interval is split into ``substeps`` reverse intervals. The
    returned snapshots are keyed by the exact supported state times.
    """
    targets = sorted(float(t) for t in targets)
    grid = sorted(float(t) for t in grid)
    if not targets:
        raise ValueError("At least one inversion target is required")
    if any(not 0 < t < float(start_time) <= 1 for t in targets):
        raise ValueError("Inversion targets must sit before the start time")
    nodes = _descending_subgrid(grid, start_time, targets[0], substeps)
    snapshots = {}
    current = state.detach().float().clone()
    with torch.no_grad():
        for left, right in zip(nodes[:-1], nodes[1:]):
            step = right - left
            first = velocity(current, current.new_tensor(left))
            proposal = current + step * first
            second = velocity(proposal, proposal.new_tensor(right))
            current = current + 0.5 * step * (first + second)
            for target in targets:
                if abs(right - target) <= 1e-7:
                    snapshots[time_key(target)] = current.clone()
    missing = [t for t in targets if time_key(t) not in snapshots]
    if missing:
        raise ValueError("Inversion grid did not reach " + ", ".join(map(str, missing)))
    return snapshots


def inversion_grid(rt, cfg, substeps=2):
    """The descending grid actually used by reverse Heun."""
    grid = phase_grid(rt, cfg)
    end = min(float(t) for t in cfg["state_times"])
    return _descending_subgrid(grid, 1.0, end, int(substeps))
