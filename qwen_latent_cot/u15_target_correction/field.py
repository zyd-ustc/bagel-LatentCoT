"""Native continuation and fixed-time state editing."""

import torch
from torch.utils.checkpoint import checkpoint
from . import GEOMETRY_PROTOCOL, KIND, PROTOCOL_VERSION

def protocol(rt, cfg):
    """What a cached record actually depends on.

    The state times a cache carries are the *grid nodes* it walked, and the
    probabilities a run draws them with are a sampling choice. Binding the
    cache to the latter made a single-node 0.36 recipe reject the three-node
    0.25/0.36/0.50 cache even though the node it reads is identically
    discretised. The grid is what is pinned here; the draw stays free.
    """
    nodes = [float(t) for t in cfg.get("phase_grid_times", cfg["state_times"])]
    return dict(
        phase_grid_times=nodes,
        kind=KIND,
        protocol_version=PROTOCOL_VERSION,
        geometry=GEOMETRY_PROTOCOL,
        inversion=inversion_solver(cfg),
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
        edit_steps=int(cfg["edit_steps"]),
        lookahead_steps=int(cfg["lookahead_steps"]),
        condition_protocol=cfg["condition_protocol"],
        preview_protocol="heun_then_euler_from_state_v1",
        tail_solver="euler",
        keep_instruction=cfg["keep_instruction"],
    )


def phase_grid(rt, cfg):
    """The deployment solver grid: the native grid plus the declared nodes.

    The nodes are ``phase_grid_times`` when a config declares them, and fall
    back to the sampled ``state_times`` otherwise. Supervised moments and
    solver nodes stay separable: a run can read at 0.36 while the cached parent
    states still span 0.25/0.36/0.50, and both sides of a cache then agree on
    the same discretisation.
    """
    values = [float(t) for t in rt.times()]
    extras = [float(cfg["read_time"])]
    extras += [float(t) for t in cfg.get("phase_grid_times", cfg["state_times"])]
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


def edit_state(editor, anchor, time, prefixes, steps, base_prediction=None, grad=False):
    """Euler in s at fixed native time t. Returns the edited state and b0.

    ``grad=True`` keeps the edit axis differentiable so a terminal image loss can
    reach the editor through the frozen continuation. The default stays detached.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError("The state edit needs at least one Euler step")
    state = anchor.float()
    if base_prediction is None:
        base = editor.image_clean(state, float(time), prefixes["image"])
    else:
        base = base_prediction.float()
    if not grad:
        with torch.no_grad():
            for index in range(steps):
                progress = index / steps
                speed = editor.state_velocity(state, float(time), progress, prefixes["joint"], base)
                state = state + speed / steps
        return state, base
    for index in range(steps):
        progress = index / steps
        speed = editor.state_velocity(state, float(time), progress, prefixes["joint"], base)
        state = state + speed / steps
    return state, base


def inversion_solver(cfg):
    """Which backward scheme produced the labels; part of the cache protocol."""
    return str(cfg.get("inversion", {}).get("solver", "reverse_heun"))
