"""Fixed-generation-time state flow, native T2I, and numerical inversion."""

import torch
from torch.utils.checkpoint import checkpoint

from . import GEOMETRY_PROTOCOL, KIND, PROTOCOL_VERSION


def protocol(rt, cfg):
    return dict(
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


def native_forward_states(state, grid, state_times, velocity):
    """Walk the native Euler map, snapshotting every supported state time.

    Each snapshot carries the preview the deployment tail actually renders from
    it, so the pair and the picture come from the same trajectory.
    """
    supported = {time_key(t) for t in state_times}
    current = state.float().clone()
    states, previews = {}, {}
    for left, right in zip(grid[:-1], grid[1:]):
        if abs(float(left) - 1.0) < 1e-7:
            break
        key = time_key(left)
        if key in supported:
            states[key] = current.detach().float().clone()
            previews[key] = preview_image(current, left, grid, velocity)
        step = float(right) - float(left)
        current = current + step * velocity(current, current.new_tensor(float(left)))
    return dict(states=states, previews=previews, final=current.detach().float().clone())


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


def _descending_subgrid(grid, start, end, substeps):
    ascending = segment_times(grid, end, start)
    nodes = [ascending[0]]
    for left, right in zip(ascending[:-1], ascending[1:]):
        for index in range(1, substeps + 1):
            nodes.append(left + (right - left) * index / substeps)
    return list(reversed(nodes))


def reverse_heun_states(state, start_time, targets, grid, velocity, substeps=2, endpoint=None):
    """One reverse Heun sweep from t=start_time down to the earliest target.

    Every native interval is split into ``substeps`` reverse intervals. The
    returned snapshots are keyed by the exact supported state times.

    ``endpoint`` truncates the *evaluation* time near the image end. The native
    field divides by ``(1-t)`` clamped at ``t_eps``, so evaluating at exactly
    t=1 divides a residual by ``t_eps`` and reports a velocity that belongs to
    no time on the grid. Passing ``1 - t_eps`` here keeps the step from 1.0
    while reading the field at its largest trained time.
    """
    targets = sorted(float(t) for t in targets)
    grid = sorted(float(t) for t in grid)
    if not targets:
        raise ValueError("At least one inversion target is required")
    if any(not 0 < t < float(start_time) <= 1 for t in targets):
        raise ValueError("Inversion targets must sit before the start time")
    nodes = _descending_subgrid(grid, start_time, targets[0], substeps)
    limit = None if endpoint is None else float(endpoint)
    read = (lambda value: value) if limit is None else (lambda value: min(value, limit))
    snapshots = {}
    current = state.detach().float().clone()
    with torch.no_grad():
        for left, right in zip(nodes[:-1], nodes[1:]):
            step = right - left
            first = velocity(current, current.new_tensor(read(left)))
            proposal = current + step * first
            second = velocity(proposal, proposal.new_tensor(read(right)))
            current = current + 0.5 * step * (first + second)
            for target in targets:
                if abs(right - target) <= 1e-7:
                    snapshots[time_key(target)] = current.clone()
    missing = [t for t in targets if time_key(t) not in snapshots]
    if missing:
        raise ValueError("Inversion grid did not reach " + ", ".join(map(str, missing)))
    return snapshots


def discrete_inverse_states(state, grid, targets, velocity, iterations=4, damping=0.5,
                            initial=None, tolerance=1e-5):
    """Invert the deployment Euler map on its own nodes.

    Deployment advances ``z_{k+1} = z_k + dt_k v(z_k, t_k)``. The state the
    deployment map would actually continue is the ``z_k`` satisfying that
    relation, so we solve

        ``z_k = z_{k+1} - dt_k v(z_k, t_k)``

    with damped Picard, starting from the reverse-Heun estimate when one is
    supplied. The returned residual is the *equation residual of the final
    state*, not the size of the last update: a finite iteration budget can leave
    a solution unfound, so the residual measures solve accuracy, not whether the
    image is reachable.

    ``segment_residual`` aggregates the maximum residual over every node between
    the target time and ``t=1``, which is the traversal that produced that
    target's state.
    """
    targets = sorted(float(t) for t in targets)
    if not targets:
        raise ValueError("At least one inversion target is required")
    if any(not 0 < t < 1 for t in targets):
        raise ValueError("Inversion targets must sit strictly inside the trajectory")
    nodes = segment_times(sorted(float(t) for t in grid), targets[0], 1.0)
    iterations = max(1, int(iterations))
    damping = float(damping)
    if not 0 < damping <= 1:
        raise ValueError("damping must lie in (0, 1]")
    snapshots, diagnostics = {}, {}
    segment_residuals = []
    current = state.detach().float().clone()
    with torch.no_grad():
        for index in range(len(nodes) - 2, -1, -1):
            left, right = nodes[index], nodes[index + 1]
            step = right - left
            moment = current.new_tensor(left)
            guess = None if initial is None else initial.get(time_key(left))
            estimate = (
                current - step * velocity(current, moment)
                if guess is None else guess.detach().float().clone()
            )
            used = 0
            for used in range(1, iterations + 1):
                updated = current - step * velocity(estimate, moment)
                delta = float((updated - estimate).abs().mean())
                estimate = estimate + damping * (updated - estimate)
                if delta <= tolerance:
                    break
            equation = float((estimate - (current - step * velocity(estimate, moment))).abs().mean())
            segment_residuals.append(equation)
            for target in targets:
                if abs(left - target) <= 1e-7:
                    snapshots[time_key(target)] = estimate.clone()
                    diagnostics[time_key(target)] = dict(
                        residual=equation,
                        segment_residual=max(segment_residuals),
                        iterations=used,
                        damping=damping,
                        tolerance=tolerance,
                        node=left,
                    )
            current = estimate
    missing = [target for target in targets if time_key(target) not in snapshots]
    if missing:
        raise ValueError("Inversion grid did not reach " + ", ".join(map(str, missing)))
    return snapshots, diagnostics


def inversion_solver(cfg):
    """Which backward scheme produced the labels; part of the cache protocol."""
    return str(cfg.get("inversion", {}).get("solver", "reverse_heun"))


def inversion_endpoint(cfg):
    """Largest native time the field is trained for; used to read the inversion.

    The field divides by ``(1-t)`` clamped at ``t_eps``, so reading at exactly
    t=1 reports a velocity that belongs to no time on the grid. Truncating the
    *read* time keeps the step from 1.0 and removes that artefact.
    """
    return 1.0 - float(cfg["t_eps"])


def inversion_grid(rt, cfg, substeps=2):
    """The descending grid actually used by reverse Heun."""
    grid = phase_grid(rt, cfg)
    end = min(float(t) for t in cfg["state_times"])
    return _descending_subgrid(grid, 1.0, end, int(substeps))
