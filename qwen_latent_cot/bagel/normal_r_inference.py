"""Normal memory only: total rounds R = 2, 4, 6, 8; R2 reference probes."""
from __future__ import annotations

import torch

from .mechanism_inference import MemoryMechanismEngine

ROUNDS = (2, 4, 6, 8)
MODES = tuple(f"normal_r{r}" for r in ROUNDS)
REFERENCE = MODES[0]
PAIRS = {f"r{r}_vs_r2": (f"normal_r{r}", REFERENCE) for r in ROUNDS[1:]}


class NormalRoundEngine(MemoryMechanismEngine):
    @torch.inference_mode()
    def velocity(self, x_t, timestep, mode, *, diagnostics=False):
        if mode not in MODES:
            raise ValueError(f"normal-R evaluation accepts only {MODES}, got {mode!r}")
        return super().velocity(x_t, timestep, "normal", diagnostics=diagnostics,
                                rounds=ROUNDS[MODES.index(mode)])


def round_metrics(velocities, diagnostics, lengths, step, timestep):
    chunks = {mode: tensor.detach().float().cpu().split(lengths)
              for mode, tensor in velocities.items()}
    rows = []
    for sample, reference in enumerate(chunks[REFERENCE]):
        norm = float(reference.norm())
        row = {"sample": sample, "step": int(step), "t": float(timestep),
               "reference_arm": REFERENCE, "v_r2_norm": norm,
               "hidden_branch": "conditional", "arms": {}}
        for name, (left, right) in PAIRS.items():
            delta = float((chunks[left][sample] - chunks[right][sample]).norm())
            row[f"delta_{name}_norm"] = delta
            row[f"relative_{name}"] = delta / (norm + 1e-12)
        for mode in MODES:
            arm = {"relative_velocity_vs_r2": float((chunks[mode][sample] - reference).norm()) / (norm + 1e-12)}
            for field in ("gen_body_hidden", "gen_suffix_hidden"):
                baseline = diagnostics[REFERENCE][field].split(lengths)[sample].float()
                value = diagnostics[mode][field].split(lengths)[sample].float()
                arm[f"relative_{field}_vs_r2"] = float((value - baseline).norm() / (baseline.norm() + 1e-12))
            diag = diagnostics[mode]
            if "memory_read" in diag:
                arm["memory_read_norm"] = float(diag["memory_read"].reshape(len(lengths), -1)[sample].float().norm())
            arm["memory_write_input_norms"] = [
                float(value.reshape(len(lengths), -1)[sample].float().norm())
                for value in diag.get("memory_write_inputs", [])]
            row["arms"][mode] = arm
        rows.append(row)
    return rows


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    return value


@torch.inference_mode()
def probe_reference_trajectory(engine, timesteps, dts, on_step):
    """All four R arms share x_t; only normal/R2 advances this trajectory."""
    x_t = engine.noise.clone()
    for step, (t, dt) in enumerate(zip(timesteps, dts)):
        velocities, diagnostics = {}, {}
        for mode in MODES:
            velocity, diag = engine.velocity(x_t.clone(), t, mode, diagnostics=True)
            if mode == REFERENCE:
                reference_velocity = velocity
            velocities[mode] = velocity.detach().float().cpu()
            diagnostics[mode] = _cpu(diag)
        rows = round_metrics(velocities, diagnostics, engine.lengths, step, t)
        on_step(step, x_t.detach().cpu().clone(), float(t), rows, diagnostics)
        x_t = engine.model.image_euler_step(x_t, reference_velocity, dt)
    return x_t


@torch.inference_mode()
def generate_trajectory(engine, timesteps, dts, mode):
    if mode not in MODES:
        raise ValueError("only normal R2/4/6/8 trajectories are supported")
    x_t = engine.noise.clone()
    for t, dt in zip(timesteps, dts):
        velocity, _ = engine.velocity(x_t, t, mode)
        x_t = engine.model.image_euler_step(x_t, velocity, dt)
    return x_t
