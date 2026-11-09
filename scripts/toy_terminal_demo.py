"""A controlled toy for the terminal-supervision claim, runnable on CPU.

The real pipeline cannot show *why* state-space regression goes rough; it only
shows that it does. This script rebuilds the same arrangement at 32x32 with a
tiny flow model so the mechanism is visible and the comparison is controlled:

    scene        smooth background plus a few blobs
    edit         one extra blob
    base flow    trained on scenes only, then frozen
    editor       pushes the state at t=0.36 from the source trajectory to the
                 target trajectory, trained two ways

``state``  regress the label state directly (what the current recipe does)
``terminal`` push the state through the frozen tail and score the picture

Both objectives see the same pairs. The question is whether the state-space
objective buys displacement with high-frequency energy, and whether the
terminal objective avoids that.

Usage::

    python scripts/toy_terminal_demo.py --steps 300 --editor-steps 200
"""

import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F

SIZE = 24
READ = 0.36
SIGMA = 1.0


def scene(batch, generator):
    """Smooth background plus two to four soft blobs."""
    grid = torch.linspace(-1, 1, SIZE)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    yy, xx = yy.view(1, 1, SIZE, SIZE), xx.view(1, 1, SIZE, SIZE)
    image = torch.zeros(batch, 3, SIZE, SIZE)
    base = torch.rand(batch, 3, 1, 1, generator=generator) * 0.4 - 0.2
    image += base
    for _ in range(3):
        amp = torch.rand(batch, 3, 1, 1, generator=generator) * 0.3
        fx = torch.rand(batch, 1, 1, 1, generator=generator) * 2.2 - 1.1
        fy = torch.rand(batch, 1, 1, 1, generator=generator) * 2.2 - 1.1
        image += amp * torch.cos(3.0 * fx * xx + 2.0 * fy * yy)
    return image


def with_blob(image, generator):
    """The edit: one soft blob at a random place, visible and local."""
    batch = image.shape[0]
    grid = torch.linspace(-1, 1, SIZE)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    yy, xx = yy.view(1, 1, SIZE, SIZE), xx.view(1, 1, SIZE, SIZE)
    cx = torch.rand(batch, 1, 1, 1, generator=generator) * 1.2 - 0.6
    cy = torch.rand(batch, 1, 1, 1, generator=generator) * 1.2 - 0.6
    radius = torch.rand(batch, 1, 1, 1, generator=generator) * 0.12 + 0.10
    colour = torch.rand(batch, 3, 1, 1, generator=generator) * 0.8 - 0.4
    blob = torch.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2)))
    return image + colour * blob


class Field(nn.Module):
    """Small convolutional velocity field, time-conditioned by feature add."""

    def __init__(self, width=24):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        self.inp = nn.Conv2d(3, width, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Conv2d(width, width, 3, padding=1) for _ in range(2)
        ])
        self.out = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, z, t, c=None):
        if c is None:
            c = torch.zeros_like(t)
        moment = torch.stack([t.view(-1), c.view(-1)], dim=1)
        h = F.silu(self.inp(z))
        h = h + self.time(moment)[:, :, None, None]
        for block in self.blocks:
            h = h + F.silu(block(h))
        return self.out(h)


class Editor(nn.Module):
    """State displacement at a fixed native time."""

    def __init__(self, width=6):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, 3, 3, padding=1),
        )
        nn.init.zeros_(self.stack[-1].weight)
        nn.init.zeros_(self.stack[-1].bias)

    def forward(self, state):
        return state + self.stack(state)


def hf(image):
    gray = image.mean(1, keepdim=True)
    lap = (
        4 * gray[:, :, 1:-1, 1:-1]
        - gray[:, :, :-2, 1:-1]
        - gray[:, :, 2:, 1:-1]
        - gray[:, :, 1:-1, :-2]
        - gray[:, :, 1:-1, 2:]
    )
    return lap.abs().flatten(1).mean(1)


TAIL_STEPS = 6


def euler(z, field, start, end, steps, c=None):
    z = z.clone()
    step = (end - start) / steps
    for index in range(steps):
        t = torch.full((z.shape[0],), start + index * step)
        cond = torch.full((z.shape[0],), float(c)) if c is not None else None
        z = z + step * field(z, t, cond)
    return z


def train_flow(field, steps, batch, generator):
    opt = torch.optim.Adam(field.parameters(), lr=2e-3)
    for step in range(steps):
        clean = scene(batch, generator)
        flag = (torch.rand(batch, 1, 1, 1, generator=generator) < 0.5).float()
        clean = torch.where(flag > 0, with_blob(clean, generator), clean)
        noise = torch.randn_like(clean) * SIGMA
        t = torch.rand(batch, 1, 1, 1, generator=generator)
        state = t * clean + (1 - t) * noise
        target = clean - noise
        loss = F.mse_loss(field(state, t.view(-1), flag.view(-1)), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"flow step {step + 1} loss {loss.item():.4f}", flush=True)
    return field


def pairs(count, generator, field):
    """Two trajectories of the same field from one noise draw, one per condition."""
    source = scene(count, generator)
    target = with_blob(source, generator)
    noise = torch.randn_like(source) * SIGMA
    anchor = euler(noise, field, 0.0, READ, 12, c=0.0)
    label_state = euler(noise, field, 0.0, READ, 12, c=1.0)
    # the picture the field itself reaches under the edit condition from this noise
    target = euler(noise, field, 0.0, 1.0, 12, c=1.0)
    return dict(source=source, target=target, noise=noise, anchor=anchor,
                label=label_state)


def train_editor(editor, field, data, steps, mode, batch, generator, weight_hf=0.0):
    opt = torch.optim.Adam(editor.parameters(), lr=3e-3)
    for step in range(steps):
        index = torch.randint(0, data["source"].shape[0], (batch,), generator=generator)
        anchor = data["anchor"][index]
        label = data["label"][index]
        target = data["target"][index]
        source = data["source"][index]
        state = editor(anchor)
        if mode == "state":
            loss = F.mse_loss(state, label)
        else:
            picture = euler(state, field, READ, 1.0, TAIL_STEPS, c=1.0)
            loss = F.l1_loss(picture, target)
            if weight_hf > 0:
                ratio = hf(picture) / hf(source).clamp_min(1e-3)
                loss = loss + weight_hf * (
                    (ratio - 1.15).clamp_min(0).square() + (0.75 - ratio).clamp_min(0).square()
                ).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"editor[{mode}] step {step + 1} loss {loss.item():.4f}", flush=True)
    return editor


@torch.no_grad()
def evaluate(editor, field, data):
    picture = euler(editor(data["anchor"]), field, READ, 1.0, TAIL_STEPS, c=1.0)
    native = euler(data["anchor"], field, READ, 1.0, TAIL_STEPS, c=0.0)
    ratio = (hf(picture) / hf(native).clamp_min(1e-3)).mean().item()
    edit_l1 = F.l1_loss(picture, data["target"]).item()
    keep_l1 = F.l1_loss(picture, native).item()
    return dict(hf_ratio=ratio, edit_l1=edit_l1, keep_l1=keep_l1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--editor-steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="/tmp/toy_terminal.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    field = train_flow(Field(), args.steps, args.batch, generator)
    field.eval()
    for parameter in field.parameters():
        parameter.requires_grad_(False)

    train_data = pairs(1024, generator, field)
    probe_data = pairs(64, generator, field)
    results = {}
    for mode, weight in (("state", 0.0), ("terminal", 0.0), ("terminal_hf", 1.0)):
        torch.manual_seed(args.seed + 1)
        editor = Editor()
        train_editor(editor, field, train_data, args.editor_steps, mode if mode != "terminal_hf" else "terminal",
                     min(args.batch, 32), generator, weight_hf=weight)
        results[mode] = evaluate(editor, field, probe_data)
        print(mode, json.dumps(results[mode]), flush=True)

    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=1)
    print("written " + args.output)


if __name__ == "__main__":
    main()
