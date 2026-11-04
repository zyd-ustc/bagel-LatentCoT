import json
import random
from pathlib import Path
import torch
from . import distributed as dist
from . import KIND


def canonical_parameters(module):
    return {
        n.replace("_fsdp_wrapped_module.", "").replace(
            "_checkpoint_wrapped_module.", ""
        ): p
        for n, p in module.named_parameters()
        if p.requires_grad
    }



def save(
    path,
    corrector,
    optimizer,
    ema,
    sampler,
    step,
    cfg,
    identity,
    device,
    *,
    kind=KIND,
):
    path = Path(path)
    if dist.rank() == 0:
        path.mkdir(exist_ok=False)
    dist.barrier()
    partial = path / f"rank_{dist.rank():05d}.partial"
    torch.save(
        dict(
            weights=cpu(canonical_parameters(corrector)),
            optimizer=cpu(optimizer.state_dict()),
            ema=cpu(ema),
            sampler=sampler.state_dict(),
            rng=rng(device),
        ),
        partial,
    )
    partial.replace(partial.with_suffix(".pt"))
    dist.barrier()
    if dist.rank() == 0:
        meta = dict(
            kind=kind,
            step=step,
            world=dist.world_size(),
            config=cfg,
            identity=identity,
        )
        (path / "manifest.partial").write_text(json.dumps(meta, indent=2) + "\n")
        (path / "manifest.partial").replace(path / "manifest.json")
    dist.barrier()


def load(path, corrector, cfg, identity, *, use_ema=False, kind=KIND):
    path = Path(path)
    meta = json.loads((path / "manifest.json").read_text())
    if (meta["kind"], meta["world"], meta["config"], meta["identity"]) != (
        kind,
        dist.world_size(),
        cfg,
        identity,
    ):
        raise ValueError("Checkpoint method, world or identity differs")
    state = torch.load(
        str(path / f"rank_{dist.rank():05d}.pt"),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    params = canonical_parameters(corrector)
    weights = state["ema"] if use_ema else state["weights"]
    if params.keys() != weights.keys():
        raise ValueError("Corrector parameter inventory differs")
    with torch.no_grad():
        for name, p in params.items():
            if p.shape != weights[name].shape:
                raise ValueError("FSDP partition differs: " + name)
            p.copy_(weights[name].to(p))
    return meta, state



def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value

def rng(device):
    return dict(cpu=torch.get_rng_state(),python=random.getstate(),
                device=getattr(torch,device.type).get_rng_state(device).cpu() if device.type!='cpu' else None)

def restore_rng(saved,device):
    torch.set_rng_state(saved['cpu']);random.setstate(saved['python'])
    if device.type!='cpu':getattr(torch,device.type).set_rng_state(saved['device'],device)
