"""Cold-start flow matching and final-target supervision through the frozen tail."""

import json
import time
from pathlib import Path
import torch
from . import distributed as dist
from .data import Sampler
from .checkpoint import restore_rng
from .loading import load_native
from .checkpoint import canonical_parameters, save, load
from . import KIND
from .data import Corpus
from .field import NativeCFG, target_rollout
from .loss import target_weights, weighted, image_loss
from .model import SketchResidualCorrector


def train(args, cfg, device):
    batch, world = cfg["global_batch"],dist.world_size()
    if batch < world or batch % world or cfg["steps"] < 1 or cfg["save_every"] < 1:
        raise ValueError("Require positive steps/save interval and global batch divisible by world")
    if cfg['stage']==2 and not (args.resume or args.init_checkpoint):
        raise ValueError('Stage 2 requires stage 1 initialization or an exact resume')
    if args.resume and args.init_checkpoint:
        raise ValueError("Choose exact resume or stage initialization")
    out = Path(args.output)
    if dist.rank()==0:
        out.mkdir(parents=True,exist_ok=bool(args.resume))
    dist.barrier()
    rt = load_native(args.model,args.source,cfg,device,args.distributed)
    corpus = Corpus(args.corpus,rt,cfg)
    sampler = Sampler(corpus,cfg)
    torch.manual_seed(cfg["seed"])
    corrector = SketchResidualCorrector(rt,cfg).eval()
    if args.init_checkpoint:
        initialize_weights(args.init_checkpoint,corrector,rt,cfg,ema=True)
    dist.broadcast_parameters(corrector)
    params = canonical_parameters(corrector)
    optimizer = torch.optim.AdamW(params.values(),lr=cfg["lr"],betas=tuple(cfg["betas"]),
                                 eps=cfg["eps"],weight_decay=cfg["weight_decay"],foreach=False)
    ema = {n:p.detach().float().clone() for n,p in params.items()}
    identity = dict(native=rt.identity,corpus=corpus.manifest,interface=KIND,
                    parameter_layout="replicated_gen_lora")
    start = 0
    if args.resume:
        meta,state = load(args.resume,corrector,cfg,identity,kind=KIND)
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        restore_rng(state["rng"],device)
        ema = {n:v.to(device) for n,v in state["ema"].items()}
        start = meta["step"]
        del state
    if dist.rank()==0:
        (out/"run.json").write_text(json.dumps(dict(config=cfg,identity=identity,
            trainable_parameters=sum(p.numel() for p in params.values()),
            trainable_names=list(params),native_layers=len(rt.decoder.layers),
            lora_layers=list(range(corrector.layer_count)),
            supervision="target images; exact deployment forward, coarse surrogate backward",
            distributed="replicated adapters; summed gradients / global batch"),indent=2)+"\n")
    for step in range(start+1,cfg["steps"]+1):
        begin = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        selected = sampler.take(batch)
        totals = [0.,0.,0.]
        for position in range(dist.rank(),batch,world):
            index,seed,fraction=selected[position]
            pair=corpus.row(index,seed)
            tau,h=pair['tau'],cfg['handoff_time']
            native=NativeCFG(rt,pair['raw_prompt'],cfg)
            sketch=corrector.prefill(pair['preview'],pair['condition'])
            edit,keep,target_f=target_weights(rt,pair['reference'],pair['target'],pair['valid'],pair['zero'],cfg)
            t=pair['anchor'].new_tensor(tau+(h-tau)*fraction)
            # Stage 1 matches the target velocity. Stage 2 rolls every tier out
            # through the frozen native tail and supervises the finished image.
            cold=cfg['stage']==1
            if cold:
                r=(t-tau)/(1-tau)
                z=(1-r)*pair['anchor']+r*pair['target']
                truth=(pair['target']-pair['anchor'])/(1-tau)
                base=native(z,t)
                proposal=(z+(1-t)*base).detach()
                delta=corrector(z,t,tau,sketch,proposal)
                prediction=base+delta
                error=(prediction-truth).square()
                loss=weighted(error,edit)+cfg['lambda_anchor']*weighted(delta.square(),keep)
                margin=(error.mean(1,keepdim=True)-(base.detach()-truth).square().mean(1,keepdim=True)+cfg['margin']).relu()
                loss=loss+cfg['lambda_improve']*weighted(margin,edit)
                loss=loss+weighted(error,1-pair['valid'])
            else:
                result,velocity_keep=target_rollout(rt,cfg,pair['anchor'],tau,native,corrector,sketch,keep)
                loss=image_loss(rt,result,pair['target'],pair['valid'],edit,keep,target_f,cfg)
                loss=loss+cfg['lambda_vel']*velocity_keep
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite target loss')
            (loss/batch).backward()
            totals[0]+=float(loss.detach())
            totals[1]+=int(not cold);totals[2]+=int(pair['zero'])
            del loss,sketch,pair,target_f
        dist.sum_gradients(corrector)
        norm = torch.nn.utils.clip_grad_norm_(params.values(),cfg["gradient_clip"])
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite target-correction gradient")
        for group in optimizer.param_groups:
            group["lr"] = cfg["lr"]*min(1.,step/max(1,cfg["warmup_steps"]))
        optimizer.step()
        with torch.no_grad():
            for n,p in params.items():
                ema[n].lerp_(p.detach().float(),1-cfg["ema_decay"])
        flow_sum,rollouts,zeros = dist.sum_statistics(totals,device)
        record = dict(step=step,flow_loss=flow_sum/batch,
                      rollout_samples=int(rollouts),zero_samples=int(zeros),gradient_norm=float(norm),
                      lr=optimizer.param_groups[0]["lr"],seconds=time.monotonic()-begin,
                      global_batch=batch,world=world)
        if dist.rank()==0:
            print(json.dumps(record),flush=True)
            with (out/"train.jsonl").open("a") as stream:
                stream.write(json.dumps(record)+"\n")
        if step==1 or step%cfg["save_every"]==0 or step==cfg["steps"]:
            save(out/f"step_{step:06d}",corrector,optimizer,ema,sampler,step,cfg,identity,device,kind=KIND)
    dist.barrier()


def initialize_weights(path,corrector,rt,cfg,*,ema):
    path=Path(path)
    meta=json.loads((path/'manifest.json').read_text())
    keys=('kind','lora_rank','freeze_gen_tail','width','height','t_eps','cfg_scale','cfg_norm',
          'cfg_interval','deep_time','shallow_time','read_time','handoff_time','num_steps','timestep_shift',
          'rho_deep','rho_shallow')
    if meta['kind']!=KIND or meta['identity']['native']!=rt.identity or any(meta['config'][k]!=cfg[k] for k in keys):
        raise ValueError('Target checkpoint architecture/native/trajectory differs')
    saved=torch.load(str(path/'rank_00000.pt'),map_location='cpu',weights_only=False,mmap=True)
    weights=saved['ema'] if ema else saved['weights']
    params=canonical_parameters(corrector)
    if params.keys()!=weights.keys():raise ValueError('Target adapter inventory differs')
    with torch.no_grad():
        for name,p in params.items():p.copy_(weights[name].to(p))
    return meta


def load_for_inference(path,rt,cfg,*,ema=False):
    corrector=SketchResidualCorrector(rt,cfg).eval()
    meta=initialize_weights(path,corrector,rt,cfg,ema=ema)
    if meta['config']['stage']!=2:
        raise ValueError('Stage 1 is cold-start only; deploy a stage 2 checkpoint')
    return corrector,meta
