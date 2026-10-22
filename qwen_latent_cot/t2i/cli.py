"""Prepare teacher pairs, train original GEN, export EMA, and sample natively."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from ..bagel.accelerator import set_device


def read_config(path):
    c=json.loads(Path(path).read_text())
    for key in ('steps','global_batch','save_every','validate_every','val_images','max_sequence_length'):
        if not isinstance(c[key],int) or c[key]<1:raise ValueError(key+' must be a positive integer')
    if not 0<=c['text_dropout']<=1 or not 0<=c['ema_decay']<1:raise ValueError('Invalid dropout/EMA')
    if c['lr']<=0 or c['eps']<=0 or c['timestep_shift']<=0:raise ValueError('Invalid optimizer/time recipe')
    return c


def main():
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare');prepare.add_argument('--pairs',required=True);prepare.add_argument('--output',required=True)
    prepare.add_argument('--model',required=True);prepare.add_argument('--config',default='configs/t2i_sft.json')
    train=commands.add_parser('train');train.add_argument('--model',required=True);train.add_argument('--corpus',required=True)
    train.add_argument('--output',required=True);train.add_argument('--resume');train.add_argument('--config',default='configs/t2i_sft.json')
    train.add_argument('--device',default='npu:0');train.add_argument('--steps',type=int)
    export=commands.add_parser('export');export.add_argument('--checkpoint',required=True);export.add_argument('--output',required=True)
    export.add_argument('--online',action='store_true')
    sample=commands.add_parser('sample');sample.add_argument('--model',required=True);sample.add_argument('--weights')
    sample.add_argument('--prompt',required=True);sample.add_argument('--output',required=True);sample.add_argument('--device',default='npu:0')
    sample.add_argument('--height',type=int,default=1024);sample.add_argument('--width',type=int,default=1024)
    sample.add_argument('--seed',type=int,default=17);sample.add_argument('--steps',type=int,default=50)
    sample.add_argument('--cfg',type=float,default=4.)
    a=p.parse_args()
    if a.command=='prepare':
        from transformers import Qwen2Tokenizer
        from .data import build
        config=read_config(a.config)
        print(json.dumps(build(a.pairs,a.output,Qwen2Tokenizer.from_pretrained(a.model),config,config['seed']),indent=2));return
    if a.command=='export':
        from .checkpoint import export
        print(export(a.checkpoint,a.output,not a.online));return
    from ..bagel.backbone import load_native
    if a.command=='sample':
        from .sampling import load_overlay,sample
        device=set_device(a.device);bundle=load_native(a.model,device,4.)
        if a.weights:load_overlay(bundle,a.weights)
        image=sample(bundle,a.prompt,a.seed,a.height,a.width,a.steps,cfg_scale=a.cfg)
        output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);image.save(output);return
    from .training import train as run
    config=read_config(a.config)
    if a.steps is not None:
        if a.steps<1:raise ValueError('--steps must be positive')
        config['steps']=a.steps
    if Path(a.output).exists() and not a.resume:raise FileExistsError('Use a fresh output path; resume requires --resume')
    if a.resume and not (Path(a.resume)/'manifest.json').exists():raise ValueError('Checkpoint is incomplete')
    kind=a.device.split(':')[0];world=int(os.environ.get('WORLD_SIZE','1'))
    device=set_device(f'{kind}:{os.environ["LOCAL_RANK"]}' if world>1 and kind!='cpu' else a.device)
    if world>1:
        dist.init_process_group({'npu':'hccl','cuda':'nccl','cpu':'gloo'}[kind],timeout=timedelta(minutes=30))
        if kind=='cpu':raise ValueError('Full-shard distributed training requires an accelerator; CPU is single-rank test oracle')
    try:
        # Load on host. FSDP moves and shards one native block at a time.
        bundle=load_native(a.model,'cpu',config['timestep_shift'])
        run(bundle,a.corpus,a.output,config,device,a.resume)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
