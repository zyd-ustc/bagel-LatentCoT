"""Natural image-text spatial correction: train and paired sampling."""
import argparse
import json
from pathlib import Path
import random
import torch
from . import checkpoint,distributed as parallel
from .model import SpatialCorrector
from .objective import guidance_safe
from .runtime import Runtime
from ..bagel.accelerator import set_device
from ..bagel.backbone import load_native


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('train','sample'):
        p=sub.add_parser(name);p.add_argument('--model',required=True);p.add_argument('--device',default='npu:0');p.add_argument('--output',required=True)
        p.add_argument('--config',default='configs/correction.json')
        if name=='train':
            p.add_argument('--corpus',required=True);p.add_argument('--resume');p.add_argument('--distributed',action='store_true')
            p.add_argument('--steps',type=int);p.add_argument('--global-batch',type=int);p.add_argument('--val-images',type=int)
        else:
            p.add_argument('--checkpoint',required=True);p.add_argument('--prompt',required=True);p.add_argument('--seed',type=int,default=17);p.add_argument('--native',action='store_true');p.add_argument('--global-only',action='store_true')
    args=parser.parse_args(argv)
    saved=checkpoint.read(args.checkpoint if args.command=='sample' else args.resume) if args.command=='sample' or args.resume else None
    cfg=dict(saved['config']) if args.command=='sample' else json.loads(Path(args.config).read_text())
    if args.command=='train':
        for name in ('steps','global_batch','val_images'):
            if getattr(args,name) is not None:cfg[name]=getattr(args,name)
        if cfg['global_batch']%int(__import__('os').environ.get('WORLD_SIZE','1')):raise ValueError('Batch must divide world size')
    guidance_safe(cfg)
    device=parallel.initialize(args.device) if getattr(args,'distributed',False) else set_device(args.device)
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    runtime=Runtime(load_native(args.model,device,cfg['timestep_shift']),cfg)
    corrector=SpatialCorrector(runtime,cfg).eval()
    print(json.dumps(dict(event='loaded',rank=parallel.rank(),trainable=sum(p.numel() for p in corrector.parameters() if p.requires_grad))),flush=True)
    if args.command=='sample':
        from .sampling import sample
        if saved['provenance']['model_path']!=str(Path(args.model).resolve()):raise ValueError('Sampling model path differs from training')
        checkpoint.apply(saved,corrector)
        image,trace=sample(corrector,args.prompt,args.seed,native=args.native,global_only=args.global_only)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True);image.save(args.output)
        Path(args.output+'.json').write_text(json.dumps(trace,indent=2));return
    from .data import Corpus
    from .training import train
    revision=Path(__file__).resolve().parents[2]/'DEPLOYED_COMMIT'
    source_revision=revision.read_text().strip() if revision.exists() else 'workspace'
    provenance=dict(source_revision=source_revision,model_path=str(Path(args.model).resolve()),corpus_path=str(Path(args.corpus).resolve()),
                    corpus_fingerprint=json.loads((Path(args.corpus)/'manifest.json').read_text())['fingerprint'],
                    initialization='native_pretrained_UND')
    train(corrector,Corpus(args.corpus),args.output,provenance,saved=saved)
    if parallel.world_size()>1:torch.distributed.destroy_process_group()


if __name__=='__main__':main()
