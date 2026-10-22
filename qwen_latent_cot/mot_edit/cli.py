"""Prepare exact pairs, train native-head internal edit loops, sample checkpoints."""
import argparse
import json
from pathlib import Path
import random
import torch
from . import checkpoint
from ..correction import distributed as parallel


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--pairs',required=True);prep.add_argument('--output',required=True)
    for name in ('train','sample'):
        c=sub.add_parser(name);c.add_argument('--model',required=True);c.add_argument('--device',default='npu:0');c.add_argument('--output',required=True)
        c.add_argument('--config',default='configs/mot_edit_8000.json')
        if name=='train':
            c.add_argument('--corpus',required=True);c.add_argument('--distributed',action='store_true');c.add_argument('--resume')
        else:
            c.add_argument('--checkpoint',required=True);c.add_argument('--prompt',required=True);c.add_argument('--seed',type=int,default=17)
            c.add_argument('--rounds',type=int,default=2)
    a=p.parse_args()
    if a.command=='prepare':
        from .data import prepare
        print(json.dumps(prepare(a.pairs,a.output),indent=2));return
    from ..bagel.accelerator import set_device
    from ..bagel.backbone import load_native
    from ..correction.runtime import Runtime
    from .model import MoTEdit
    cfg=json.loads(Path(a.config).read_text())
    saved=checkpoint.read(a.checkpoint if a.command=='sample' else a.resume) if a.command=='sample' or a.resume else None
    if a.command=='sample':cfg=saved['config']
    device=parallel.initialize(a.device) if getattr(a,'distributed',False) else set_device(a.device)
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    rt=Runtime(load_native(a.model,device,cfg['timestep_shift']),cfg);model=MoTEdit(rt,cfg).eval()
    print(json.dumps(dict(event='loaded',rank=parallel.rank(),roles={r:sum(p.numel() for p in getattr(model,r).adapters().values()) for r in ('observation','feedback','editor')})),flush=True)
    if a.command=='sample':
        from .sampling import sample
        checkpoint.apply(saved,model);model.set_stage('edit')
        image,trace=sample(model,a.prompt,a.seed,a.rounds);Path(a.output).parent.mkdir(parents=True,exist_ok=True)
        image.save(a.output);Path(a.output+'.json').write_text(json.dumps(trace,indent=2)+'\n');return
    from .data import Corpus
    from .training import train
    corpus=Corpus(a.corpus);revision=Path(__file__).resolve().parents[2]/'DEPLOYED_COMMIT'
    provenance=dict(source_revision=revision.read_text().strip() if revision.exists() else 'workspace',model_path=str(Path(a.model).resolve()),
        corpus_path=str(Path(a.corpus).resolve()),corpus_identity=corpus.manifest['identity'])
    try:train(model,corpus,a.output,provenance,saved)
    finally:
        if parallel.world_size()>1:torch.distributed.destroy_process_group()


if __name__=='__main__':main()
