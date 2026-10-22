"""Pure U1.5 RGB T2I, hidden-caption inheritance and guidance-safe correction."""
import argparse
import json
import os
from pathlib import Path
import random
import torch
from ..bagel.accelerator import set_device
from ..correction import distributed
from .runtime import Runtime,upstream
from . import checkpoint


def config(path):
    c=json.loads(Path(path).read_text())
    if not 0<c['active_steps']<=c['num_steps']:raise ValueError('Invalid correction interval')
    if c['cfg_norm'] not in ('none','global','channel'):raise ValueError('Unsupported CFG norm')
    if not 0<=c['text_dropout']<=1 or c['global_batch']<1:raise ValueError('Invalid training configuration')
    return c


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['native','sample','t2i','caption','correct','describe','prepare','export'])
    parser.add_argument('--source',help='Pinned SenseNova-U1 repository root')
    parser.add_argument('--model');parser.add_argument('--device',default='npu:0')
    parser.add_argument('--config',default=str(Path(__file__).resolve().parents[2]/'configs/u15/correction.json'))
    parser.add_argument('--output',required=True);parser.add_argument('--checkpoint');parser.add_argument('--gen-weights')
    parser.add_argument('--reader');parser.add_argument('--corpus');parser.add_argument('--pairs');parser.add_argument('--resume')
    parser.add_argument('--prompt');parser.add_argument('--prompts');parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--arm',choices=['native','global','corrected'],default='corrected')
    parser.add_argument('--distributed',action='store_true');parser.add_argument('--ema',action='store_true')
    parser.add_argument('--node',type=int,default=12)
    parser.add_argument('--evaluation-manifests',type=Path)
    parser.add_argument('--source-off',action='store_true');parser.add_argument('--swap-seed',type=int)
    a=parser.parse_args(argv)
    if a.command=='export':checkpoint.export(a.checkpoint,a.output,a.ema);return
    if not a.model or not a.source:parser.error('--model and --source are required')
    cfg=config(a.config)
    if a.command=='correct' and not a.reader:parser.error('--reader is required for correction training')
    if a.command in ('describe','sample') and (a.command=='describe' or a.arm!='native') and not a.checkpoint:parser.error('--checkpoint is required')
    if a.command=='prepare':
        upstream(a.source)
        from transformers import AutoTokenizer
        from ..t2i.data import build
        tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
        settings=dict(min_short_side=min(cfg['width'],cfg['height']),max_long_side=max(cfg['width'],cfg['height']),max_sequence_length=40960,expected_geometry=[cfg['width'],cfg['height']])
        if not a.evaluation_manifests:parser.error('--evaluation-manifests is required for U1.5 corpus preparation')
        from .data import evaluation_prompts
        forbidden,benchmarks=evaluation_prompts(a.evaluation_manifests)
        print(json.dumps(build(a.pairs,a.output,tokenizer,settings,cfg['seed'],excluded_prompts=forbidden,excluded_benchmarks=benchmarks),indent=2));return
    device=distributed.initialize(a.device) if a.distributed else set_device(a.device)
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    # Full native weights first on CPU for sharded GEN SFT; other modes BF16 on device.
    load_device='cpu' if a.command=='t2i' else str(device)
    rt=Runtime.load(a.model,a.source,cfg,load_device,a.gen_weights);rt.device=device
    if a.command in ('t2i','caption','correct'):
        from .training import train
        if not a.corpus:parser.error('--corpus is required')
        train(rt,cfg,a.command,a.corpus,a.output,a.reader,a.resume)
        if distributed.world_size()>1:torch.distributed.destroy_process_group()
        return
    from .readout import Corrector
    from .sampling import sample
    corrector=None
    if a.command=='describe':
        from .reader import SemanticReader
        reader=SemanticReader(rt,cfg);checkpoint.load_portable(a.checkpoint,reader,rt,'caption')
        with torch.no_grad():
            prefix=rt.prefix(a.prompt);reference=rt.prefix(None);t=rt.times()[a.node]
            noise=rt.noise_scale()*torch.randn((1,3,rt.height,rt.width),device=device,dtype=next(rt.model.parameters()).dtype,
                generator=torch.Generator(device=device).manual_seed(a.swap_seed if a.swap_seed is not None else a.seed))
            # Observe an ACTUAL native rollout, not fresh noise labelled as node12.
            for index in range(a.node):
                left,right=rt.times()[index:index+2]
                cond=rt.predict(noise,left,prefix).velocity
                uncond=rt.predict(noise,left,reference).velocity if cfg['cfg_scale']>1 else cond
                noise=noise+(right-left)*rt.guided(cond,uncond,cfg['cfg_scale'])
            observed=rt.predict(noise,t,prefix,capture=True).observation
            if a.source_off:observed=tuple(torch.zeros_like(x) for x in observed)
            result=dict(source_off=a.source_off,swap_seed=a.swap_seed,prompt=a.prompt,seed=a.seed,node=a.node,t=float(t),caption=reader.describe(observed,t,cfg['caption_max_tokens']))
        Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2));return
    arm='native' if a.command=='native' else a.arm
    if arm!='native':
        corrector=Corrector(rt,cfg);corrector.initialize_global(cfg['active_steps'])
        saved=checkpoint.load_portable(a.checkpoint,corrector,rt,'correct')
    output=Path(a.output)
    if a.prompts:
        prompts=[json.loads(x) for x in Path(a.prompts).read_text().splitlines() if x.strip()]
        output.mkdir(parents=True,exist_ok=False);records=[]
        for i,row in enumerate(prompts):
            prompt=row.get('prompt',row.get('instruction'));seed=row.get('seed',a.seed)
            image,trace=sample(rt,prompt,seed,corrector,arm);image.save(output/f'{i:04d}.png')
            records.append(dict(input=row,seed=seed,arm=arm,image=f'{i:04d}.png',trace=trace))
        (output/'records.json').write_text(json.dumps(dict(identity=rt.identity,config=cfg,records=records),ensure_ascii=False,indent=2))
    else:
        if not a.prompt:parser.error('--prompt or --prompts is required')
        image,trace=sample(rt,a.prompt,a.seed,corrector,arm)
        output.parent.mkdir(parents=True,exist_ok=True);image.save(output)
        output.with_suffix('.json').write_text(json.dumps(dict(identity=rt.identity,config=cfg,prompt=a.prompt,seed=a.seed,arm=arm,trace=trace),indent=2))


if __name__=='__main__':main()
