"""Target-driven correction: train, trace native failures, sample, or judge offline."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest='command',required=True)
    j=subs.add_parser('judge')
    for key in ('requests','output','base-url','model','key-env'):j.add_argument('--'+key,required=True)
    for name in ('train','trace','sample'):
        s=subs.add_parser(name)
        for key in ('model','source','config','output'):s.add_argument('--'+key,required=True)
        s.add_argument('--device',default='npu:0');s.add_argument('--distributed',action='store_true')
        if name=='train':
            s.add_argument('--corpus',required=True);s.add_argument('--resume');s.add_argument('--init-checkpoint')
        else:
            s.add_argument('--prompts',required=True);s.add_argument('--limit',type=int,default=32)
        if name=='sample':
            s.add_argument('--checkpoint',required=True);s.add_argument('--ema',action='store_true')
            s.add_argument('--compare-native',action='store_true')
    a=p.parse_args()
    if a.command=='judge':
        from .judge import judge
        judge(a.requests,a.output,a.base_url,a.model,a.key_env);return
    import torch
    from . import distributed as dist
    from .accelerator import set_device
    from .loading import load_native
    from .images import save_image
    from .field import validate_config
    cfg=json.loads(Path(a.config).read_text());validate_config(cfg)
    device=dist.initialize(a.device) if a.distributed else set_device(a.device)
    if a.command=='train':
        from .train import train
        train(a,cfg,device)
    else:
        rt=load_native(a.model,a.source,cfg,device,a.distributed)
        out=Path(a.output)
        if dist.rank()==0:out.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        from .sampling import run,native_trace
        if a.command=='sample':
            from .train import load_for_inference
            corrector,meta=load_for_inference(a.checkpoint,rt,cfg,ema=a.ema)
        rows=[json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()][:a.limit]
        for i in range(dist.rank(),len(rows),dist.world_size()):
            row=rows[i];seed=row.get('seed',cfg['seed']+i)
            questions=row if 'questions' in row else None
            raw_prompt=row.get('raw_prompt') or row.get('prompt')
            if not raw_prompt:
                raise ValueError(f"Row {i} is missing 'raw_prompt' or 'prompt'")
            uid=row.get('uid') or row.get('prompt_id') or f'{i:05d}'
            if a.command=='trace':
                trace=native_trace(rt,raw_prompt,seed,cfg,questions)
                torch.save(trace,out/f'{i:05d}.pt');record=trace['record']
                images={'preview':trace['preview'],'fail':trace['final']}
            else:
                images,record=run(rt,raw_prompt,seed,cfg,corrector=corrector,question_record=questions,compare_native=a.compare_native)
                record.update(checkpoint=a.checkpoint,step=meta['step'],ema=a.ema)
            record['uid']=uid
            for key,image in images.items():save_image(out/f'{i:05d}_{key}.png',image)
            (out/f'{i:05d}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
        dist.barrier()
        if dist.rank()==0:(out/'complete.json').write_text(json.dumps(dict(samples=len(rows))))
    if a.distributed:torch.distributed.destroy_process_group()


if __name__=='__main__':main()
