"""Register all annotation sources and build task indexes without image access."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_latent_cot.diagnostics.corpus import build


def registry_for(unimoe_root, old_root):
    new=Path(unimoe_root);old=Path(old_root);sources=[]
    def add(path,kind,**options):
        if path.is_file():sources.append(dict(name=str(path.relative_to(old) if path.is_relative_to(old) else path.stem).replace('/','__').replace('.jsonl',''),path=str(path.resolve()),format=kind,**options))
    base=old/'qwen_latent_cot_v3_adapted_20260816/manifests'
    for split in ('train','val','test'):add(base/f'cort36k_{split}.jsonl','cort',split=split,family_namespace='cort')
    for p in sorted(base.glob('cort114*.jsonl')):add(p,'cort',inherit_split=True,family_namespace='cort114')
    for p in sorted(base.glob('stage12_stage2_*.jsonl')):
        add(p,'cort',split=p.stem.rsplit('_',1)[-1],family_namespace='cort')
    for p in sorted((old/'cort_sft_133k/manifests').glob('*.jsonl')):
        add(p,'nofix' if 'nofix' in p.name else 'legacy_manifest',family_namespace='cort114')
    for p in sorted((old/'entry_split').glob('*.jsonl')):
        if p.name=='echo4o_raw_manifest.jsonl':add(p,'echo_raw',family_namespace='echo4o')
        else:
            images=old/'qwen_latent_cot_v3_adapted_20260816/images/cort36k' if p.name.startswith('cort_sft_consis_36k_') else old/'entry_split/images'
            add(p,'entry',inherit_split=True,family_namespace='cort',images_root=str(images))
    for directory in ('full_seed42_v2','smoke4k_seed42_v2'):
        for p in sorted((old/'phase1_structured'/directory).glob('*_candidates.jsonl')):add(p,'structured',family_namespace='cort')
    for p in sorted(old.glob('*.jsonl')):
        if p.name.endswith('.neutral_prompt.jsonl'):continue
        diagnostic=p.name.startswith(('memory_gen','memory_response','visual_source','c1_text','c1_response'))
        # Historical text feedback is a text diagnostic, not a second visual replay pool.
        diagnostic=diagnostic or p.name.startswith('c1_')
        add(p,'paired',diagnostic=diagnostic,inherit_split=diagnostic,family_namespace='cort')
    for folder in (new,new/'data'):
        for p in sorted(folder.glob('unimoe2.5_*_fixed.json')):add(p,'unimoe')
    roots={'/mnt/sfs/unimoe_dataset/':'/data/unimoe_dataset/',
           '/data/bagel-LatentCoT/datasets/':str(old)+'/',
           '/private/codes/exp/deep_learning/data/datasets/cort/':str(old)+'/',
           '/private/codes/exp/deep_learning/data/datasets/':str(old)+'/',
           '/Echo-4o-Image/':str(old/'Echo-4o-Image')+'/'}
    roots[str(old/'entry_split/images')+'/']=str(old/'qwen_latent_cot_v3_adapted_20260816/images/cort36k')+'/'
    roots[str(old/'entry_split/images/cort_sft_consis_36k')+'/']=str(old/'qwen_latent_cot_v3_adapted_20260816/images/cort36k')+'/'
    return dict(sources=sources,roots=roots,context_token_limit=4096)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unimoe-root',default='/data/zx_wang_shared_work/ModelScope-zx/zx_unzip/mnt/1/visual_gen')
    parser.add_argument('--old-root',default='/data/zyd_workspace/bagel-LatentCoT/datasets')
    parser.add_argument('--registry',required=True);parser.add_argument('--output');parser.add_argument('--model')
    parser.add_argument('--shard-size',type=int,default=100000)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--progress-every',type=int,default=100000)
    args=parser.parse_args();registry=registry_for(args.unimoe_root,args.old_root)
    if not registry['sources']:parser.error('no source manifests found')
    Path(args.registry).write_text(json.dumps(registry,indent=2)+'\n')
    print(f'Registered {len(registry["sources"])} sources',flush=True)
    if args.output:
        if not args.model:parser.error('build requires --model for native BAGEL token lengths')
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.model)
        result=build(registry,args.output,
                     batch_tokenizer=lambda texts:[len(ids) for ids in tokenizer(texts,add_special_tokens=False)['input_ids']],
                     shard_size=args.shard_size,batch_size=args.batch_size,progress_every=args.progress_every)
        print(json.dumps(dict(counts=result['counts'],fingerprint=result['fingerprint']),indent=2))


if __name__=='__main__':main()
