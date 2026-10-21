#!/usr/bin/env python3
"""Inspect deterministic Stage A clause dropout on existing paired v1/v2 data."""
import argparse
from collections import Counter
import json
from pathlib import Path
from qwen_latent_cot.interaction.data import GenerationCorpus, CorpusPairs
from qwen_latent_cot.interaction.stage_a import split_prompt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',required=True)
    parser.add_argument('--config',default='configs/interaction_stage_a.json')
    parser.add_argument('--count',type=int,default=4096)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    corpus=GenerationCorpus(args.corpus)
    pairs=CorpusPairs(corpus,dict(cfg,sampler_steps=50,timestep_shift=3))
    sampler=pairs.sampler()
    reasons,sources=Counter(),Counter()
    words,removed,examples=[],[],[]
    for i,row in enumerate(sampler.take(args.count)):
        split=split_prompt(row['caption'],sampler.rng.randrange(2**63-1),
                           drop_fraction=cfg['clause_drop_fraction'],full_probability=cfg['full_prompt_probability'],
                           min_words=cfg['min_prompt_words'])
        reasons[split.reason]+=1;sources[row['source']]+=1;words.append(len(split.full.split()))
        if split.removed: removed.append(len(split.removed.split())/max(1,len(split.full.split())))
        if len(examples)<24 and (split.removed or len(examples)<4):
            examples.append(dict(sample_id=row['sample_id'],family_id=row['family_id'],source=row['source'],
                                 image=row['image'],full=split.full,partial=split.partial,removed=split.removed,
                                 reason=split.reason,dropped_units=split.dropped))
    ordered=sorted(words)
    report=dict(corpus=args.corpus,format=corpus.manifest['format'],fingerprint=corpus.fingerprint,
                available_generation_pairs={s:sum(len(v) for v in corpus.groups(s)) for s in ('train','val','test')},
                sampled=len(words),seed=cfg['seed'],reasons=dict(reasons),sources=dict(sources),
                words={str(q):ordered[min(len(ordered)-1,int(q*len(ordered)))] for q in (0,.25,.5,.75,.95,1)},
                mean_removed_fraction_when_masked=sum(removed)/len(removed) if removed else 0.,examples=examples)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='examples'},ensure_ascii=False,indent=2))
    corpus.close()


if __name__=='__main__':main()
