#!/usr/bin/env python
"""Validate existing full-static images against paired Base; no inference."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from qwen_latent_cot.evaluation.io import load_manifests,identity
from qwen_latent_cot.evaluation.loop_depth import mode_of


def validate(records,arms):
    groups={arm:{} for arm in arms}
    for row in records:groups[row['arm']][identity(row)]=row
    base=groups['BASE'];results={}
    for arm,group in groups.items():
        if mode_of(arm)!='LAYERWISE_FULL_SEED_REPLACE':continue
        if set(group)!=set(base):raise ValueError('full static paired coverage mismatch')
        mismatches=[]
        for key,row in group.items():
            reference=base[key]
            same_inputs=all(row[k]==reference[k] for k in ('prompt','noise_sha256','height','width'))
            if not (same_inputs and row['valid_file'] and reference['valid_file']
                    and row['image_sha256']==reference['image_sha256']):mismatches.append(list(key))
        results[arm]={'paired_images':len(group),'identical_images':len(group)-len(mismatches),'mismatches':mismatches}
    if not results:raise ValueError('full static arms missing')
    return {'passed':all(not value['mismatches'] for value in results.values()),'arms':results}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests',nargs='+',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();records,run=load_manifests(args.manifests)
    result=validate(records,run['arms']);result['source_sha256']=run['source_sha256']
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
