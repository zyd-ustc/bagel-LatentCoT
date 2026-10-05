#!/usr/bin/env python
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from qwen_latent_cot.evaluation.io import load_manifests,read_jsonl,identity
from quality_report import write_summary


def main():
    p=argparse.ArgumentParser(description='Validate all scoring shards and build paired quality/semantic report')
    p.add_argument('--manifests',nargs='+',required=True);p.add_argument('--score-dirs',nargs='+',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--bootstrap-replicates',type=int,default=10000)
    a=p.parse_args();records,run=load_manifests(a.manifests)
    expected={(r['arm'],*identity(r)):r for r in records};scored={};bindings=[]
    for d in a.score_dirs:
        path=Path(d);bindings.append(json.loads((path/'scorer.json').read_text()))
        for r in read_jsonl(path/'scores.jsonl'):
            key=(r['arm'],*identity(r))
            if key in scored or key not in expected:raise ValueError('unknown or duplicate scoring identity')
            if r['image_sha256']!=expected[key]['image_sha256']:raise ValueError('scored image changed')
            scored[key]=r
    if set(scored)!=set(expected):raise ValueError('incomplete scoring; no missing-pair removal')
    for binding in bindings:
        if binding['run']!=run:raise ValueError('scoring provenance differs from generation')
        if binding['run']!=bindings[0]['run'] or binding['scorer']!=bindings[0]['scorer']:raise ValueError('incompatible scoring provenance')
    output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True)
    values=[scored[k] for k in sorted(scored)]
    (output/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
    (output/'scorer.json').write_text(json.dumps({'run':run,'scorer':bindings[0]['scorer'],'scoring_shards':len(bindings)},indent=2)+'\n')
    write_summary(values,run,output,a.bootstrap_replicates)


if __name__=='__main__':main()
