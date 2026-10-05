#!/usr/bin/env python
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from qwen_latent_cot.evaluation.io import read_jsonl
from qwen_latent_cot.evaluation.probe_results import summarize_probe
p=argparse.ArgumentParser(description='Merge native Memory QA and independently observed image labels')
p.add_argument('--qa-dirs',nargs='+',required=True);p.add_argument('--label-dirs',nargs='+',required=True)
p.add_argument('--output-dir',required=True);a=p.parse_args()
rows=[r for d in a.qa_dirs for r in read_jsonl(Path(d)/'qa.jsonl')]
labels=[r for d in a.label_dirs for r in read_jsonl(Path(d)/'labels.jsonl')]
bindings=[json.loads((Path(d)/'label_run.json').read_text()) for d in a.label_dirs]
for b in bindings:
    for key in ('qa','judge','minimum_confidence'):
        if b[key]!=bindings[0][key]:raise ValueError('label bindings differ')
qa=bindings[0]['qa'];run=qa['run']
for d in a.qa_dirs:
    b=json.loads((Path(d)/'qa_run.json').read_text())
    if {k:v for k,v in b.items() if k!='shard'}!=qa:raise ValueError('QA provenance differs from labeled QA')
expected={(p,s,step,q) for p,qs in qa['expected_question_ids'].items() for q in qs for s in run['seeds'] for step in run['probe_steps']}
from qwen_latent_cot.evaluation.probe_results import row_key
if {row_key(r) for r in rows}!=expected:raise ValueError('incomplete native QA coverage')
result=summarize_probe(rows,labels);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
result['provenance']={k:v for k,v in bindings[0].items() if k!='shard'}
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Native UND Memory QA probe','','This diagnoses information and readout, not final T2I quality. Unknown labels are excluded from accuracy and counted separately.','',
       '|Source|Observed accuracy|Accuracy when prompt differs from image|Prompt echo on mismatches|',
       '|---|---:|---:|---:|']
for source,v in result['by_scope'].get('all',{}).items():
    fmt=lambda x:'unknown' if x is None else f'{x:.4f}'
    lines.append(f"|{source}|{fmt(v['accuracy_vs_observed'])}|{fmt(v['accuracy_on_prompt_actual_mismatch'])}|{fmt(v['prompt_echo_rate_on_mismatch'])}|")
lines+=['',f"Known labels: {result['known_labels']}; unknown: {result['unknown_labels']}; prompt/image mismatches: {result['prompt_actual_mismatches']}."]
(out/'summary.md').write_text('\n'.join(lines)+'\n')
print(out/'summary.md')
