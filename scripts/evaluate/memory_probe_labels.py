#!/usr/bin/env python
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from qwen_latent_cot.evaluation.io import read_jsonl,sha256
from qwen_latent_cot.evaluation.probe_results import row_key,label_answer
from qwen_latent_cot.evaluation.scoring import LocalScorer


def main():
    p=argparse.ArgumentParser(description='Label actual x0 proxy content, without the desired prompt/answer')
    p.add_argument('--qa-dirs',nargs='+',required=True);p.add_argument('--judge-model',required=True)
    p.add_argument('--geneval2-source',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--minimum-confidence',type=float,default=.8)
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    a=p.parse_args()
    if not 0<=a.shard_index<a.num_shards or not 0<=a.minimum_confidence<=1:raise ValueError('invalid shard/confidence')
    values={};bindings=[]
    for d in a.qa_dirs:
        path=Path(d);bindings.append(json.loads((path/'qa_run.json').read_text()))
        for r in read_jsonl(path/'qa.jsonl'):
            if r['source']!='DYNAMIC':continue
            key=row_key(r)
            if key in values:raise ValueError('duplicate QA identity')
            values[key]=r
    for b in bindings:
        for key in ('run','qa_source_sha256','max_questions','max_count','sources','expected_question_ids','memory_arm','full_prompt_kv_read','seed_reads_selected_prompt_kv'):
            if b.get(key)!=bindings[0].get(key):raise ValueError('incompatible QA provenance')
    qa=bindings[0];run=qa['run']
    expected={(p,s,step,q) for p,qs in qa['expected_question_ids'].items() for q in qs for s in run['seeds'] for step in run['probe_steps']}
    if set(values)!=expected:raise ValueError('incomplete native QA coverage')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    judge=LocalScorer(a.judge_model,a.geneval2_source,a.device)
    binding={'qa':{k:v for k,v in bindings[0].items() if k!='shard'},'judge':judge.provenance,
        'minimum_confidence':a.minimum_confidence,'shard':[a.shard_index,a.num_shards],'desired_prompt_given_to_judge':False}
    path=out/'label_run.json'
    if path.exists() and json.loads(path.read_text())!=binding:raise ValueError('label run binding changed')
    path.write_text(json.dumps(binding,indent=2)+'\n')
    labels=out/'labels.jsonl';cached=read_jsonl(labels) if labels.exists() else []
    done={row_key(r):r for r in cached}
    if len(done)!=len(cached):raise ValueError('duplicate observed label cache')
    for key,r in done.items():
        if key not in values or r['image_sha256']!=values[key]['image_sha256'] or r['candidates']!=values[key]['candidates']:
            raise ValueError('observed label cache differs from QA')
    for ordinal,(key,row) in enumerate(sorted(values.items())):
        if ordinal%a.num_shards!=a.shard_index:continue
        if sha256(row['image_path'])!=row['image_sha256']:raise ValueError('observed image changed')
        if key in done:continue
        question=('Inspect only the visible image. This may be an incomplete noisy image estimate. '+row['question']+
            '\nAllowed answers: '+', '.join(row['candidates'])+'. Use unknown when the content is not reliably discernible. '
            'Return only JSON: {"answer": "one allowed answer", "confidence": 0.0}. Confidence must lie between 0 and 1.')
        raw=judge.answer(row['image_path'],question)
        payload=json.loads(raw.removeprefix('```json').removeprefix('```').removesuffix('```').strip())
        record={k:row[k] for k in ('prompt_id','seed','step','question_id','image_path','image_sha256','candidates','image_kind')}
        record.update(label_answer(payload,row['candidates'],a.minimum_confidence));record['raw_judge_response']=raw
        with labels.open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
        print(f'observed label {key}: {record["observed_answer"]}',flush=True)


if __name__=='__main__':main()
