#!/usr/bin/env python
"""Offline full-depth native UND QA. Does not change the generation trajectory."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from qwen_latent_cot.evaluation.io import load_manifests,read_jsonl,sha256,source_hash,identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests',nargs='+',required=True);p.add_argument('--benchmark',required=True)
    p.add_argument('--model-path',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--max-questions',type=int,default=4);p.add_argument('--max-count',type=int,default=12)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    if not 0<=args.shard_index<args.num_shards:raise ValueError('invalid shard')
    records,run=load_manifests(args.manifests)
    if sha256(args.benchmark)!=run['benchmark_sha256']:raise ValueError('benchmark changed')
    data=read_jsonl(args.benchmark)
    alljobs=sorted([r for r in records if r['arm']=='MEMORY_DYNAMIC'],key=identity)
    jobs=[r for i,r in enumerate(alljobs) if i%args.num_shards==args.shard_index]
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    from qwen_latent_cot.bagel.backbone import load_native
    from qwen_latent_cot.bagel.memory_probe import NativeMemoryQA,validate_capture,snapshot_sources,probe_questions
    from PIL import Image
    bundle=load_native(args.model_path,args.device)
    for name,digest in run['model_sha256'].items():
        if sha256(Path(args.model_path)/name)!=digest:raise ValueError('QA checkpoint differs from generation')
    reader=NativeMemoryQA(bundle)
    binding={'run':run,'qa_source_sha256':source_hash(ROOT),'max_questions':args.max_questions,
             'max_count':args.max_count,'shard':[args.shard_index,args.num_shards],
             'expected_question_ids':{r['prompt_id']:[q['question_id'] for q in probe_questions(data[r['index']],args.max_questions,args.max_count)] for r in alljobs},
             'sources':['DYNAMIC','SEED','EMPTY','VIT_IMAGE'],'prompt_kv_read':False,
             'question_format':'native_BOS_question_EOS_BOS_answer','answer_score':'mean_token_log_likelihood'}
    runfile=out/'qa_run.json'
    if runfile.exists() and json.loads(runfile.read_text())!=binding:raise ValueError('QA run binding changed')
    runfile.write_text(json.dumps(binding,indent=2)+'\n')
    resultfile=out/'qa.jsonl';done={}
    if resultfile.exists():
        for row in read_jsonl(resultfile):
            key=(row['prompt_id'],row['seed'],row['step'],row['question_id'],row['source'])
            if key in done:raise ValueError('duplicate QA record')
            done[key]=row
    for row in jobs:
        if 'probe_capture' not in row:raise ValueError('generation requires --probe-steps before QA')
        capture=validate_capture(row['probe_capture'],row['probe_capture_sha256'])
        if capture['noise_sha256']!=row['noise_sha256'] or capture['prompt_id']!=row['prompt_id'] or capture['seed']!=row['seed']:
            raise ValueError('capture/generation identity mismatch')
        questions=probe_questions(data[row['index']],args.max_questions,args.max_count)
        for snapshot in capture['snapshots']:
            sources=snapshot_sources(snapshot)
            with Image.open(snapshot['image_path']) as image:sources['VIT_IMAGE']=reader.image_cache(image)
            for question in questions:
                for source,kv in sources.items():
                    key=(row['prompt_id'],row['seed'],snapshot['step'],question['question_id'],source)
                    if key in done:continue
                    anchor=1 if source=='VIT_IMAGE' else snapshot['question_position_start']
                    result=reader.score(question['question'],question['candidates'],kv,anchor)
                    record={**question,**result,'prompt_id':row['prompt_id'],'seed':row['seed'],
                        'step':snapshot['step'],'progress':snapshot['sampling_progress'],'timestep':snapshot['timestep'],
                        'source':source,'image_path':snapshot['image_path'],'image_sha256':snapshot['image_sha256'],
                        'noise_sha256':row['noise_sha256'],'capture_sha256':row['probe_capture_sha256'],
                        'image_kind':snapshot['observed_image_kind'],'prompt_kv_read':False}
                    with resultfile.open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
                print(f"QA prompt={row['prompt_id']} seed={row['seed']} step={snapshot['step']} question={question['question_id']}",flush=True)


if __name__=='__main__':main()
