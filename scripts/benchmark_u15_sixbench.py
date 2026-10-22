"""U1.5 adapter: original six manifests/rubrics, isolated model and scoring arm."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from benchmark_sixbench import COUNTS,SOURCE,PACKAGE,rows,digest,atomic,report_path
import score_sixbench as scoring


def materialize(run,config):
    from PIL import Image
    helper=scoring.load(PACKAGE/'code/prepare_sixbench_scoring.py','_original_views')
    plan=json.loads((run/'PLAN.json').read_text())
    for job in config['jobs']:
        bench=job['benchmark'];generated,source=helper.manifest_pair(run,bench);bindings=[]
        view=Path(job.get('pipeline_root',job.get('image_root')))
        for item in generated:
            image=run/'images'/item['output_rel'];report=json.loads(report_path(image).read_text())
            if report['input']!=item or report['identity']!=plan['identity'] or report['status']!='ok':raise ValueError('U1.5 generation binding differs')
            with Image.open(image) as png:
                if png.size!=(plan['config']['width'],plan['config']['height']):raise ValueError('Geometry differs')
                png.verify()
            if digest(image)!=report['image_sha256']:raise ValueError('PNG integrity differs')
            bindings.append(dict(record_id=item['record_id'],index=item.get('index'),image=str(image),image_sha256=report['image_sha256'],scorer_output_rel=source[item['record_id']]['output_rel']))
        for b in bindings:
            if bench.endswith('_pp'):
                helper.locked_json(view/'cases'/f"case-{b['index']:03d}"/'report.json',dict(status='ok',final_image=b['image'],final_image_sha256=b['image_sha256'],provenance=b))
            else:helper.symlink_locked(view/b['scorer_output_rel'],b['image'])
        helper.locked_json(view/'VIEW_COMPLETE.json',dict(complete=True,status='complete',expected=COUNTS[bench],present=len(bindings),bindings=bindings))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','generate','materialize','score','status'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--model');p.add_argument('--source')
    p.add_argument('--config',default='configs/u15/correction.json');p.add_argument('--checkpoint');p.add_argument('--gen-weights')
    p.add_argument('--arm',choices=['native','global','corrected'],default='native');p.add_argument('--label',default='u15_native')
    p.add_argument('--device',default='npu:0');p.add_argument('--api-base',default='http://127.0.0.1:18364/v1')
    a=p.parse_args();run=a.run.resolve()
    if a.action=='prepare':
        if not a.model or not a.source:p.error('--model and --source required')
        run.mkdir(parents=True,exist_ok=False)
        original=scoring.load(PACKAGE/'code/prepare_bundle.py','_original_normalize')
        combined=[]
        for b,n in COUNTS.items():
            values=rows(SOURCE/'manifests'/f'{b}.jsonl')
            if len(values)!=n:raise ValueError('Source benchmark incomplete')
            converted=[original.normalize(x,b,len(combined)+i) for i,x in enumerate(values)];combined.extend(converted)
            (run/'manifests').mkdir(exist_ok=True);(run/'score_manifests').mkdir(exist_ok=True)
            (run/'manifests'/f'{b}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in converted))
            shutil.copyfile(SOURCE/'manifests'/f'{b}.jsonl',run/'score_manifests'/f'{b}.jsonl')
            if b.endswith('_pp'):shutil.copyfile(SOURCE/'bindings'/f'{b}_scorer_manifest.json',run/'score_manifests'/f'{b}.json')
        (run/'manifests/combined.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in combined))
        config=json.loads(Path(a.config).read_text())
        identity=dict(model=str(Path(a.model).resolve()),source=str(Path(a.source).resolve()),
            config=config,checkpoint=str(Path(a.checkpoint).resolve()) if a.checkpoint else None,
            checkpoint_sha256=digest(a.checkpoint) if a.checkpoint else None,
            gen_weights=str(Path(a.gen_weights).resolve()) if a.gen_weights else None,
            gen_sha256=digest(a.gen_weights) if a.gen_weights else None,arm=a.arm)
        if a.arm!='native' and not a.checkpoint:raise ValueError('Correction checkpoint required')
        atomic(run/'PLAN.json',dict(schema='u15_sixbench_v1',arm=a.label,display_name=a.label,identity=identity,config=config,counts=COUNTS))
        scoring.ARM=a.label;scoring.NAMESPACE=run.name+'_27b_v1';scoring.prepare(run,a.api_base,protocol='u15_original_sixbench_rubric_local27b_v1',artifact_prefix='U15_27B');return
    plan=json.loads((run/'PLAN.json').read_text());scoring.ARM=plan['arm'];scoring.materialize=materialize
    if a.action in ('materialize','score','status'):
        sys.argv=[scoring.__file__,a.action,'--run',str(run)];scoring.main();return
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from qwen_latent_cot.u15.runtime import Runtime
    from qwen_latent_cot.u15.readout import Corrector
    from qwen_latent_cot.u15.checkpoint import load_portable
    from qwen_latent_cot.u15.sampling import sample
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    device=f'npu:{os.environ.get("LOCAL_RANK","0")}' if a.device.startswith('npu') else a.device
    identity=plan['identity'];rt=Runtime.load(identity['model'],identity['source'],plan['config'],device,identity['gen_weights'])
    corrector=None
    if identity['arm']!='native':
        corrector=Corrector(rt,plan['config']);corrector.initialize_global(plan['config']['active_steps'])
        load_portable(identity['checkpoint'],corrector,rt,'correct')
    for row in rows(run/'manifests/combined.jsonl'):
        if row['global_index']%world!=rank:continue
        image=run/'images'/row['output_rel'];receipt=report_path(image)
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['identity']!=identity or old['input']!=row or digest(image)!=old['image_sha256']:raise ValueError('Resume identity/integrity differs')
            continue
        result,trace=sample(rt,row['instruction'],row['seed'],corrector,identity['arm'])
        image.parent.mkdir(parents=True,exist_ok=True);temporary=image.with_suffix('.partial.png');result.save(temporary);temporary.replace(image)
        atomic(receipt,dict(status='ok',input=row,identity=identity,image_sha256=digest(image),trace=trace))
        print(json.dumps(dict(rank=rank,global_index=row['global_index'],benchmark=row['benchmark'])),flush=True)


if __name__=='__main__':main()
