"""Full-model execution acceptance before a 64-rank main run."""
import argparse
import json
from pathlib import Path
import random
import torch
from qwen_latent_cot.bagel.backbone import load_native
from qwen_latent_cot.bagel.accelerator import set_device,synchronize,max_memory_allocated
from qwen_latent_cot.correction.runtime import Runtime
from qwen_latent_cot.correction.model import SpatialCorrector,proposal_from
from qwen_latent_cot.correction.data import Corpus
from qwen_latent_cot.correction.training import paired_state,train,write
from qwen_latent_cot.correction.sampling import sample,times,active_steps
from qwen_latent_cot.correction import checkpoint
from qwen_latent_cot.correction.objective import residual_target,guidance_safe


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--corpus',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='npu:0');p.add_argument('--config',default='configs/correction.json');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());cfg.update(steps=3,global_batch=1,val_images=0,save_steps=[3],warmup_steps=0,semantic_eval_steps=[])
    device=set_device(a.device);torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    runtime=Runtime(load_native(a.model,device,cfg['timestep_shift']),cfg);corrector=SpatialCorrector(runtime,cfg).eval()
    corpus=Corpus(a.corpus);row=corpus.validation(1)[0]
    x0,epsilon,x_t,flow,base,context=paired_state(runtime,row,.8,17)
    with torch.no_grad():
        delta=corrector(proposal_from(x_t,.8,base),.8,flow,context)
        assert delta.count_nonzero()==0,'Zero head changed baseline'
    provenance=dict(model_path=str(Path(a.model).resolve()),corpus_path=str(Path(a.corpus).resolve()),
                    corpus_fingerprint=corpus.manifest['fingerprint'],initialization='native_pretrained_UND')
    train(corrector,corpus,a.output,provenance)
    saved=checkpoint.read(Path(a.output)/'step_000003.pt')
    with torch.no_grad():
        expected=corrector(proposal_from(x_t,.8,base),.8,flow,context)
        corrector.head.weight.zero_();corrector.head.bias.zero_();checkpoint.apply(saved,corrector)
        restored=corrector(proposal_from(x_t,.8,base),.8,flow,context)
        torch.testing.assert_close(expected,restored,rtol=0,atol=0)
    del x0,epsilon,x_t,flow,base,context,delta,expected,restored
    corrector.config['steps']=4
    train(corrector,Corpus(a.corpus),a.output,provenance,saved=saved)
    del saved
    if hasattr(corrector,'calibration'):
        # The tiny pilot has fewer updates than time nodes. Complete the diagnostic
        # baseline using one TRAIN pair, never validation labels.
        pilot_corpus=Corpus(a.corpus);training_row=pilot_corpus.groups('train')[0][0]
        pending=torch.zeros((len(corrector.calibration.nodes),corrector.calibration.sums.shape[1]+1),device=device)
        for index,t in enumerate(corrector.calibration.nodes):
            if corrector.calibration.counts[index]>0:continue
            with torch.no_grad():
                x0,epsilon,x_t,flow,base,context,conditional=paired_state(runtime,training_row,t,17,with_conditional=True)
                corrector.calibration.observe(pending,t,residual_target(base,x0,epsilon,cfg,conditional))
        corrector.calibration.update(pending);pilot_corpus.close()
        del x0,epsilon,x_t,flow,base,context
    for label,kwargs in [('native',dict(native=True)),('global',dict(global_only=True)),('corrected',{})] if cfg.get('correction_space')=='post_cfg' else [('native',dict(native=True)),('corrected',{})]:
        image,trace=sample(corrector,row['caption'],17,**kwargs)
        image.save(Path(a.output)/(label+'.png'));write(Path(a.output)/(label+'.json'),trace)
        if guidance_safe(cfg):
            assert all(node.get('guidance_projection_applied',0.)>=-1e-5 for node in trace)
    records=[json.loads(x) for x in (Path(a.output)/'train.jsonl').read_text().splitlines()]
    assert records[0]['gradient_head']>0 and records[0]['gradient_und']==0
    assert records[1]['gradient_und']>0 and records[1]['delta_rms']>0
    if guidance_safe(cfg):assert all(r['applied_negative_fraction']==0 for r in records)
    synchronize(device)
    result=dict(status='passed',checks=['zero_init','head_then_und_gradients','frozen_native','checkpoint_restore','optimizer_sampler_rng_resume','paired_sampling'],
                steps=4,peak_memory_gib=max_memory_allocated(device)/1024**3,caption=row['caption'])
    write(Path(a.output)/'acceptance.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
