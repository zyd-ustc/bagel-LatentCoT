"""Real512 native NPU pilot: caption, two-round edit, saved adapters and images."""
import argparse
import json
from pathlib import Path
import random
import torch
from qwen_latent_cot.bagel.accelerator import set_device
from qwen_latent_cot.bagel.backbone import load_native
from qwen_latent_cot.correction.runtime import Runtime
from qwen_latent_cot.mot_edit.model import MoTEdit
from qwen_latent_cot.mot_edit.data import Corpus
from qwen_latent_cot.mot_edit.training import train
from qwen_latent_cot.mot_edit import checkpoint
from qwen_latent_cot.mot_edit.sampling import sample

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--corpus',required=True);p.add_argument('--output',required=True)
p.add_argument('--device',default='npu:15');a=p.parse_args()
cfg=json.loads(Path('configs/mot_edit_8000.json').read_text())
cfg.update(steps=3,observation_steps=1,global_batch=1,save_steps=[1,2,3],semantic_eval_steps=[],observation_eval_steps=[],warmup_steps=1)
torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);device=set_device(a.device)
rt=Runtime(load_native(a.model,device,cfg['timestep_shift']),cfg);model=MoTEdit(rt,cfg).eval()
provenance={'pilot':'full512_caption_then_edit'}
train(model,Corpus(a.corpus),a.output,provenance)
saved=checkpoint.read(Path(a.output)/'step_000003.pt');checkpoint.apply(saved,model)
row=Corpus(a.corpus).row(Corpus(a.corpus).ids('val')[0])
x0=rt.encode(row['image'],17)
print(json.dumps(dict(event='pilot_caption',caption=model.caption_text(x0,max_tokens=8))),flush=True)
for arm,rounds in [('native',0),('two_round',2)]:
 image,trace=sample(model,'A photograph of two red cubes on the left and one blue ball on the right, on a wooden table.',17,rounds)
 image.save(Path(a.output)/(arm+'.png'));(Path(a.output)/(arm+'.json')).write_text(json.dumps(trace,indent=2))
print(json.dumps({'event':'pilot_complete','step':saved['step'],'stage':saved['stage']}),flush=True)
