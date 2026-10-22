import os,sys,json
from pathlib import Path
import torch
import torch.distributed as dist
from test_t2i_sft import bundle,cfg,fixture
from qwen_latent_cot.t2i.training import train
from qwen_latent_cot.t2i import checkpoint as ck
from qwen_latent_cot.t2i.sampling import load_overlay
root=Path(sys.argv[1]);torch.set_num_threads(1)
dist.init_process_group('gloo');r=dist.get_rank();c=cfg();c['val_images']=1
if r==0:fixture(root,bundle().tokenizer,c)
dist.barrier()
train(bundle(),root/'corpus',root/'sharded',c,torch.device('cpu'))
train(bundle(),root/'corpus',root/'sharded',dict(c,steps=3),torch.device('cpu'),root/'sharded/step_000002')
train(bundle(),root/'corpus',root/'full',dict(c,steps=3),torch.device('cpu'))
a=torch.load(root/f'sharded/step_000003/rank_{r:05d}.pt',weights_only=False)
b=torch.load(root/f'full/step_000003/rank_{r:05d}.pt',weights_only=False)
for n in a['weights']:torch.testing.assert_close(a['weights'][n],b['weights'][n],rtol=0,atol=0)
for n in a['ema']:torch.testing.assert_close(a['ema'][n],b['ema'][n],rtol=0,atol=0)
dist.barrier()
if r==0:
 out=ck.export(root/'sharded/step_000003',root/'export');load_overlay(bundle(),out)
 print('FSDP_TWO_RANK_RESUME_EMA_EXPORT_PASSED',flush=True)
dist.destroy_process_group()
