"""Two-rank real HCCL/FSDP test of native RGB SFT; synthetic tiny weights only."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent))
import test_pipeline as fixture
from qwen_latent_cot.correction import distributed
from qwen_latent_cot.u15.training import train
from qwen_latent_cot.u15.checkpoint import export


def main():
    device=distributed.initialize('npu:0');fixture.DEVICE=str(device)
    rt,cfg=fixture.tiny();rt.model.cpu();rt.device=device
    root=Path(sys.argv[1]);corpus=root/'corpus'
    if distributed.rank()==0:
        corpus.mkdir(parents=True,exist_ok=False)
        Image.new('RGB',(64,64),'red').save(corpus/'image.png')
        db=sqlite3.connect(corpus/'pairs.sqlite')
        db.execute('CREATE TABLE pairs(id INTEGER PRIMARY KEY,uid TEXT,image TEXT,prompt TEXT,width INTEGER,height INTEGER,split TEXT)')
        for i in range(3):db.execute('INSERT INTO pairs VALUES(?,?,?,?,?,?,?)',(i,str(i),str(corpus/'image.png'),'red ball',64,64,'val' if i==2 else 'train'))
        db.commit();db.close();(corpus/'manifest.json').write_text(json.dumps(dict(format='bagel_t2i_pairs_v1',identity='distributed-test')))
    distributed.barrier()
    train(rt,cfg,'t2i',corpus,root/'sft')
    distributed.barrier()
    if distributed.rank()==0:
        result=export(root/'sft/step_000002',root/'gen.pt')
        assert result['stage']=='t2i' and result['weights']
        print('HCCL_FSDP_GEN_EXPORT_PASSED',flush=True)
    distributed.barrier();torch.distributed.destroy_process_group()


if __name__=='__main__':main()
