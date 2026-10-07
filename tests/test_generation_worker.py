"""Exercise the new Base/R2 worker on an actual tiny CPU decoder."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import hashlib
import torch
from PIL import Image
from helpers import prepared
from qwen_latent_cot.bagel import backbone,inferencer
from qwen_latent_cot.evaluation.io import load_manifests


def test_default_r2_generation_pairing_resume_and_removed_arm_rejected(tmp_path,monkeypatch):
    model,kwargs,old_runtime=prepared(batch=1,start=0,rounds=2,special_tokens=True)
    cache=kwargs['past_key_values'];seed=old_runtime.layerwise.seeds[cache];old_runtime.close()
    generated=[]
    class Generator:
        def __init__(self,bundle,runtime):self.runtime=runtime;self.prompt_lengths=(3,)
        def generate(self,prompts,shapes,seeds,**sampling):
            self.runtime.layerwise.seeds[cache]=seed
            noise=kwargs['packed_query_sequence'].clone()
            output=model.language_model.model.forward_inference(**{**kwargs,'packed_query_sequence':noise}).packed_query_sequence
            assert torch.isfinite(output).all()
            generated.append((self.runtime.config.mode,self.runtime.config.extra_rounds))
            self.runtime.clear_prompt_state()
            return [Image.new('RGB',shapes[0][::-1])],[hashlib.sha256(noise.float().numpy().tobytes()).hexdigest()]
    monkeypatch.setattr(backbone,'load_native',lambda *args:SimpleNamespace(model=model))
    monkeypatch.setattr(inferencer,'T2IGenerator',Generator)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'tiny_cpu_fixture')
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    script=Path(__file__).resolve().parents[1]/'scripts/evaluate/training_free.py'
    spec=importlib.util.spec_from_file_location('state_worker',script);worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    prompts=tmp_path/'prompts.jsonl';prompts.write_text(''.join(json.dumps(dict(prompt_id=str(i),prompt=f'{i} red cubes.'))+'\n' for i in range(2)))
    args=['generate','--model-path',str(tmp_path),'--prompts',str(prompts),'--output-dir',str(tmp_path/'worker_0'),
          '--device','cpu','--end-layer','3','--image-size','256']
    import sys
    monkeypatch.setattr(sys,'argv',args);worker.main()
    records,run=load_manifests([tmp_path/'worker_0/manifest.jsonl'])
    assert run['arms']==['BASE','LAYERWISE_UND_STATE_REPLACE_R2']
    assert len(records)==4 and generated==[('BASE',0)]*2+[('LAYERWISE_UND_STATE_REPLACE',2)]*2
    assert run['loop']['extra_rounds']==2
    for i in range(2):
        pair=[r for r in records if r['index']==i]
        assert pair[0]['noise_sha256']==pair[1]['noise_sha256']
        assert pair[1]['body_passes']==3 and pair[1]['writer_body_passes']==2 and pair[1]['writer_suffix_passes']==1
    generated.clear();worker.main();assert not generated
    monkeypatch.setattr(sys,'argv',args+['--arms','BASE,REMOVED_MODE'])
    import pytest
    with pytest.raises(ValueError,match='persistent UND'):worker.main()
