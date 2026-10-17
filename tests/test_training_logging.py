"""Real offline W&B runs, checkpoint continuation and failed-run accounting."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from test_training_corpus import lm, runtime, visual_runtime, tiny_config, tiny_corpus
from qwen_latent_cot.diagnostics.visual import VisualStudent
from qwen_latent_cot.diagnostics.visual_training import train


def test_tracking_accepts_long_api_keys(monkeypatch):
    wandb=pytest.importorskip('wandb')
    from wandb.sdk.lib import apikey
    saved=[]
    monkeypatch.setattr(apikey,'write_netrc',lambda host,entity,key:saved.append(key))
    key='a'*86
    apikey.write_key(wandb.Settings(),key,api=object())
    assert saved==[key]


def test_offline_tracking_uses_fresh_runs_and_restored_training_steps(visual_runtime,tmp_path,monkeypatch):
    pytest.importorskip('wandb')
    runtime,_=visual_runtime;cfg=tiny_config(runtime,'C2-observe');corpus=tiny_corpus(tmp_path)
    monkeypatch.setenv('WANDB_RUN_ID','old-run-id');monkeypatch.setenv('WANDB_RESUME','must')
    options=dict(mode='offline',project='bagel-training-tests')
    student=VisualStudent(runtime)
    first=tmp_path/'first';first.mkdir()
    train(student,corpus,'C2-observe',first,wandb_options=options)
    first_run=json.loads((first/'wandb.json').read_text())
    assert first_run['id']!='old-run-id' and first_run['mode']=='offline'
    assert len(list(first.glob('wandb/offline-run-*/*.wandb')))==1
    validation=json.loads((first/'validation.jsonl').read_text())
    assert validation['metrics']['trainer/step']==1
    assert 'val/observe/source_off_nll' in validation['metrics']
    assert 'step=1/1' in (first/'logs/rank_00.log').read_text()
    cfg['steps']=2;second=tmp_path/'second';second.mkdir()
    train(student,corpus,'C2-observe',second,resume=str(first/'last.pt'),wandb_options=options)
    second_run=json.loads((second/'wandb.json').read_text())
    assert second_run['id']!=first_run['id']
    record=json.loads((second/'training.jsonl').read_text())
    summary=json.loads((second/'training_summary.json').read_text())
    assert record['step']==2 and record['step_seconds']>0
    assert summary['status']=='completed' and summary['completed_steps']==2 and summary['resumed_from_step']==1
    corpus.close()


def test_failed_training_records_actual_completed_steps(visual_runtime,tmp_path):
    runtime,_=visual_runtime;tiny_config(runtime,'C2-observe');corpus=tiny_corpus(tmp_path)
    for name in ('a.png','b.png'):(tmp_path/name).unlink()
    output=tmp_path/'failed';output.mkdir()
    with pytest.raises(FileNotFoundError):
        train(VisualStudent(runtime),corpus,'C2-observe',output,skip_validation=True)
    summary=json.loads((output/'training_summary.json').read_text())
    assert summary['status']=='failed' and summary['completed_steps']==0 and summary['target_steps']==1
    assert summary['failure']['type']=='FileNotFoundError'
    assert 'FileNotFoundError' in (output/'logs/rank_00.log').read_text()
    corpus.close()


@pytest.mark.parametrize('parent,resume,expected',[(None,None,'--writer-init'),('observe.pt',None,'--checkpoint'),
                                                ('observe.pt','last.pt','--resume')])
def test_launcher_does_not_mix_stale_initialization_weights(tmp_path,parent,resume,expected):
    executable=tmp_path/'capture-python';captured=tmp_path/'arguments.json'
    executable.write_text(f'#!{sys.executable}\nimport os,sys,json\n'
                          'with open(os.environ["CAPTURE"],"w") as f: json.dump(sys.argv[1:],f)\n')
    executable.chmod(0o755)
    env=dict(os.environ,MODEL_PYTHON=str(executable),CAPTURE=str(captured),NPROC_PER_NODE='1',
             WRITER_INIT='old-c1.pt',VISUAL_PARENT=parent or '',VISUAL_RESUME=resume or '',
             WANDB_MODE='offline',WANDB_PROJECT='bagel-LatentCoT')
    script=Path(__file__).resolve().parents[1]/'scripts/visual_training_npu.sh'
    subprocess.run(['bash',str(script),'feedback',str(tmp_path/'run')],env=env,check=True)
    args=json.loads(captured.read_text())
    assert [key for key in ('--writer-init','--checkpoint','--resume') if key in args]==[expected]
    assert args[args.index('--stage')+1]=='C2-feedback'
    assert args[args.index('--wandb-project')+1]=='bagel-LatentCoT'
