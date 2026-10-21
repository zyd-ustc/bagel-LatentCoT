"""Prompt visibility, semantic suite and complete offline comparison output."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest
from PIL import Image


def module(name):
    spec=importlib.util.spec_from_file_location(name,Path('scripts')/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


sampling=module('sample_stage_a_compare')


def test_arm_visibility_and_explicit_gap_suite():
    prepare=module('prepare_stage_a_sample_prompts')
    rows=prepare.build_authored()
    assert len(rows)==16 and len({r['id'] for r in rows})==16
    for row in rows:
        assert len(row['full_prompt'].split())>=30
        assert row['removed_clauses'] in row['full_prompt']
        assert row['removed_clauses'] not in row['partial_prompt']
        assert row['full_prompt'].split(row['removed_clauses'])[0].strip() in row['partial_prompt']
        assert row['checks']
    row,donor=rows[:2]
    assert sampling.arm_inputs(row,'native_full',donor)==dict(prompt=row['full_prompt'],native=True,reader_prompt=None)
    assert sampling.arm_inputs(row,'native_partial',donor)==dict(prompt=row['partial_prompt'],native=True,reader_prompt=None)
    assert sampling.arm_inputs(row,'loop_partial',donor)==dict(prompt=row['partial_prompt'],native=False,reader_prompt=row['full_prompt'])
    assert sampling.arm_inputs(row,'wrong_reader',donor)==dict(prompt=row['partial_prompt'],native=False,reader_prompt=donor['full_prompt'])
    assert len(sampling.arms_for(row))==5
    assert len(sampling.arms_for(dict(row,removed_clauses='')))==2


def test_gallery_requires_complete_consistent_images(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    renderer=module('render_stage_a_compare')
    rows=module('prepare_stage_a_sample_prompts').build_authored()[:2]
    (tmp_path/'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for i,row in enumerate(rows):
        arms={}
        for arm in sampling.arms_for(row):
            filename=f'case_{i:03d}_{arm}.png';Image.new('RGB',(32,32),'red').save(tmp_path/filename)
            arms[arm]=dict(image=filename,seconds=1.,peak_memory_gib=2.)
        record=dict(row,seed=17,checkpoint='/ckpt/step_003000.pt',checkpoint_step=3000,
                    donor_id='other',donor_prompt='<unsafe>',arms=arms)
        (tmp_path/f'case_{i:03d}.json').write_text(json.dumps(record))
    (tmp_path/'worker_00.json').write_text(json.dumps(dict(status='complete',sampling={'resolution':512},blocks=28,active_updates=25)))
    summary=renderer.render(tmp_path)
    assert summary['cases']==2 and summary['images']==10 and summary['checkpoint_step']==3000
    page=(tmp_path/'index.html').read_text()
    assert '&lt;unsafe&gt;' in page and '<unsafe>' not in page
    assert 'id="filter"' in page and 'id="gap"' in page and '__META__' not in page
    (tmp_path/'case_001_wrong_reader.png').unlink()
    with pytest.raises(FileNotFoundError):renderer.render(tmp_path)
