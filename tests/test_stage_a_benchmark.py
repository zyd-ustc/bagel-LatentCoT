import importlib.util
import json
from pathlib import Path
import pytest
from PIL import Image

spec=importlib.util.spec_from_file_location('benchmark_stage_a',Path('scripts/benchmark_stage_a.py'))
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


def test_official_export_and_subset_score_contract(tmp_path):
    cases=[dict(id='geneval_00012',official_item_index=12,full_prompt='a red dog',benchmark_metadata=dict(prompt='a red dog',tag='colors',include=[dict(**{'class':'dog'},count=1,colors=['red'])]))]
    plan=dict(checkpoint='/checkpoints/step_004000.pt',seeds=[0],expected_images=2,subset=True,formal_default_protocol=False)
    (tmp_path/'plan.json').write_text(json.dumps(plan));(tmp_path/'prompts.jsonl').write_text(json.dumps(cases[0])+'\n')
    directory=tmp_path/'samples/seed_0000';directory.mkdir(parents=True)
    arms={}
    for arm in bench.ARMS:
        filename=arm+'.png';Image.new('RGB',(8,8),'red').save(directory/filename)
        arms[arm]=dict(image=filename,prompt='a red dog',native=arm=='native_full')
    record=dict(id=cases[0]['id'],seed=0,checkpoint=plan['checkpoint'],arms=arms)
    record_path=directory/'case_000.json';record_path.write_text(json.dumps(record))
    bench.export(tmp_path)
    for arm in bench.ARMS:
        path=tmp_path/'official'/arm/'00012'
        assert json.loads((path/'metadata.jsonl').read_text())==cases[0]['benchmark_metadata']
        assert (path/'samples/0000.png').is_file()
    (tmp_path/'scores').mkdir()
    for arm in bench.ARMS:
        (tmp_path/'scores'/f'{arm}.jsonl').write_text(json.dumps(dict(official_item_index=12,seed=0,tag='colors',correct=True))+'\n')
    summary=bench.summarize(tmp_path)
    for arm in summary['arms'].values():
        assert arm['coverage_complete'] and arm['scored_subset_macro_accuracy']==1.
        assert 'geneval_macro_accuracy' not in arm and 'full_553_prompt_macro_accuracy' not in arm
    record['checkpoint']='/wrong.pt';record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='identity'):bench.export(tmp_path)
