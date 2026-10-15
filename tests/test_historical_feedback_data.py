"""Historical C1 pairs preserve alignment without becoming checked trajectories."""
import json
from pathlib import Path
import pytest
from PIL import Image
from scripts.restore_visual_prompts import restore, digest
from scripts.prepare_historical_feedback import chain_rows, prepare, pilot
from qwen_latent_cot.diagnostics.data import load_dataset, visual_label


def test_restore_changes_only_prompt_and_keeps_backup_and_hash_metadata(tmp_path):
    path = tmp_path/'pairs.jsonl'
    row = dict(label_kind='target_clean', target_caption='Two cups.', prompt='An image.', split='train', image='same.png')
    path.write_text(json.dumps(row)+'\n')
    original = path.read_bytes(); original_hash = digest(path)
    path.with_suffix('.validation.json').write_text(json.dumps(dict(passed=True, dataset_sha256=original_hash)))
    result = restore(path)
    assert result['changed'] == 1 and Path(result['backup']).read_bytes() == original
    updated = json.loads(path.read_text()); assert updated == dict(row, prompt=row['target_caption'])
    metadata = json.loads(path.with_suffix('.validation.json').read_text())
    assert metadata['dataset_sha256'] == digest(path) and metadata['prompt_restore']['modified_fields'] == ['prompt']
    assert restore(path)['changed'] == 0


def test_restore_missing_goal_leaves_original_untouched(tmp_path):
    path = tmp_path/'bad.jsonl'; path.write_text(json.dumps(dict(label_kind='target_clean',prompt='An image.')))
    original = path.read_bytes()
    with pytest.raises(ValueError, match='target_caption'): restore(path)
    assert path.read_bytes() == original and not path.with_suffix('.neutral_prompt.jsonl').exists()


def chain(sid, images):
    return dict(sample_id=sid, prompt='Two blue cups.', num_turns=1, chain_status='terminal',
                adapter_role='stage11_stage12_stage2_gold', image_paths=[str(p) for p in images],
                steps=[dict(analysis='One red cup.',fix='Add a second cup and make both cups blue.', output_image=images[1].name),
                       dict(analysis='The image matches the prompt.')])


def test_fix_uses_before_image_and_keep_uses_terminal(tmp_path):
    images = [tmp_path/'img0.png',tmp_path/'img1.png']
    rows, error = chain_rows(chain('cups',images),'train',tmp_path,str.split)
    assert error is None and len(rows)==2
    assert rows[0]['image']==str(images[0]) and rows[1]['image']==str(images[1])
    assert rows[0]['feedback']['origin']=='source_annotation' and rows[0]['human_reviewed'] is False
    assert rows[1]['feedback']['satisfied'] and rows[1]['caption']=='No change is needed.'
    value = chain('cups',images); value['steps'][0]['output_image']='img2.png'
    assert chain_rows(value,'train',tmp_path,str.split)[1]=='output_alignment'


def test_source_feedback_only_accepted_by_c1_not_visual_teacher(tmp_path):
    rows,_ = chain_rows(chain('cups',[tmp_path/'img0.png',tmp_path/'img1.png']),'train',tmp_path,str.split)
    row=rows[0]
    with pytest.raises(ValueError,match='checked'):visual_label(row)
    normalized=visual_label(row,c1_feedback=True)
    assert normalized['confirmed_label'] is False and normalized['verified'] is True
    with pytest.raises(ValueError,match='checked'):visual_label(dict(row,state='state.pt'),c1_feedback=True)


def test_convert_decode_split_isolation_and_load_c1(tmp_path):
    manifests=tmp_path/'qwen_latent_cot_v3_adapted_20260816/manifests';manifests.mkdir(parents=True)
    for i,split in enumerate(('train','val','test')):
        images=[tmp_path/f'{split}_img{n}.png' for n in range(2)]
        for n,p in enumerate(images):Image.new('RGB',(8,8),(i*60,n*50,20)).save(p)
        (manifests/f'cort36k_{split}.jsonl').write_text(json.dumps(chain(split,images))+'\n')
    output=tmp_path/'feedback.jsonl'
    summary=prepare(tmp_path,output,str.split,keep_ratio=1.)
    assert summary['rows']==6 and summary['feedback_origin']=='source_annotation'
    rows=load_dataset(output,visual=True,c1_feedback=True)
    assert len(rows)==6 and all(not r['confirmed_label'] for r in rows)
    with pytest.raises(ValueError,match='checked'):load_dataset(output,visual=True)
    small=tmp_path/'pilot.jsonl';pilot(output,small)
    assert len(load_dataset(small,visual=True,c1_feedback=True))==6
