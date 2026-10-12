"""Verify donor-preserving partitions and complete GEN result aggregation."""
import json
from pathlib import Path
import pytest
from qwen_latent_cot.diagnostics.gen_parallel import prepare, aggregate, ARMS, digest


def setup_plan(tmp_path):
    dataset=tmp_path/'data'; dataset.mkdir()
    rows=[]
    for pair in range(8):
        for member,color in enumerate(('red','blue')):
            image=dataset/f'{pair}_{member}.png'; image.write_bytes(b'image fixture')
            rows.append(dict(sample_id=f'p{pair}_{member}',scene_id=f's{pair}',family_id=f'f{pair}',pair_id=f'pair{pair}',
                             swap_id=f'p{pair}_{1-member}',split='test',label_type='clean',verified=True,
                             image=image.name,prompt='A cube on a table.',caption=f'A {color} cube on a table.',
                             facts=[dict(category='attribute',aliases=[color],alternatives=[],unknown=False)]))
    data=dataset/'colors.jsonl'; data.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    config=tmp_path/'config.json'; config.write_text(json.dumps(dict(eval_t=[0.9,1.],noise_seeds=[4],seed=0)))
    c0=tmp_path/'c0.json'; c0.write_text('{"passed":true}')
    checkpoint=tmp_path/'best.pt'; checkpoint.write_bytes(b'checkpoint fixture')
    output=tmp_path/'output'
    plan=prepare(data,config,output,tmp_path/'model',checkpoint,c0)
    return output,plan


def make_worker_outputs(plan):
    for task in plan['tasks']:
        run=Path(task['output']); run.mkdir()
        manifest=dict(model_files={'native':'hash'},source_hashes={'source':'hash'},depth=28,slots=16,topology='C',
                      checkpoints={'text_checkpoint':plan['checkpoint_hash']},config=dict(plan['common_config'],noise_seeds=[task['seed']]),
                      dataset_hash=digest(task['data']),status='executed; quality unverified')
        (run/'manifest.json').write_text(json.dumps(manifest))
        rows=[]
        for sample in task['sample_ids']:
            for arm in sorted(ARMS):
                rows.append(dict(sample_id=sample,t=0.5,noise_seed=task['seed'],arm=arm,frozen_guard='passed',
                                 supplementary_caption='A red cube.',target_facts=[],
                                 conditional={'normalized_difference':0.1},cfg={'normalized_difference':0.2,'norm_ratio':1.},
                                 repeat_error={'cfg':{'normalized_difference':0.}}))
                (run/f'{sample}_000_{arm}_x0_hat.png').write_bytes(b'image fixture')
        (run/'gen.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        (run/'timing.jsonl').write_text('{}\n')


def test_complete_pairs_and_relative_images_survive_partition(tmp_path):
    output,plan=setup_plan(tmp_path)
    assert len(plan['tasks'])==16
    for task in plan['tasks']:
        shard=[json.loads(line) for line in Path(task['data']).read_text().splitlines()]
        assert len(shard)==2
        assert {row['swap_id'] for row in shard}=={row['sample_id'] for row in shard}
        assert all(Path(row['image']).is_absolute() and Path(row['image']).is_file() for row in shard)
    make_worker_outputs(plan)
    aggregate(output,plan)
    summary=json.loads((output/'summary.json').read_text())
    assert summary['records']==224 and summary['states']==32
    assert len(list((output/'local_blind').glob('*.png')))==224
    assert not (output/'blind_key.json').exists()  # Local predictions cannot be reported as final samples.


def test_missing_arm_cannot_be_reported_as_complete(tmp_path):
    output,plan=setup_plan(tmp_path)
    make_worker_outputs(plan)
    path=Path(plan['tasks'][0]['output'])/'gen.jsonl'
    lines=path.read_text().splitlines(); path.write_text('\n'.join(lines[:-1])+'\n')
    with pytest.raises(ValueError,match='every requested'):
        aggregate(output,plan)
    assert not (output/'summary.json').exists()


def test_three_times_partition_across_sixteen_workers(tmp_path):
    output, original = setup_plan(tmp_path)
    data = Path(original['data'])
    rows = [json.loads(line) for line in data.read_text().splitlines()][:4]
    data.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    plan = prepare(data, tmp_path/'config.json', tmp_path/'recheck', tmp_path/'model',
                   tmp_path/'best.pt', tmp_path/'c0.json', [0.7, 0.8, 0.9], [0, 1, 2, 3], 2)
    assert len(plan['tasks']) == 16
    assert plan['samples'] == 4 and plan['states'] == 48 and plan['expected_records'] == 336
    actual = [(sample, t, task['seed']) for task in plan['tasks']
              for sample in task['sample_ids'] for t in task['eval_t']]
    assert len(actual) == len(set(actual)) == 48
    for task in plan['tasks']:
        config = json.loads(Path(task['config']).read_text())
        assert config['eval_t'] == task['eval_t']+[1.0]
