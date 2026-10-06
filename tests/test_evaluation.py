import json
from pathlib import Path
import pytest
from qwen_latent_cot.evaluation.scoring import normalize_atoms
from qwen_latent_cot.evaluation.report import paired_report, summarize
from qwen_latent_cot.evaluation.io import load_manifests,sha256


def row(prompt='p0',seed=0,arm='BASE',atoms=(0.,1.),**extra):
    return dict(prompt_id=prompt,seed=seed,arm=arm,prompt=prompt,noise_sha256='noise',height=256,width=256,
        semantic_atoms=list(atoms),quality_proxy=.75,invalid=False,generation_seconds=1.,
        peak_allocated_bytes=10,bucket='structural',**extra)


def test_repair_damage_uses_all_atoms_denominator():
    a=row();b=row(arm='MEMORY_LOOP',atoms=(1.,0.))
    result=paired_report({('p0',0):a},{('p0',0):b},resamples=20)
    assert result['repair_count']==result['damage_count']==1
    assert result['net_repair']['mean']==0
    assert result['repair_given_base_fail']==result['damage_given_base_pass']==1


def test_bootstrap_keeps_all_seeds_of_each_prompt_together():
    # Opposite changes for two seeds in one prompt cancel in every resample.
    reference={('p0',s):row(seed=s,atoms=(float(s),)) for s in (0,1)}
    candidate={('p0',s):row(seed=s,arm='MEMORY_LOOP',atoms=(float(1-s),)) for s in (0,1)}
    result=paired_report(reference,candidate,resamples=200)
    assert result['net_repair']=={'mean':0.,'ci95':[0.,0.]}
    assert result['prompt_clusters']==1 and result['paired_images']==2


def test_pairing_rejects_changed_noise_or_coverage():
    a=row();b={**row(arm='MEMORY_LOOP'),'noise_sha256':'other'}
    with pytest.raises(ValueError):paired_report({('p0',0):a},{('p0',0):b})
    with pytest.raises(ValueError):paired_report({('p0',0):a},{})


def test_atom_tolerance_is_not_a_general_clamp():
    assert normalize_atoms([1+2e-7,-2e-7],2)==[1.,0.]
    for atoms in ([1.01],[float('nan')],[True],[]):
        with pytest.raises(ValueError):normalize_atoms(atoms,1)


def test_scorer_failure_cannot_be_reported_as_zero():
    a=row();b={**row(arm='MEMORY_LOOP'),'semantic_atoms':None}
    with pytest.raises(ValueError):summarize([a,b],resamples=20)


def test_layerwise_report_compares_base_and_explicit_legacy_control():
    result=summarize([row(),row(arm='MEMORY_LOOP',atoms=(1.,0.)),
                      row(arm='LAYERWISE_MEMORY_KV',atoms=(1.,1.))],resamples=20)
    assert result['LAYERWISE_MEMORY_KV']['vs_MEMORY_LOOP']['repair_count']==1
    assert result['LAYERWISE_MEMORY_KV']['vs_BASE']['damage_count']==0


def test_manifest_requires_complete_shards(tmp_path):
    run={'source_sha256':'s','model_sha256':{},'benchmark_sha256':'b','sampling':{},'loop':{},
         'seeds':[0,1],'arms':['BASE'],'prompt_ids':['p0']}
    (tmp_path/'run.json').write_text(json.dumps(run))
    (tmp_path/'manifest.jsonl').write_text(json.dumps({**row(),'valid_file':False})+'\n')
    with pytest.raises(ValueError,match='incomplete'):load_manifests([tmp_path/'manifest.jsonl'])


def test_cluster_ci_does_not_depend_on_shard_record_order():
    reference={(p,0):row(prompt=p,atoms=(0.,)) for p in ('c','b','a')}
    candidate={(p,0):row(prompt=p,arm='MEMORY_LOOP',atoms=(float(p=='a'),)) for p in ('c','b','a')}
    result=paired_report(reference,candidate,resamples=200)
    reverse_reference=dict(reversed(list(reference.items())))
    reverse_candidate=dict(reversed(list(candidate.items())))
    assert result==paired_report(reverse_reference,reverse_candidate,resamples=200)


def test_quality_shard_merge_checks_complete_generation_binding(tmp_path,monkeypatch):
    import importlib.util,sys
    from PIL import Image
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts/evaluate'))
    spec=importlib.util.spec_from_file_location('merge_quality_test',root/'scripts/evaluate/merge_quality.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    run={'source_sha256':'test','model_sha256':{},'benchmark_sha256':'b','sampling':{},'loop':{},
         'seeds':[0,1],'arms':['BASE','MEMORY_LOOP'],'prompt_ids':['p0'],'stage':'evaluation'}
    image=tmp_path/'image.png';Image.new('RGB',(28,28)).save(image)
    manifests=[];dirs=[]
    for seed in range(2):
        worker=tmp_path/f'worker_{seed}';worker.mkdir()
        values=[{**row(seed=seed,arm=arm,atoms=(float(arm!='BASE'),)),
                 'path':str(image),'image_sha256':sha256(image),'valid_file':True} for arm in run['arms']]
        (worker/'run.json').write_text(json.dumps(run))
        manifest=worker/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in values))
        (worker/'scorer.json').write_text(json.dumps({'run':run,'scorer':{'test':'synthetic'},'scoring_shard':[seed,2]}))
        (worker/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reversed(values)))
        manifests.append(str(manifest));dirs.append(str(worker))
    args=['merge_quality.py','--manifests',*manifests,'--score-dirs',*dirs,'--output-dir',str(tmp_path/'report'),'--bootstrap-replicates','20']
    monkeypatch.setattr(sys,'argv',args);module.main()
    summary=json.loads((tmp_path/'report/summary.json').read_text())
    assert summary['arms']['MEMORY_LOOP']['vs_BASE']['repair_count']==2
    assert not summary['training_admitted']
    bad=Path(dirs[0])/'scorer.json';binding=json.loads(bad.read_text());binding['run']['source_sha256']='other';bad.write_text(json.dumps(binding))
    with pytest.raises(ValueError,match='differs from generation'):module.main()
    binding['run']=run;bad.write_text(json.dumps(binding))
    (Path(dirs[0])/'scores.jsonl').write_text('')
    with pytest.raises(ValueError,match='incomplete scoring'):module.main()
