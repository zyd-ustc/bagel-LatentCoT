import json
from pathlib import Path
import pytest
from qwen_latent_cot.evaluation.scoring import normalize_atoms
from qwen_latent_cot.evaluation.report import paired_report, summarize
from qwen_latent_cot.evaluation.io import load_manifests,sha256
from qwen_latent_cot.evaluation.protocol import validate_confirmation


def row(prompt='p0',seed=0,arm='BASE',atoms=(0.,1.),**extra):
    return dict(prompt_id=prompt,seed=seed,arm=arm,prompt=prompt,noise_sha256='noise',height=256,width=256,
        semantic_atoms=list(atoms),quality_proxy=.75,invalid=False,generation_seconds=1.,
        peak_allocated_bytes=10,bucket='structural',**extra)


def test_repair_damage_uses_all_atoms_denominator():
    a=row();b=row(arm='MEMORY_DYNAMIC',atoms=(1.,0.))
    result=paired_report({('p0',0):a},{('p0',0):b},resamples=20)
    assert result['repair_count']==result['damage_count']==1
    assert result['net_repair']['mean']==0
    assert result['repair_given_base_fail']==result['damage_given_base_pass']==1


def test_bootstrap_keeps_all_seeds_of_each_prompt_together():
    # Opposite changes for two seeds in one prompt cancel in every resample.
    reference={('p0',s):row(seed=s,atoms=(float(s),)) for s in (0,1)}
    candidate={('p0',s):row(seed=s,arm='MEMORY_DYNAMIC',atoms=(float(1-s),)) for s in (0,1)}
    result=paired_report(reference,candidate,resamples=200)
    assert result['net_repair']=={'mean':0.,'ci95':[0.,0.]}
    assert result['prompt_clusters']==1 and result['paired_images']==2


def test_pairing_rejects_changed_noise_or_coverage():
    a=row();b={**row(arm='MEMORY_DYNAMIC'),'noise_sha256':'other'}
    with pytest.raises(ValueError):paired_report({('p0',0):a},{('p0',0):b})
    with pytest.raises(ValueError):paired_report({('p0',0):a},{})


def test_atom_tolerance_is_not_a_general_clamp():
    assert normalize_atoms([1+2e-7,-2e-7],2)==[1.,0.]
    for atoms in ([1.01],[float('nan')],[True],[]):
        with pytest.raises(ValueError):normalize_atoms(atoms,1)


def test_scorer_failure_cannot_be_reported_as_zero():
    a=row();b={**row(arm='MEMORY_DYNAMIC'),'semantic_atoms':None}
    with pytest.raises(ValueError):summarize([a,b],resamples=20)


def test_manifest_requires_complete_shards(tmp_path):
    run={'source_sha256':'s','model_sha256':{},'benchmark_sha256':'b','sampling':{},'loop':{},
         'seeds':[0,1],'arms':['BASE'],'prompt_ids':['p0']}
    (tmp_path/'run.json').write_text(json.dumps(run))
    (tmp_path/'manifest.jsonl').write_text(json.dumps({**row(),'valid_file':False})+'\n')
    with pytest.raises(ValueError,match='incomplete'):load_manifests([tmp_path/'manifest.jsonl'])


def test_confirmation_cannot_use_historical_small_set():
    with pytest.raises(ValueError):validate_confirmation([{'prompt':'x'}],[0],['BASE'],None,'missing')


def test_cluster_ci_does_not_depend_on_shard_record_order():
    reference={(p,0):row(prompt=p,atoms=(0.,)) for p in ('c','b','a')}
    candidate={(p,0):row(prompt=p,arm='MEMORY_DYNAMIC',atoms=(float(p=='a'),)) for p in ('c','b','a')}
    result=paired_report(reference,candidate,resamples=200)
    reverse_reference=dict(reversed(list(reference.items())))
    reverse_candidate=dict(reversed(list(candidate.items())))
    assert result==paired_report(reverse_reference,reverse_candidate,resamples=200)
