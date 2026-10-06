import importlib.util
import json
from pathlib import Path
import pytest
from PIL import Image
from qwen_latent_cot.evaluation.loop_depth import parse_depths,expand_arms
from qwen_latent_cot.evaluation.report import summarize
from qwen_latent_cot.evaluation.io import sha256
from test_evaluation import row


def test_depth_identity_single_base_and_invalid_depths():
    depths=parse_depths('3,1,2')
    assert depths==(1,2,3)
    specs=expand_arms(['BASE','LAYERWISE_MEMORY_KV'],depths,1)
    assert specs==[('BASE','BASE',0)]+[(f'LAYERWISE_MEMORY_KV_R{r}','LAYERWISE_MEMORY_KV',r) for r in depths]
    for invalid in ['0,1','1,1','-1','']:
        with pytest.raises(ValueError):parse_depths(invalid)
    with pytest.raises(ValueError):expand_arms(['BASE','MEMORY_LOOP'],depths,1)


def test_depth_report_has_paired_preceding_depth_comparators():
    rows=[row(atoms=(0.,0.,0.))]+[row(arm=f'LAYERWISE_MEMORY_KV_R{r}',
        atoms=tuple(float(i<r) for i in range(3))) for r in [1,2,3]]
    result=summarize(rows,resamples=20)
    assert result['LAYERWISE_MEMORY_KV_R2']['vs_LAYERWISE_MEMORY_KV_R1']['repair_count']==1
    assert result['LAYERWISE_MEMORY_KV_R3']['vs_LAYERWISE_MEMORY_KV_R1']['repair_count']==2
    assert result['LAYERWISE_MEMORY_KV_R3']['vs_BASE']['repair_count']==3


def test_portable_depth_gallery_validates_coverage_and_image_hashes(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/evaluate/export_comparison_html.py'
    spec=importlib.util.spec_from_file_location('gallery',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    image=tmp_path/'image.png';Image.new('RGB',(16,16),'red').save(image)
    rows=[{**row(arm=arm,prompt='<b>unsafe</b>'), 'index':0,'path':str(image),
           'valid_file':True,'image_sha256':sha256(image)} for arm in ['BASE','LAYERWISE_MEMORY_KV_R1','LAYERWISE_MEMORY_KV_R2','LAYERWISE_MEMORY_KV_R3']]
    report=tmp_path/'quality_report';report.mkdir()
    (report/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (report/'summary.json').write_text(json.dumps({'arms':summarize(rows,resamples=20),'source_sha256':'test'}))
    output=tmp_path/'comparison.html';result=module.export(tmp_path,output)
    assert result['images']==4 and output.read_text().count('src="data:image/png;base64,')==4
    assert '&lt;b&gt;unsafe&lt;/b&gt;' in output.read_text()
    Image.new('RGB',(16,16),'blue').save(image)
    with pytest.raises(ValueError,match='hash'):module.export(tmp_path,output)
