import json
from pathlib import Path

import pytest
from PIL import Image


def test_t2i_uses_original_prompt_not_historical_image(tmp_path):
    from qwen_latent_cot.benchmarks.cases import load_case
    file = tmp_path / 'rows.jsonl'
    file.write_text(json.dumps({'benchmark': 'geneval2', 'record_id': 'r1', 'original_prompt': 'a blue bird', 'evaluation_prompt': 'a blue bird', 'image_path': '/old.png', 'image_seed': 9})+'\n')
    case = load_case('t2i', file, 'r1')
    assert case.prompt == 'a blue bird'
    assert case.seed == 9
    assert case.source_image is None


def test_edit_keeps_manifest_instruction_and_source(tmp_path):
    from qwen_latent_cot.benchmarks.cases import load_case
    image = tmp_path / 'source.png'
    Image.new('RGB', (24, 12)).save(image)
    file = tmp_path / 'rows.jsonl'
    file.write_text(json.dumps({'benchmark': 'imgedit', 'record_id': 'r1', 'instruction': 'Make it blue.', 'source_image': str(image), 'seed': 3})+'\n')
    case = load_case('edit', file, 'r1')
    assert case.source_image == image
    assert case.prompt == 'Make it blue.'


def test_duplicate_and_rewritten_records_are_rejected(tmp_path):
    from qwen_latent_cot.benchmarks.cases import load_case
    file = tmp_path / 'rows.jsonl'
    row = {'benchmark': 'geneval2', 'record_id': 'r1', 'original_prompt': 'a dog', 'evaluation_prompt': 'a cat', 'image_seed': 2}
    file.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='original'):
        load_case('t2i', file, 'r1')
    row['evaluation_prompt'] = 'a dog'
    file.write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError, match='unique'):
        load_case('t2i', file, 'r1')


def test_restore_output_to_original_field_of_view(tmp_path):
    from qwen_latent_cot.benchmarks.cases import restore_geometry
    image = Image.new('RGB', (512, 512), 'black')
    image.paste('red', (0, 128, 512, 384))
    restored = restore_geometry(image, (1024, 512))
    assert restored.size == (1024, 512)
    assert restored.getpixel((0, 0)) == (255, 0, 0)


def test_unknown_score_remains_unknown():
    from qwen_latent_cot.benchmarks.scoring import score_t2i_answers
    row={'benchmark':'geneval2', 'original_prompt':'a dog', 'evaluation_prompt':'a dog', 'native_qa':[{'question':'Is there a dog?', 'expected_answer':'yes'}]}
    result=score_t2i_answers(row, ['unknown'])
    assert result['status'] == 'unknown'
    assert result['score'] is None


def test_t2i_generated_pair_reuses_one_read_state(monkeypatch):
    import torch
    from types import SimpleNamespace
    from qwen_latent_cot.benchmarks import generate
    rt=SimpleNamespace(width=2,height=2,device=torch.device('cpu'),noise_scale=lambda:1)
    cfg={'read_time':0.36,'edit_steps':2}
    monkeypatch.setattr(generate,'phase_grid',lambda *args:[0,.36,1])
    monkeypatch.setattr(generate,'NativeT2I',lambda *args:None)
    monkeypatch.setattr(generate,'euler',lambda state,*args:state)
    monkeypatch.setattr(generate,'preview_image',lambda state,*args:state)
    monkeypatch.setattr(generate,'review_reflection',lambda *args:{'reflection':{'outcome':'edit','instruction':'Add a dog.','raw':'text'}})
    monkeypatch.setattr(generate,'_edit',lambda editor,state,*args:state+1)
    images, info=generate.t2i_pair(rt,None,cfg,'a dog',4)
    torch.testing.assert_close(images['result'],images['baseline']+1)
    assert info['reflection']['instruction']=='Add a dog.'


def test_edit_override_rejects_wrong_source(tmp_path):
    import hashlib
    from qwen_latent_cot.benchmarks.cases import load_case
    image = tmp_path / 'source.png'; Image.new('RGB', (2,2), 'red').save(image)
    other = tmp_path / 'wrong.png'; Image.new('RGB', (2,2), 'blue').save(other)
    row={'benchmark':'imgedit','record_id':'r1','instruction':'Make it blue',
         'source_image':str(image),'source_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'seed':4}
    manifest=tmp_path/'records.jsonl'; manifest.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='source image'):
        load_case('edit', manifest, 'r1', source_override=other)


def test_score_requires_generation_record_to_match_case(tmp_path):
    from types import SimpleNamespace
    from qwen_latent_cot.benchmarks.cli import score
    row={'benchmark':'geneval2','record_id':'r1','original_prompt':'a dog','evaluation_prompt':'a dog','image_seed':9}
    manifest=tmp_path/'rows.jsonl'; manifest.write_text(json.dumps(row)+'\n')
    gen=tmp_path/'gen'; gen.mkdir(); (gen/'case.json').write_text(json.dumps({'kind':'t2i','record_id':'different','benchmark':'geneval2','original_prompt':'a dog','seed':9}))
    args=SimpleNamespace(kind='t2i',manifest=str(manifest),record_id='r1',generation=str(gen),arm='result',output=str(tmp_path/'score.json'),source_root='none',source_image=None)
    with pytest.raises(ValueError, match='generation record'):
        score(args)


def test_score_rejects_substituted_image(tmp_path):
    from types import SimpleNamespace
    from qwen_latent_cot.benchmarks.cli import score
    row={'benchmark':'geneval2','record_id':'r1','original_prompt':'a dog','evaluation_prompt':'a dog','image_seed':9}
    manifest=tmp_path/'rows.jsonl'; manifest.write_text(json.dumps(row)+'\n')
    gen=tmp_path/'gen'; gen.mkdir(); Image.new('RGB',(2,2)).save(gen/'result.png')
    (gen/'case.json').write_text(json.dumps({'kind':'t2i','record_id':'r1','benchmark':'geneval2','original_prompt':'a dog','seed':9,'image_sha256':{'result':'not-the-saved-image'}}))
    args=SimpleNamespace(kind='t2i',manifest=str(manifest),record_id='r1',generation=str(gen),arm='result',output=str(tmp_path/'score.json'),source_root='none',source_image=None)
    with pytest.raises(ValueError, match='generated image'):
        score(args)


def test_imgedit_scoring_uses_source_then_result(monkeypatch, tmp_path):
    from qwen_latent_cot.benchmarks import scoring
    source, result = tmp_path/'source.png', tmp_path/'result.png'
    Image.new('RGB',(3,3),'red').save(source)
    Image.new('RGB',(3,3),'blue').save(result)
    seen = {}
    def judge(system, prompt, images, **kwargs):
        seen.update(system=system,prompt=prompt,images=images)
        import re
        labels=re.findall(r'^([^\n:]+): A number from 1 to 5\.$', prompt, re.M)
        return '\n'.join(label + ': 4' for label in labels)
    monkeypatch.setattr(scoring, 'call_judge', judge)
    row={'suite':'basic','task_type':'adjust','instruction':'Make it blue.'}
    scored=scoring.score_imgedit(row,source,result)
    assert scored['score']==4
    assert seen['images']==[scoring.image_url(source),scoring.image_url(result)]
    assert 'Make it blue.' in seen['prompt']


def test_six_benchmark_geneval_uses_source_questions(monkeypatch, tmp_path):
    from qwen_latent_cot.benchmarks import scoring
    image=tmp_path/'result.png'; Image.new('RGB',(2,2)).save(image)
    monkeypatch.setattr(scoring, 'call_judge', lambda *args,**kwargs:'yes')
    row={'benchmark':'geneval','geneval_include':[{'class':'dog','count':1}], 'geneval_exclude':[]}
    result=scoring.score_t2i(row,image,source_root='unused')
    assert result['status']=='scored' and result['score']==1


def test_reflection_failure_never_becomes_baseline_score(monkeypatch):
    import torch
    from types import SimpleNamespace
    from qwen_latent_cot.benchmarks import generate
    rt=SimpleNamespace(width=2,height=2,device=torch.device('cpu'),noise_scale=lambda:1)
    cfg={'read_time':0.36,'edit_steps':2}
    monkeypatch.setattr(generate,'phase_grid',lambda *args:[0,.36,1])
    monkeypatch.setattr(generate,'NativeT2I',lambda *args:None)
    monkeypatch.setattr(generate,'euler',lambda state,*args:state)
    monkeypatch.setattr(generate,'preview_image',lambda state,*args:state)
    monkeypatch.setattr(generate,'review_reflection',lambda *args:{'reflection':{'outcome':'failed_parse','raw':'bad'}})
    with pytest.raises(ValueError, match='reflection'):
        generate.t2i_pair(rt,None,cfg,'a dog',4)


def test_env_file_loads_only_judge_configuration(tmp_path, monkeypatch):
    import os
    from qwen_latent_cot.benchmarks.scoring import load_env_file
    file=tmp_path/'.env'; file.write_text('METTO_GLM_KEY=example\nMETTO_BASE_URL=https://example.test\nOTHER_KEY=not-used\n')
    monkeypatch.delenv('METTO_GLM_KEY',raising=False)
    monkeypatch.delenv('METTO_BASE_URL',raising=False)
    monkeypatch.delenv('OTHER_KEY',raising=False)
    load_env_file(file)
    assert os.environ['METTO_GLM_KEY']=='example'
    assert os.environ['METTO_BASE_URL']=='https://example.test'
    assert 'OTHER_KEY' not in os.environ


def test_two_editors_share_reflection_and_initial_state(monkeypatch):
    import torch
    from types import SimpleNamespace
    from qwen_latent_cot.benchmarks import generate
    rt=SimpleNamespace(width=2,height=2,device=torch.device('cpu'),noise_scale=lambda:1)
    cfg={'read_time':0.36,'edit_steps':2}
    monkeypatch.setattr(generate,'phase_grid',lambda *args:[0,.36,1])
    monkeypatch.setattr(generate,'NativeT2I',lambda *args:None)
    monkeypatch.setattr(generate,'euler',lambda state,*args:state)
    monkeypatch.setattr(generate,'preview_image',lambda state,*args:state)
    monkeypatch.setattr(generate,'review_reflection',lambda *args:{'reflection':{'outcome':'edit','instruction':'Add a dog.','raw':'text'}})
    mode={'trained':False}
    monkeypatch.setattr(generate,'_edit',lambda editor,state,*args:state+(2 if mode['trained'] else 1))
    def load(): mode['trained']=True
    images,_=generate.t2i_pair(rt,None,cfg,'a dog',4,load_trained=load)
    torch.testing.assert_close(images['native_result'],images['baseline']+1)
    torch.testing.assert_close(images['result'],images['baseline']+2)


def test_glm_request_sends_image_and_reads_final_content(monkeypatch):
    import io
    from qwen_latent_cot.benchmarks import scoring
    seen = {}
    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self,*args): self.close()
    def open_request(req,timeout):
        seen['body'] = json.loads(req.data)
        seen['auth'] = req.get_header('Authorization')
        return Response(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'yes'}}]}).encode())
    monkeypatch.setenv('METTO_GLM_KEY','test-only')
    monkeypatch.setattr(scoring,'urlopen',open_request)
    assert scoring.call_judge('system','question',['data:image/png;base64,eA=='])=='yes'
    assert seen['body']['model']=='glm-5.3-flash'
    assert seen['body']['messages'][1]['content'][1]['type']=='image_url'
    assert seen['auth']=='Bearer test-only'


def test_gedit_uses_separate_semantic_and_quality_calls(monkeypatch, tmp_path):
    import math
    from qwen_latent_cot.benchmarks import scoring
    source, result=tmp_path/'source.png', tmp_path/'result.png'
    Image.new('RGB',(4,4),'red').save(source); Image.new('RGB',(4,4),'blue').save(result)
    calls=[]
    def judge(system,prompt,images,**kwargs):
        calls.append((prompt,images))
        return '{"score": [9, 8]}' if len(calls)==1 else '{"score": [4, 5]}'
    monkeypatch.setattr(scoring,'call_judge',judge)
    value=scoring.score_gedit({'instruction':'Make it blue.'},source,result)
    assert value['semantic_score']==8 and value['quality_score']==4
    assert value['score']==math.sqrt(32)
    assert len(calls[0][1])==2 and len(calls[1][1])==1
    assert 'Make it blue.' in calls[0][0]


def test_t2i_rejects_empty_original_even_with_rewrite(tmp_path):
    from qwen_latent_cot.benchmarks.cases import load_case
    file=tmp_path/'rows.jsonl'; file.write_text(json.dumps({'benchmark':'geneval2','record_id':'r1','original_prompt':'','instruction':'rewritten','evaluation_prompt':'rewritten','image_seed':9})+'\n')
    with pytest.raises(ValueError,match='original'):
        load_case('t2i',file,'r1')


def test_t2i_rejects_generation_and_scoring_prompt_mismatch(tmp_path):
    from qwen_latent_cot.benchmarks.cases import load_case
    file=tmp_path/'rows.jsonl'; file.write_text(json.dumps({'benchmark':'geneval_pp','record_id':'r1','instruction':'a dog','required_prompt':'a cat','seed':9})+'\n')
    with pytest.raises(ValueError,match='original'):
        load_case('t2i',file,'r1')


def test_gedit2_pairwise_swaps_positions_and_scores_ties(monkeypatch, tmp_path):
    from qwen_latent_cot.benchmarks import scoring
    paths=[tmp_path/x for x in ('source.png','baseline.png','result.png')]
    for path in paths: Image.new('RGB',(2,2)).save(path)
    template={'system_prompt':'system','user_prompt':[{'text':'Instruction: {{ instruction }}'},
             {'image':{'source':'input_image'}},{'image':{'source':'edited_images','index':0}},
             {'image':{'source':'edited_images','index':1}}]}
    monkeypatch.setattr(scoring,'_gedit2_templates',lambda root: {x:template for x in ('IF','VQ','VC')})
    calls=[]
    def judge(messages, **kwargs):
        calls.append(messages)
        return '{"winner":"Image A"}' if len(calls)%2 else '{"winner":"Image B"}'
    monkeypatch.setattr(scoring,'call_messages',judge)
    out=scoring.score_gedit2({'instruction':'make it blue'},*paths,reference_root='unused')
    assert out['status']=='scored'
    assert all(v['result_win_rate']==1.0 for v in out['dimensions'].values())
    assert len(calls)==6
    assert calls[0][1]['content'][0]['text']=='Instruction: make it blue'
    assert calls[0][1]['content'][2] == calls[1][1]['content'][3]


def test_abacus_never_confuses_operation_count_with_final_total():
    from qwen_latent_cot.benchmarks.reference import abacus_protocol as original
    label={'operation':'add','requested_k':2}
    payload={'operation':'add','requested_k':2,
             'observations':{'target_before':1,'target_after':3,'source_before':None,'source_after':None,'changed_count':2},
             'dimensions':{k:{'verdict':'yes','evidence':'visible'} for k in original.dims('add')}}
    result=original.parse_response({'labels':label},{'choices':[{'finish_reason':'stop','message':{'content':json.dumps(payload)}}]})
    assert result['observations']['changed_count']==2
    payload['observations']['changed_count']=None
    result=original.parse_response({'labels':label},{'choices':[{'finish_reason':'stop','message':{'content':json.dumps(payload)}}]})
    assert result is None


def test_metadata_override_checks_frozen_identity(tmp_path):
    import hashlib
    from qwen_latent_cot.benchmarks.scoring import _metadata_path
    file=tmp_path/'metadata.json'; file.write_text('{"id":"r1"}')
    row={'evaluation_metadata':'/old/metadata.json','evaluation_metadata_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    assert _metadata_path(row,file)==file
    file.write_text('{"id":"r2"}')
    with pytest.raises(ValueError,match='metadata'):
        _metadata_path(row,file)


def test_magicbrush_rejects_nonfrozen_caption_before_loading_models(tmp_path):
    import hashlib
    from qwen_latent_cot.benchmarks.scoring import score_edit_metrics
    assets=tmp_path/'assets'; assets.mkdir(); (assets/'TOOL_ASSETS_READY.json').write_text('{"status":"READY"}')
    manifest=tmp_path/'MANIFEST.jsonl'; manifest.write_text('frozen manifest\n')
    captions=tmp_path/'local_descriptions.json'; captions.write_text('{}')
    receipt={'benchmark':'magicbrush','manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
             'source_files':[{'path':'/original/local_descriptions.json','sha256':'different'}]}
    (tmp_path/'READY.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='caption file'):
        score_edit_metrics({'benchmark':'magicbrush'},tmp_path/'image.png',assets,captions,manifest=manifest)
