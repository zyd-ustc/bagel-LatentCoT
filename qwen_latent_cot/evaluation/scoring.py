"""Offline judge. Original GenEval2 functions are bound to a local Qwen model."""
import ast
import hashlib
import json
import math
from pathlib import Path
from .io import sha256

QUALITY_PROMPT = ('Assess visual coherence and visible artifacts, independently of instruction following. '
    'Quality: 1=severely broken, 2=poor, 3=usable, 4=good, 5=excellent. '
    'Invalid means an unusable or severely corrupted image. '
    'Return only JSON with integer quality and boolean invalid.')


def normalize_atoms(atoms, expected):
    if len(atoms) != expected or not expected: raise ValueError('incomplete semantic atoms')
    values=[]
    for value in atoms:
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
            raise ValueError('nonfinite/invalid semantic atom')
        if not -1e-6 <= value <= 1+1e-6: raise ValueError(f'semantic atom out of range: {value}')
        values.append(min(1.,max(0.,float(value))))
    return values


class LocalScorer:
    def __init__(self, judge_path, official_source, device='cuda:0'):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor=AutoProcessor.from_pretrained(judge_path,local_files_only=True)
        self.model=AutoModelForImageTextToText.from_pretrained(judge_path,local_files_only=True,
            dtype=torch.bfloat16).to(device).eval()
        source=Path(official_source)
        # Load the audited upstream functions without executing its top-level HF download.
        tree=ast.parse(source.read_text())
        names={'return_numeric_string','construct_message_with_image','send_message_with_image','soft_tifa'}
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != names: raise ValueError('unexpected GenEval2 source API')
        namespace={'torch':torch,'qwen_model':self.model,'qwen_processor':self.processor}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
        self.soft_tifa=namespace['soft_tifa']
        weights=sorted(Path(judge_path).glob('*.safetensors'))
        self.provenance={'semantic':'upstream_GenEval2_soft_tifa_gm_functions',
            'scoring_code_sha256':sha256(Path(__file__)),
            'official_source_sha256':sha256(source),'official_source':str(source.resolve()),
            'judge_model_sha256':{p.name:sha256(p) for p in weights},
            'judge_config_sha256':sha256(Path(judge_path)/'config.json'),
            'judge_path':str(Path(judge_path).resolve()), 'device':device,
            'quality_prompt':QUALITY_PROMPT, 'quality_scale':'(integer_1_to_5 - 1)/4',
            'quality_is_proxy':True,'tiif':'local_deterministic_yes_no_variant',
            'boundary_tolerance':1e-6,'raw_atoms_preserved':True}

    def answer(self,path,question):
        import torch
        from PIL import Image
        with Image.open(path) as image:
            messages=[{'role':'user','content':[{'type':'image','image':image.convert('RGB')},
                                                   {'type':'text','text':question}]}]
            inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                return_dict=True,return_tensors='pt').to(self.model.device)
            with torch.inference_mode(): out=self.model.generate(**inputs,max_new_tokens=128,do_sample=False)
        return self.processor.batch_decode(out[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()

    def score(self,row,benchmark):
        from PIL import Image
        result={**row, 'semantic_atoms':None,'raw_semantic_atoms':None,'quality_proxy':None,'invalid':None,'semantic_skills':benchmark.get('skills'),
            'semantic_questions':benchmark.get('vqa_list',list(zip(benchmark.get('yn_question_list',[]),benchmark.get('yn_answer_list',[]))))}
        try:
            if not row['valid_file']: raise ValueError(row.get('decode_error') or 'invalid generated image')
            with Image.open(row['path']) as image:
                if (image.height,image.width)!=(row['height'],row['width']): raise ValueError('decoded shape mismatch')
                image.verify()
        except (OSError, ValueError) as error:
            count=len(benchmark.get('vqa_list',benchmark.get('yn_question_list',[])))
            if not count: raise ValueError('benchmark has no constraints')
            result.update(semantic_atoms=[0.]*count,raw_semantic_atoms=[0.]*count,
                quality_proxy=0.,invalid=True,decode_error=str(error));return result
        if 'vqa_list' in benchmark:
            _,atoms=self.soft_tifa(benchmark['vqa_list'],row['path'])
        elif 'yn_question_list' in benchmark:
            questions=benchmark['yn_question_list'];answers=benchmark['yn_answer_list']
            if not questions or len(questions)!=len(answers): raise ValueError('invalid TIIF questions')
            atoms=[]
            for q,a in zip(questions,answers):
                answer=self.answer(row['path'],q+' Answer only yes or no.').lower().rstrip('. ')
                if answer not in ('yes','no') or str(a).lower() not in ('yes','no'): raise ValueError('invalid TIIF answer')
                atoms.append(float(answer==str(a).lower()))
        else: raise ValueError('semantic evaluation requires benchmark constraints')
        result['raw_semantic_atoms']=atoms
        result['semantic_atoms']=normalize_atoms(atoms,len(benchmark.get('vqa_list',benchmark.get('yn_question_list',[]))))
        answer=self.answer(row['path'],QUALITY_PROMPT)
        payload=json.loads(answer.removeprefix('```json').removeprefix('```').removesuffix('```').strip())
        if type(payload.get('quality')) is not int or not 1<=payload['quality']<=5 or type(payload.get('invalid')) is not bool:
            raise ValueError('quality judge did not return the required JSON schema')
        result['quality_proxy']=(payload['quality']-1)/4;result['invalid']=payload['invalid']
        return result
