"""Internal GLM scoring using the frozen benchmark rubrics."""

import base64
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

from .reference import t2i_protocols as protocol

REFERENCE = Path(__file__).parent / 'reference'
SCORE_SOURCES = REFERENCE / 'scoring_sources'
SYSTEM_BASIC = ('Follow the supplied image-editing rubric. Return the three requested score labels exactly once, '
                'each as Label: N, with N an integer from 1 to 5. Include at most 20 words of brief reasoning. '
                'Do not add an overall score.')
SYSTEM_UGE = ('Follow the supplied image-editing rubric. Include at most 20 words of brief reasoning and '
              'exactly one final line: Score: N, with N an integer from 1 to 5.')
MODEL = 'glm-5.3-flash'


def load_env_file(path):
    """Load only the two judge settings from the user-supplied env file."""
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if not text or text.startswith('#'):
            continue
        key, sep, value = text.partition('=')
        key = key.strip().removeprefix('export ').strip()
        if sep and key in ('METTO_GLM_KEY', 'METTO_BASE_URL') and not os.environ.get(key):
            os.environ[key] = value.strip().strip('"').strip("'")


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _t2i_source(bench):
    if bench in ('geneval', 'dpg', 't2i_corebench'):
        base = SCORE_SOURCES / 'bench_tools'
        sys.path.insert(0, str(base / 'common'))
        return _module('_bench_' + bench, base / 'benchmarks' / bench / 'scripts' / f'eval_{bench}_qwen35.py')
    if bench in ('geneval_pp', 't2i_compbench_pp'):
        return _module('_bench_pp', SCORE_SOURCES / 'pp' / 'eval_geneval_pp_qwen.py')
    if bench == 'wise_verified':
        return _module('_bench_wise', SCORE_SOURCES / 'wise' / 'vllm_eval.py')
    raise ValueError('No frozen text-to-image scoring protocol for ' + bench)


def image_url(path):
    image = Path(path)
    return 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode('ascii')


def call_messages(messages, *, max_tokens=1024):
    key = os.environ.get('METTO_GLM_KEY', '').strip()
    if not key:
        raise ValueError('Set METTO_GLM_KEY in the scoring process environment')
    base = os.environ.get('METTO_BASE_URL', 'https://metto-api.com').rstrip('/')
    body = {'model': MODEL, 'stream': False, 'temperature': 0, 'max_tokens': max_tokens,
            'messages': messages}
    req = Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}, method='POST')
    with urlopen(req, timeout=180) as response:
        result = json.load(response)
    choice = result['choices'][0]
    if choice.get('finish_reason') != 'stop' or not choice['message'].get('content'):
        raise ValueError('Judge did not return a complete final answer')
    return choice['message']['content']


def call_judge(system, question, images, *, max_tokens=1024, image_labels=None):
    content = [{'type': 'text', 'text': question}]
    for i, value in enumerate(images):
        if image_labels:
            content.append({'type': 'text', 'text': image_labels[i]})
        content.append({'type': 'image_url', 'image_url': {'url': value}})
    return call_messages([{'role': 'system', 'content': system},
                          {'role': 'user', 'content': content}], max_tokens=max_tokens)


def t2i_metadata(bench):
    meta = protocol.metadata(bench)
    meta['rubric_source_protocol'] = meta.pop('protocol_id')
    meta['protocol_id'] = 'glm-5.3-flash:' + bench
    return meta


def score_t2i_answers(row, answers):
    normalized = [protocol.normalize(row['benchmark'], a) for a in answers]
    score = protocol.image_score(row, normalized)
    return {'status': 'scored' if score is not None else 'unknown',
            'score': score, 'answers': normalized,
            'protocol': t2i_metadata(row['benchmark'])}


def score_t2i(row, image, source_root):
    bench = row['benchmark']
    if bench in protocol.BENCHES:
        questions = protocol.questions(row)
        answers, raw = [], []
        for item in questions:
            response = call_judge(item['system'], item['question'], [image_url(image)], max_tokens=256)
            raw.append(response)
            answers.append(protocol.parse_response(bench, response, 'stop'))
        score = protocol.image_score(row, answers)
        return {'status': 'scored' if score is not None else 'unknown', 'score': score,
                'answers': answers, 'raw': raw, 'protocol': t2i_metadata(bench)}
    module = _t2i_source(bench)
    if bench in ('geneval_pp', 't2i_compbench_pp'):
        question = ('Instruction:\n' + row['required_prompt'] + '\n\nExpectation checklist:\n'
                    + module.metadata_to_checklist(row['benchmark_metadata']))
        url = 'data:image/jpeg;base64,' + module.encode_jpeg(Path(image))
        raw = call_judge(module.SYSTEM_PROMPTS[bench], question, [url], max_tokens=256)
        result = module.extract_result(raw)
        return {'status': 'scored', 'score': result['correct'], 'raw': raw, 'details': result}
    if bench == 'wise_verified':
        messages = module.build_evaluation_messages(
            {'Prompt': row['instruction'], 'Explanation': row['expected_explanation']},
            base64.b64encode(Path(image).read_bytes()).decode())
        system = messages[0]['content'][0]['text']
        question = messages[1]['content'][0]['text']
        raw = call_judge(system, question, [image_url(image)], max_tokens=500)
        values = module.extract_scores(raw)
        return {'status': 'scored' if 'score' in values else 'unknown', 'score': values.get('score'), 'raw': raw}
    if bench == 'geneval':
        questions = module.build_questions(row)
    elif bench == 'dpg':
        questions = module.load_questions(Path(source_root) / 'bench_tools/common/dpg_bench.csv')[str(row['item_id'])]
    else:
        questions = row['checklist']
    if not questions:
        raise ValueError('Benchmark record has no scoring questions')
    answers, raw = [], []
    for item in questions:
        question = item['question']
        if bench == 't2i_corebench':
            question = f'Prompt: "{row["instruction"]}"\nQuestion: "{question}"'
        response = call_judge(module.SYSTEM_PROMPT, question, [module.jc.image_data_url(image)], max_tokens=64)
        response = module.jc.clean_think(response)
        answer = 'yes' if re.search(r'\byes\b', response, re.I) else 'no' if re.search(r'\bno\b', response, re.I) else None
        raw.append(response); answers.append(answer)
    if None in answers:
        return {'status': 'unknown', 'score': None, 'answers': answers, 'raw': raw}
    if bench == 'geneval':
        score = float(all(a == q['must_be'] for a, q in zip(answers, questions)))
    elif bench == 't2i_corebench':
        score = round(sum(a == 'yes' for a in answers) / len(answers), 6)
    else:
        adjusted = {}
        for q, answer in zip(questions, answers):
            adjusted[q['proposition_id']] = int(answer == 'yes' and all(adjusted.get(d, 1) for d in q['dependencies']))
        score = round(sum(adjusted.values()) / len(questions), 6)
    return {'status': 'scored', 'score': score, 'answers': answers, 'raw': raw}


def score_imgedit(row, source_image, image):
    suite = row['suite'].lower()
    if suite == 'basic':
        prompts = json.loads((REFERENCE / 'imgedit_basic_prompts.json').read_text())
        template = prompts[row['task_type']]
        labels = re.findall(r'^([^\n:]+): A number from 1 to 5\.$', template, re.M)
        system = SYSTEM_BASIC
    elif suite == 'uge':
        template = (REFERENCE / 'imgedit_uge_prompt.txt').read_text()
        labels = ['Score']
        system = SYSTEM_UGE
    else:
        raise ValueError('ImgEdit suite must be basic or uge')
    if (suite == 'basic' and len(labels) != 3) or not labels:
        raise ValueError('ImgEdit rubric labels differ from the frozen protocol')
    prompt = template.replace('<edit_prompt>', row['instruction'])
    raw = call_judge(system, prompt, [image_url(source_image), image_url(image)], max_tokens=1000)
    found = {}
    for line in raw.splitlines():
        match = re.fullmatch(r'([^:]+):\s*([1-5])\s*\.?', line.strip().replace('**', ''))
        if match and match.group(1).strip() in labels:
            name = match.group(1).strip()
            if name in found:
                raise ValueError('Judge repeated a rubric label')
            found[name] = int(match.group(2))
    if set(found) != set(labels):
        return {'status': 'unknown', 'score': None, 'raw': raw, 'labels': labels}
    values = [found[label] for label in labels]
    return {'status': 'scored', 'score': round(sum(values) / len(values), 2),
            'labels': labels, 'values': values, 'raw': raw}


def score_edit_metrics(row, image, assets_root, captions_path=None, device='cpu'):
    """Reuse the frozen CLIP/DINO and pixel metrics for MagicBrush and Emu Edit."""
    from types import SimpleNamespace
    from .reference import edit_metrics as metrics

    root = Path(assets_root)
    ready = json.loads((root / 'TOOL_ASSETS_READY.json').read_text())
    if ready.get('status') != 'READY':
        raise ValueError('Editing metric assets are not ready')
    captions = json.loads(Path(captions_path).read_text()) if captions_path else None
    if row['benchmark'] == 'magicbrush' and captions is None:
        raise ValueError('MagicBrush requires its frozen local caption file')
    ref = metrics.reference_for(row, captions)
    options = dict(device=device)
    for name, code, weights in (('clip', 'CLIP', 'ViT-B-32.pt'),
                                 ('dino', 'dino', 'dino_deitsmall16_pretrain.pth')):
        options[name + '_code'] = str(root / code)
        options[name + '_weights'] = str(root / 'weights' / weights)
        options[name + '_code_sha256'] = ready[name + '_code_sha256']
        options[name + '_weights_sha256'] = ready[name + '_weights_sha256']
    encoders = metrics.LocalEncoders(SimpleNamespace(**options))
    value = metrics.score_case(row['benchmark'], image, ref, encoders)
    value.update(status='scored' if any(v is not None for v in value['metrics'].values()) else 'unknown',
                 reference_role=ref['role'], metric_versions=encoders.versions,
                 protocol='magicbrush_single_native_tools_v1' if row['benchmark'] == 'magicbrush' else 'emu_tools_INTERNAL_v1')
    return value


def score_abacus(row, source_image, image):
    """Score operation count as a change, never as the target total."""
    from .reference import abacus_protocol as source

    label = source.labels(row)
    prompt = source.prompt(label, row['instruction'])
    raw = call_judge(source.SYSTEM, prompt, [image_url(source_image), image_url(image)],
                     max_tokens=2048, image_labels=('ORIGINAL', 'RESULT'))
    response = {'choices': [{'finish_reason': 'stop', 'message': {'content': raw}}]}
    parsed = source.parse_response({'labels': label}, response)
    return {'status': 'scored' if parsed else 'unknown', 'labels': label,
            'diagnosis': parsed, 'raw': raw, 'protocol': 'abacus_glm_internal_diagnostic_v1'}


def score_gedit(row, source_image, image):
    """GEdit v1 frozen semantic and perceptual rubrics with a GLM judge."""
    import math
    from .reference import gedit_eval as original, gedit_prompts as prompts, gedit_utils

    system = 'You are a strict professional digital artist and image editor evaluator.'
    sc = '\n'.join((prompts._context_no_delimit, prompts._prompts_0shot_two_image_edit_rule,
                    prompts._prompts_0shot_tie_rule_SC)).replace('<instruction>', row['instruction'])
    pq = '\n'.join((prompts._context_no_delimit, prompts._prompts_0shot_rule_PQ))
    source = original.image_data_url(Path(source_image))
    generated = original.image_data_url(Path(image))
    sc_raw = call_judge(system, sc, [source, generated], max_tokens=1000)
    sc_scores, sc_mode = original.parse_scores(sc_raw, gedit_utils.mllm_output_to_dict)
    if sc_scores is None:
        return {'status': 'unknown', 'score': None, 'sc_raw': sc_raw, 'failed_dimension': 'sc'}
    pq_raw = call_judge(system, pq, [generated], max_tokens=1000)
    pq_scores, pq_mode = original.parse_scores(pq_raw, gedit_utils.mllm_output_to_dict)
    if pq_scores is None:
        return {'status': 'unknown', 'score': None, 'sc_raw': sc_raw, 'pq_raw': pq_raw,
                'failed_dimension': 'pq'}
    semantic, quality = min(sc_scores), min(pq_scores)
    return {'status': 'scored', 'score': math.sqrt(semantic * quality),
            'semantic_score': semantic, 'quality_score': quality,
            'sc_scores': sc_scores, 'pq_scores': pq_scores,
            'sc_parse_mode': sc_mode, 'pq_parse_mode': pq_mode, 'sc_raw': sc_raw, 'pq_raw': pq_raw,
            'protocol': 'gedit_viescore_glm_internal_v1'}


def _gedit2_templates(root):
    import hashlib
    import yaml

    names = {'IF': 'instruction_following', 'VQ': 'visual_quality', 'VC': 'visual_consistency'}
    result = {}
    for dim, name in names.items():
        path = Path(root) / 'src/prompts/assets/vlm/assessment' / name / 'pairwise/v1.yaml'
        receipt = json.loads(path.with_name(path.name + '.public-receipt.json').read_text())
        if receipt.get('revision') != 'a083bbbf28f86c6e1f9e50c11082b80fccbe4ceb' or receipt.get('sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('GEdit v2 frozen rubric differs: ' + str(path))
        value = yaml.safe_load(path.read_text())
        if value['version'] != 'v1' or value['prompt_id'] != f'vlm/assessment/{name}/pairwise':
            raise ValueError('GEdit v2 rubric identity differs')
        result[dim] = value
    return result


def score_gedit2(row, source_image, baseline, image, reference_root):
    """Compare two edits in both orders; preserve ties and each dimension."""
    paths = {'input_image': image_url(source_image)}
    new, control = image_url(image), image_url(baseline)
    dimensions = {}
    for dim, template in _gedit2_templates(reference_root).items():
        outcomes, responses = [], []
        for candidates in ([new, control], [control, new]):
            content = []
            for block in template['user_prompt']:
                if 'text' in block:
                    text = re.sub(r'\{\{\s*instruction\s*\}\}', lambda _: row['instruction'], block['text'])
                    if '{{' in text:
                        raise ValueError('Unresolved GEdit v2 template variable')
                    content.append({'type': 'text', 'text': text})
                else:
                    item = block['image']
                    url = candidates[item['index']] if 'index' in item else paths[item['source']]
                    content.append({'type': 'image_url', 'image_url': {'url': url}})
            raw = call_messages([{'role': 'system', 'content': template['system_prompt']},
                                 {'role': 'user', 'content': content}])
            responses.append(raw)
            if '<think>' in raw:
                raw = raw.split('</think>')[-1] if '</think>' in raw else ''
            winner = None
            for match in re.finditer(r'\{[^{}]*\}', raw, re.S):
                try:
                    value = json.loads(match.group())
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and value.get('winner') in ('Image A', 'Image B', 'Tie'):
                    winner = value['winner']
                    break
            outcomes.append(winner)
        if None in outcomes:
            rate = None
        else:
            rate = ((1 if outcomes[0] == 'Image A' else .5 if outcomes[0] == 'Tie' else 0)
                    + (1 if outcomes[1] == 'Image B' else .5 if outcomes[1] == 'Tie' else 0)) / 2
        dimensions[dim] = {'result_win_rate': rate, 'winners': outcomes, 'raw': responses}
    return {'status': 'scored' if all(d['result_win_rate'] is not None for d in dimensions.values()) else 'unknown',
            'dimensions': dimensions, 'protocol': 'gedit2_pairwise_glm_internal_v1'}
