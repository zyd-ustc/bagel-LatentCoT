"""Caption-derived atom rewards and two resident file-queue judges."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import time


def load_questions(path, rows):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    approved = {}
    captions = {r['caption'] for r in rows}
    for row in records:
        caption, questions = row['caption'], row['questions']
        if caption not in captions:
            raise ValueError('Reward questions must come from the selected training captions')
        if caption in approved or not questions:
            raise ValueError('Duplicate caption or empty reward atom list')
        if not row.get('approved') or not row.get('teacher'):
            raise ValueError('Reward cache requires teacher provenance and approved=true')
        if any(len(q) != 2 or not q[0].strip() or q[1].strip().lower() not in ('yes', 'no') for q in questions):
            raise ValueError('Each atom must be [question, Yes|No]')
        approved[caption] = questions
    if not approved:
        raise ValueError('No approved training reward questions')
    return approved


def answer_aliases(tokenizer, answer):
    aliases = (answer.capitalize(), answer.lower(), ' '+answer.lower(), ' '+answer.capitalize())
    unique = {}
    for alias in aliases:
        unique.setdefault(tokenizer.encode(alias)[0], alias)
    return list(unique.values())


def scorer(judge, source, device, *, training=False):
    """Reuse upstream decoding; training atoms also honor negative answers."""
    import torch
    if device.startswith('npu'):
        import torch_npu
        torch.npu.set_device(device)
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(judge, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(judge, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    tree = ast.parse(Path(source).read_text())
    names = {'return_numeric_string', 'construct_message_with_image', 'send_message_with_image', 'soft_tifa'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError('Unexpected upstream GenEval2 scoring functions')
    namespace = dict(torch=torch, qwen_model=model, qwen_processor=processor)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    original, hard, expected = namespace['send_message_with_image'], [], []
    aliases = {answer: answer_aliases(processor.tokenizer, answer) for answer in ('yes', 'no')} if training else {}

    def trace(prompt, image_filepath, answer_list=None):
        if training:
            answer_list = aliases[expected[len(hard)]]
        pred, prob = original(prompt, image_filepath, answer_list)
        hard.append(int(pred.lower() in answer_list))
        return pred, prob
    namespace['send_message_with_image'] = trace

    @torch.inference_mode()
    def score(questions, path):
        hard.clear()
        expected[:] = [q[1].strip().lower() for q in questions]
        _, atoms = namespace['soft_tifa'](questions, str(path))
        if not atoms or any(not math.isfinite(v) or not -1e-6 <= v <= 1+1e-6 for v in atoms):
            raise ValueError(f'GenEval2 returned invalid atom probabilities: {atoms}')
        return atoms, list(hard)
    return score


def atomic_json(path, value):
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    os.replace(temporary, path)


class JudgeQueue:
    def __init__(self, root, timeout):
        self.root, self.timeout = Path(root), timeout

    def submit(self, key, image, questions):
        worker = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2
        directory = self.root/f'judge_{worker}'
        if not (directory/'ready.json').exists():
            raise RuntimeError(f'Judge not ready: {directory}')
        image_path = directory/f'{key}.png'
        image.save(image_path)
        atomic_json(directory/f'{key}.request.json', dict(image=str(image_path.resolve()), questions=questions))
        return directory/f'{key}.result.json'

    def collect(self, requests):
        started = time.monotonic()
        pending, results = set(requests), {}
        while pending:
            for path in list(pending):
                if path.exists():
                    result = json.loads(path.read_text())
                    if 'error' in result:
                        raise RuntimeError(f'Judge failed: {result["error"]}')
                    results[path] = result
                    pending.remove(path)
            if time.monotonic()-started > self.timeout:
                raise TimeoutError(f'{len(pending)} judge requests exceeded {self.timeout}s')
            if pending:
                time.sleep(.25)
        return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--worker', type=int, required=True, choices=(0, 1))
    parser.add_argument('--judge', required=True)
    parser.add_argument('--official-source', required=True)
    parser.add_argument('--device', default='npu:0')
    args = parser.parse_args(argv)
    directory = Path(args.queue)/f'judge_{args.worker}'
    directory.mkdir(parents=True, exist_ok=True)
    score = scorer(args.judge, args.official_source, args.device, training=True)
    atomic_json(directory/'ready.json', dict(pid=os.getpid(), model=args.judge, device=args.device,
                                          official_source=args.official_source))
    while True:
        for path in sorted(directory.glob('*.request.json')):
            result_path = path.with_name(path.name.replace('.request.json', '.result.json'))
            if result_path.exists():
                continue
            try:
                row = json.loads(path.read_text())
                atoms, hard = score(row['questions'], row['image'])
                result = dict(reward=sum(atoms)/len(atoms), atoms=atoms, hard_atoms=hard)
            except Exception as error:
                result = dict(error=f'{type(error).__name__}: {error}')
            atomic_json(result_path, result)
        time.sleep(.25)


if __name__ == '__main__':
    main()
