"""Use original benchmark prompt families to exclude evaluation training sources."""
import json
from pathlib import Path

BENCHMARKS=('geneval','geneval_pp','t2i_compbench_pp','dpg','t2i_corebench','wise_verified')


def evaluation_prompts(root):
    prompts=set()
    for benchmark in BENCHMARKS:
        path=Path(root)/f'{benchmark}.jsonl'
        for line in path.read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line)
            prompt=row.get('instruction',row.get('required_prompt'))
            if not prompt:raise ValueError('Evaluation record has no generation prompt')
            prompts.add(prompt)
    return prompts,BENCHMARKS
