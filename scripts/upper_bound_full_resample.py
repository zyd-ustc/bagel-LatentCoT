"""Full 50-step resample for prompts that had a detected failure.

Reads the merged records of an upper-bound run, resamples every prompt that
produced at least one retry with a new seed, and stores the judged image.
"""
import argparse
import json
from pathlib import Path

from qwen_latent_cot.bagel.backbone import load_native
from qwen_latent_cot.correction.runtime import Runtime
from qwen_latent_cot.upper_bound.atoms import score_atoms, image_pass
from qwen_latent_cot.upper_bound.native import NativeSampler
from qwen_latent_cot.upper_bound.reader import UNDReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--seed-offset', type=int, default=300000)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    records = [r for r in json.loads(Path(args.records).read_text())
               if r['retries'] and not r['base']['passed']]
    records = records[args.shard::args.shards]
    output = Path(args.output)
    (output / 'images').mkdir(parents=True, exist_ok=True)

    bundle = load_native(args.model, args.device)
    runtime = Runtime(bundle, config)
    sampler = NativeSampler(runtime)
    reader = UNDReader(bundle, template=config.get('vqa_template', '{question}\nAnswer:'),
                       max_answer_tokens=config.get('max_answer_tokens', 8))
    print(json.dumps(dict(event='loaded', device=str(runtime.device), prompts=len(records))), flush=True)

    with (output / 'full_resample.jsonl').open('w') as stream:
        for record in records:
            seed = int(record['seed']) + args.seed_offset
            latent = sampler.run(record['prompt'], seed)['latent']
            image = sampler.decode(latent)
            name = f"{record['prompt_id']}_full.png"
            image.save(output / 'images' / name)
            atoms = score_atoms(reader.score(image, record['vqa']), record['vqa'])
            row = dict(prompt_id=record['prompt_id'], seed=seed, image=name,
                       atoms=atoms, passed=image_pass(atoms))
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print(json.dumps(dict(event='resample', prompt_id=record['prompt_id'],
                                  passed=row['passed'])), flush=True)


if __name__ == '__main__':
    main()
