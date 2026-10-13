"""Read-only caption evidence for correct, disabled and swapped visual sources."""
import json
from pathlib import Path
import torch
from PIL import Image
from .data import file_hash
from .flow_training import conditional_gen, ExecutionCounter
from .record import FrozenGuard, append_json, write_json
from .visual_objectives import semantic_result


@torch.no_grad()
def source_review(student, tasks, rows, output):
    runtime = student.runtime
    student.eval().requires_grad_(False)
    guard = FrozenGuard(runtime.model, runtime.bundle.vae, student.text_teacher,
                        student.writer, student.visual)
    by_id = {row['sample_id']: row for row in rows}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    counter = ExecutionCounter(runtime)
    results = []
    try:
        for row, t, seed in tasks:
            donor = by_id.get(row.get('swap_id'))
            if (donor is None or donor['sample_id'] == row['sample_id']
                or donor['split'] != row['split'] or donor['prompt'] != row['prompt']):
                raise ValueError('source review requires a distinct same-split/same-prompt swap partner')
            state = runtime.training_state(row, t, seed)
            donor_state = runtime.training_state(donor, t, seed)
            if (state.metadata['prompt'], state.t) != (donor_state.metadata['prompt'], donor_state.t):
                raise ValueError('source swap must preserve actual prompt and t')
            base = conditional_gen(runtime, state)
            donor_base = conditional_gen(runtime, donor_state)
            banks = {'Correct': student.write(base.source, t),
                     'Source-off': student.write(base.source, t, source_off=True),
                     'Source-swap': student.write(donor_base.source, t)}
            key = f"{row['sample_id']}_t{t}_s{seed}"
            assets = {}
            for role, item, current, prediction in (
                    ('recipient', row, state, base), ('donor', donor, donor_state, donor_base)):
                image_name = f'{key}_{role}.jpg'
                with Image.open(item['image']) as image:
                    image.convert('RGB').resize((runtime.config['resolution'],)*2,
                        Image.Resampling.BICUBIC).save(output/image_name, quality=94)
                preview_name = f'{key}_{role}_preview.png'
                runtime.image(current.x_t-current.t*prediction.velocity).save(output/preview_name)
                assets[role] = dict(image=image_name, preview=preview_name,
                                    original_sha256=file_hash(item['image']),
                                    preview_policy='native conditional x0_hat; noisy-state proxy, not hidden-state truth')
            for arm, bank in banks.items():
                source = donor_state.metadata if arm == 'Source-swap' else state.metadata
                result = dict(key=key, sample_id=row['sample_id'], donor_id=donor['sample_id'],
                    split=row['split'], t=t, noise_seed=seed, arm=arm,
                    target_caption=row['caption'], donor_caption=donor['caption'],
                    state=state.metadata, source_state=source,
                    source_ignored=arm=='Source-off', images=assets,
                    caption=semantic_result(runtime, row, bank),
                    readout='frozen UND reads only Memory + fixed instruction; free caption has no target prefix',
                    label_origin=row.get('label_origin'), human_reviewed=row.get('human_reviewed'),
                    gen_memory_injection=False, frozen_guard='passed')
                append_json(output/'source_captions.jsonl', result); results.append(result)
            guard.check()
    finally:
        write_json(output/'execution.json', dict(states=len(tasks), gen_entries=counter.counts[0],
                   gen_layer_entries=counter.counts[1], und_layer_entries=counter.counts[2],
                   inference_only=True))
        counter.close()
    return results


def summarize_source_review(output, checkpoint):
    output = Path(output)
    files = sorted(output.glob('rank_*/source_captions.jsonl'))
    if not files and (output/'source_captions.jsonl').exists():
        files = [output/'source_captions.jsonl']
    records = []
    for path in files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for asset in row['images'].values():
                for key in ('image', 'preview'):
                    asset[key] = str(path.parent.relative_to(output)/asset[key])
            records.append(row)
    write_json(output/'source_review.json', dict(checkpoint=checkpoint,
               checkpoint_sha256=file_hash(checkpoint), records=records,
               state_count=len({row['key'] for row in records}),
               arms=['Correct','Source-off','Source-swap'],
               scope='caption source-dependence review; no GEN quality conclusion; human review pending'))
