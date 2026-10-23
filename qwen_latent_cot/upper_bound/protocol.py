"""Explicit closed-loop upper bound: preview, judge, rollback, retry.

No training. Three arms share one base trajectory and one detection pass:
  loop      - roll back to t_early and rerun with the failure constraints
              appended to the prompt
  resample  - roll back to the same t_early, re-noise that state with a new
              seed, rerun with the original prompt
Both retries run exactly the same number of Euler steps, so the comparison is
compute matched. The base trajectory is the no-retry baseline.
"""
import json
from pathlib import Path

from .atoms import score_atoms, image_pass, correction_text


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def _atom_dicts(answers, vqa):
    rows = score_atoms(answers, vqa)
    for row in rows:
        row['correct'] = bool(row['correct'])
    return rows


def run(records, sampler, reader, config, output, log=print):
    out = Path(output)
    if out.exists():
        raise FileExistsError(f'Fresh run requires a fresh directory: {out}')
    images = out / 'images'
    images.mkdir(parents=True)

    read_indexes = [int(i) for i in config['read_indexes']]
    rollback = int(config['rollback_index'])
    seed0 = int(config.get('seed', 17))
    steps = len(sampler.grid) - 1
    retry_steps = steps - rollback
    if rollback >= min(read_indexes):
        raise ValueError('rollback_index must be earlier (smaller) than every read_index')

    results = []
    stream = (out / 'records.jsonl').open('w')
    for position, record in enumerate(records):
        pid = record['prompt_id']
        prompt = record['prompt']
        vqa = record['vqa_list']
        seed = seed0 + position

        trajectory = sampler.run(prompt, seed,
                                 save_indexes={rollback, *read_indexes},
                                 save_velocity={rollback})
        base_image = sampler.decode(trajectory['latent'])
        base_path = images / f'{pid}_base.png'
        base_image.save(base_path)
        base_atoms = _atom_dicts(reader.score(base_image, vqa), vqa)
        base_pass = image_pass(base_atoms)

        entry = dict(prompt_id=pid, prompt=prompt, seed=seed, vqa=vqa,
                     t_rollback=sampler.t_at(rollback), rollback_index=rollback,
                     steps_base=steps, steps_retry=retry_steps,
                     base=dict(image=base_path.name, atoms=base_atoms, passed=base_pass),
                     previews=[], retries=[])

        for index in read_indexes:
            preview_image = sampler.decode(trajectory['states'][index])
            preview_path = images / f'{pid}_preview_{index:02d}.png'
            preview_image.save(preview_path)
            preview_atoms = _atom_dicts(reader.score(preview_image, vqa), vqa)
            detected = not image_pass(preview_atoms)
            entry['previews'].append(dict(
                index=index, t=sampler.t_at(index), image=preview_path.name,
                atoms=preview_atoms, passed=not detected, detected_failure=detected))

            if not detected:
                continue

            failed = [row for row in preview_atoms if not row['correct']]
            suffix = correction_text(failed)
            corrected_prompt = (prompt + ' ' + suffix).strip() if suffix else prompt

            loop_latent = sampler.resume(trajectory['states'][rollback],
                                         sampler.flow(corrected_prompt), rollback)
            loop_image = sampler.decode(loop_latent)
            loop_path = images / f'{pid}_loop_{index:02d}.png'
            loop_image.save(loop_path)
            loop_atoms = _atom_dicts(reader.score(loop_image, vqa), vqa)

            resample_seed = seed + 100000 + index
            renoised = sampler.renoise(trajectory['states'][rollback],
                                       trajectory['velocities'][rollback],
                                       rollback, resample_seed)
            resample_latent = sampler.resume(renoised, sampler.flow(prompt), rollback)
            resample_image = sampler.decode(resample_latent)
            resample_path = images / f'{pid}_resample_{index:02d}.png'
            resample_image.save(resample_path)
            resample_atoms = _atom_dicts(reader.score(resample_image, vqa), vqa)

            retry = dict(
                index=index, t=sampler.t_at(index), detected_failure=True,
                correction=suffix, corrected_prompt=corrected_prompt,
                loop=dict(image=loop_path.name, atoms=loop_atoms,
                          passed=image_pass(loop_atoms)),
                resample=dict(seed=resample_seed, image=resample_path.name,
                              atoms=resample_atoms, passed=image_pass(resample_atoms)))
            if config.get('full_resample'):
                full_seed = seed + 200000 + index
                full_latent = sampler.run(prompt, full_seed)['latent']
                full_image = sampler.decode(full_latent)
                full_path = images / f'{pid}_full_{index:02d}.png'
                full_image.save(full_path)
                full_atoms = _atom_dicts(reader.score(full_image, vqa), vqa)
                retry['full_resample'] = dict(seed=full_seed, image=full_path.name,
                                              atoms=full_atoms,
                                              passed=image_pass(full_atoms),
                                              steps=steps)
            entry['retries'].append(retry)

        results.append(entry)
        stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
        stream.flush()
        log(json.dumps(dict(event='prompt', prompt_id=pid, base_pass=base_pass,
                            detected=[p['detected_failure'] for p in entry['previews']],
                            loop=[r['loop']['passed'] for r in entry['retries']],
                            resample=[r['resample']['passed'] for r in entry['retries']]),
                       ensure_ascii=False))

    stream.close()
    _json(out / 'records.json', results)
    _json(out / 'run.json', dict(config=config, prompts=len(results),
                                 steps_base=steps, steps_retry=retry_steps,
                                 arms=['base', 'loop', 'resample'],
                                 judge='native UND VQA', training=False))
    return results
