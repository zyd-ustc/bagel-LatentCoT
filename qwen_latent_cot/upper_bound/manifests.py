"""Adapters from the official GenEval / GenEval++ manifests to atomic records.

Official manifests carry structured objects instead of a VQA list:
  geneval     : geneval_prompt + geneval_include[{class,count,color,position}]
  geneval_pp  : benchmark_metadata.prompt + benchmark_metadata.include/exclude
                [{class,count,color,region}]
"""
COUNT_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six',
               'seven', 'eight', 'nine', 'ten']


def _word(count):
    return COUNT_WORDS[count] if 0 <= count < len(COUNT_WORDS) else str(count)


def _atoms_from_objects(objects, exclude=()):
    atoms = []
    for entry in objects:
        noun = entry.get('class')
        if not noun:
            continue
        count = int(entry.get('count', 1))
        atoms.append([f'Are there exactly {_word(count)} {noun} in the image?', 'Yes'])
        if entry.get('color'):
            atoms.append([f'Is the {noun} {entry["color"]}?', 'Yes'])
        if entry.get('region') in ('left', 'right'):
            atoms.append([f'Is the {noun} on the {entry["region"]} side of the image?', 'Yes'])
        position = entry.get('position')
        if position:
            relation, other = position
            reference = objects[int(other)].get('class') if int(other) < len(objects) else None
            if reference:
                atoms.append([f'Is the {noun} to the {relation} the {reference}?', 'Yes'])
    for entry in exclude:
        noun = entry.get('class')
        if not noun:
            continue
        count = int(entry.get('count', 1))
        atoms.append([f'Are there exactly {_word(count)} {noun} in the image?', 'No'])
    return atoms


def normalize(row, index=0):
    """Return {prompt_id, prompt, vqa_list} for an official row, else None."""
    if row.get('vqa_list') and row.get('prompt'):
        return dict(prompt_id=str(row.get('prompt_id', f'row{index:05d}')),
                    prompt=row['prompt'], vqa_list=row['vqa_list'])

    metadata = row.get('benchmark_metadata') or {}
    objects = row.get('geneval_include') or metadata.get('include')
    prompt = row.get('geneval_prompt') or metadata.get('prompt') or row.get('prompt')
    if not objects or not prompt:
        return None
    atoms = _atoms_from_objects(objects, row.get('geneval_exclude') or metadata.get('exclude') or [])
    if not atoms:
        return None
    pid = row.get('case_key') or row.get('global_index') or row.get('dataset_index') or index
    return dict(prompt_id=str(pid), prompt=prompt, vqa_list=atoms)
