"""Restore actual goal prompts in the existing paired manifest, preserving assets."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def restore(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        if row.get('label_kind') != 'target_clean' or not isinstance(row.get('target_caption'), str) or not row['target_caption'].strip():
            raise ValueError('expected target_clean pairs with nonempty target_caption')
    changed = sum(row.get('prompt') != row['target_caption'] for row in rows)
    if not changed:
        return dict(rows=len(rows), changed=0, sha256=digest(path))
    before = digest(path)
    backup = path.with_suffix('.neutral_prompt.jsonl')
    if backup.exists():
        raise ValueError(f'backup already exists: {backup}')
    shutil.copy2(path, backup)
    temporary = path.with_suffix('.goal_prompt.partial')
    with temporary.open('x') as stream:
        for row in rows:
            row['prompt'] = row['target_caption']
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    after = digest(temporary)
    temporary.replace(path)
    result = dict(rows=len(rows), changed=changed, splits=dict(Counter(r['split'] for r in rows)),
                  before_sha256=before, sha256=after, backup=str(backup),
                  timestamp_utc=datetime.now(timezone.utc).isoformat(), modified_fields=['prompt'])
    for suffix in ('.summary.json', '.validation.json'):
        metadata = path.with_suffix(suffix)
        if metadata.exists():
            old = metadata.with_name(metadata.name + '.neutral_prompt')
            shutil.copy2(metadata, old)
            value = json.loads(metadata.read_text())
            field = 'dataset_sha256' if suffix == '.validation.json' else 'sha256'
            value[field] = after
            value['prompt_restore'] = result
            if suffix == '.summary.json':
                value['caption_policy'] = 'complete source prompt; prompt equals target_caption; stage chooses neutral/deployment context'
            # Historical image verification remains historical; no new image scan.
            metadata.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    path.with_suffix('.prompt_restore.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='/data/zyd_workspace/bagel-LatentCoT/datasets/visual_memory_full.jsonl')
    print(json.dumps(restore(parser.parse_args().data), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
