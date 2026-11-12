"""Read frozen benchmark records without using their historical outputs."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from PIL import Image



@dataclass(frozen=True)
class Case:
    kind: str
    benchmark: str
    record_id: str
    prompt: str
    seed: int
    source_image: Path | None
    row: dict


def load_case(kind, manifest, record_id, *, source_override=None):
    path = Path(manifest)
    if path.suffix == '.jsonl':
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        with path.open() as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data['records']
    matching = [row for row in rows if str(row.get('record_id')) == record_id]
    if len(matching) != 1:
        raise ValueError('record_id must be unique in the supplied manifest')
    row = matching[0]
    if kind == 't2i':
        prompt = row['original_prompt'] if 'original_prompt' in row else row.get('instruction', row.get('prompt'))
        if row.get('evaluation_prompt', prompt) != prompt or row.get('required_prompt', prompt) != prompt:
            raise ValueError('scoring prompt differs from original prompt')
        image = None
        seed = row.get('image_seed', row.get('seed'))
    elif kind == 'edit':
        prompt = row['instruction']
        image = Path(source_override or row['source_image'])
        if not image.is_file():
            raise FileNotFoundError(f'source image is unavailable: {image}')
        expected = row.get('source_sha256')
        if source_override and not expected:
            raise ValueError('Frozen record must include source_sha256 to use a copied source image')
        if expected and hashlib.sha256(image.read_bytes()).hexdigest() != expected:
            raise ValueError('source image differs from the frozen record')
        seed = row['seed']
    else:
        raise ValueError('kind must be t2i or edit')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('original prompt is missing')
    return Case(kind, row['benchmark'], record_id, prompt, int(seed), image, row)


def restore_geometry(image, source_size):
    """Undo the shared black canvas before comparing with source-size references."""
    width, height = image.size
    scale = min(width / source_size[0], height / source_size[1])
    w, h = round(source_size[0] * scale), round(source_size[1] * scale)
    left, top = (width - w) // 2, (height - h) // 2
    rect = (left, top, left + w, top + h)
    return image.crop(rect).resize(source_size, Image.Resampling.BICUBIC)
