import hashlib
import json
from pathlib import Path


def test_native_vendor_changes_are_only_namespace_and_attention_dispatch():
    root=Path(__file__).resolve().parents[1]
    ledger=json.loads((root/'qwen_latent_cot/bagel/modeling/native_source.json').read_text())
    reference=root.parent/'refs/Bagel'
    for destination,entry in ledger['files'].items():
        vendored=(root/destination).read_text()
        restored=vendored.replace('from qwen_latent_cot.bagel.modeling.bagel_utils import','from data.data_utils import')
        restored=restored.replace('from qwen_latent_cot.bagel.modeling.','from modeling.')
        restored=restored.replace('from qwen_latent_cot.bagel.attention import flash_attn_varlen_func','from flash_attn import flash_attn_varlen_func')
        if entry.get('removed_unused_imports')==['cv2']:
            restored=restored.replace('\nimport numpy as np','\nimport cv2\nimport numpy as np')
        assert hashlib.sha256(restored.encode()).hexdigest()==entry['sha256']
        if reference.exists():
            raw=(reference/entry['source']).read_text()
            if 'slice_end_line' in entry:raw=''.join(raw.splitlines(keepends=True)[:entry['slice_end_line']])
            assert raw==restored
