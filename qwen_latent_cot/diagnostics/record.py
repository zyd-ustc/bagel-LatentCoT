"""Auditable run manifests and mandatory frozen-weight guards."""
import json
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
import platform
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
import torch
from .data import file_hash, load_dataset
from ..bagel.accelerator import device_info


class FrozenGuard:
    def __init__(self, *modules):
        self.modules = modules
        self.reference = self.snapshot()
        self.check()

    def snapshot(self):
        return {(i, name): (id(p), p._version) for i, m in enumerate(self.modules) for name, p in m.named_parameters()}

    def check(self):
        if self.snapshot() != self.reference:
            raise RuntimeError('native parameter identity/version changed; stop interpreting results')
        if any(p.requires_grad or p.grad is not None for m in self.modules for p in m.parameters()):
            raise RuntimeError('native parameters have enabled gradients; stop')


def json_value(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, SimpleNamespace):
        return vars(value)
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    raise TypeError(f'non-serializable diagnostic record: {type(value).__name__}')


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=json_value)+'\n')


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def git_context(root):
    if not (root/'.git').exists():
        return dict(source_commit=None, branch=None, source_origin='installed_package_with_source_hashes')
    return dict(
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip(),
        source_origin='repository_with_source_hashes')


def manifest(runtime, args, data_path):
    bundle = runtime.bundle
    root = Path(__file__).resolve().parents[2]
    native_root = Path(bundle.model_path)
    native_files = [p for p in native_root.iterdir() if p.is_file() and
                    (p.suffix in ('.json', '.safetensors') or p.name.startswith(('tokenizer', 'vocab', 'merges')))]
    result = dict(protocol=runtime.config['protocol'], config=runtime.config,
                  config_hash=file_hash(args.config), dataset_hash=file_hash(data_path),
                  data_asset_hashes={str(p): file_hash(p) for p in sorted({r[k] for r in load_dataset(data_path) for k in ('image', 'state') if r.get(k)})},
                  timestamp_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(), command=sys.argv,
                  model_path=str(native_root.resolve()), dataset_path=str(Path(data_path).resolve()),
                  config_path=str(Path(args.config).resolve()),
                  **git_context(root),
                  source_hashes={str(p.relative_to(root)): file_hash(p)
                                 for base in (root/'qwen_latent_cot', root/'scripts')
                                 for p in base.rglob('*.py') if '__pycache__' not in p.parts},
                  model_files={p.name: file_hash(p) for p in sorted(native_files)},
                  native_config=json.loads(json.dumps(bundle.model.config.to_dict(), default=json_value)),
                  device=device_info(runtime.device), dtype=str(runtime.decoder.embed_tokens.weight.dtype),
                  diagnostic_attention='torch_SDPA_explicit_mask; CPU_float32',
                  python=platform.python_version(), topology=runtime.writer.topology,
                  vae_dtype=str(next(bundle.vae.parameters()).dtype),
                  depth=len(runtime.decoder.layers), slots=runtime.writer.slots,
                  seed_token_ids=runtime.seed_ids, seed_selection='first_16_non_special_tokens_of_fixed_neutral_text',
                  readout='post_block_same_layer_UND_norm_K_V_Knorm_unrotated',
                  trainable={n: p.numel() for n, p in runtime.writer.named_parameters()},
                  frozen_parameter_names=[n for n, _ in bundle.model.named_parameters()],
                  vae_frozen_parameter_names=[n for n, _ in bundle.vae.named_parameters()],
                  checkpoints={name: file_hash(value) for name in ('checkpoint', 'text_checkpoint', 'visual_checkpoint')
                               if (value := getattr(args, name, None))},
                  position_policy=dict(writer_start=runtime.config['writer_position_start'],
                                       caption_memory_start=0, caption_query_start=runtime.writer.slots,
                                       gen='original_prompt_then_memory_then_native_GEN_query_shifted_by_slots'),
                  freeze_check='parameter_identity_and_inplace_version_plus_requires_grad_and_grad',
                  status='started; engineering_validity_and_quality_unverified')
    return result


def save_checkpoint(path, runtime, step, stage, parent=None):
    torch.save(dict(adapters=runtime.writer.state_dict(), topology=runtime.writer.topology,
                    slots=runtime.writer.slots, rank=runtime.writer.rank, depth=len(runtime.decoder.layers),
                    seed_ids=runtime.seed_ids, config=runtime.config, step=step, stage=stage, parent=parent,
                    native_files=runtime.native_files, parent_hash=file_hash(parent) if parent else None), path)


def load_checkpoint(path, runtime, expected_stage=None):
    saved = torch.load(path, map_location=runtime.device, weights_only=True)
    if saved.get('native_files') != runtime.native_files:
        raise ValueError('checkpoint native model/tokenizer/VAE provenance differs')
    if (saved['topology'], saved['slots'], saved['rank'], saved['depth'], saved['seed_ids']) != (
        runtime.writer.topology, runtime.writer.slots, runtime.writer.rank, len(runtime.decoder.layers), runtime.seed_ids):
        raise ValueError('checkpoint topology/capacity/depth/initialization mismatch')
    for key in ('writer_position_start', 'caption_instruction', 'neutral_prompt', 'resolution',
                'adapter_type', 'posterior_seed_policy', 'posterior_seed', 'training_state_sampling', 'fact_scoring'):
        if saved['config'].get(key) != runtime.config[key]:
            raise ValueError('checkpoint interface configuration mismatch: '+key)
    if expected_stage and saved['stage'] != expected_stage:
        raise ValueError('checkpoint has incorrect training stage')
    runtime.writer.load_state_dict(saved['adapters'], strict=True)
    return saved
