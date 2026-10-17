"""One inference API for native, UND-parent and interaction memory controls."""
from pathlib import Path
import zipfile


ARMS = ('bagel', 'c2_no_memory', 'c2_memory', 'c2_memory_masked', 'c2_memory_swap')
INTERACTION_ARMS = ('bagel', 'parent_no_memory', 'parent_memory', 'interaction_memory',
                    'interaction_memory_masked', 'interaction_memory_swap')
ALL_ARMS = tuple(dict.fromkeys((*ARMS, *INTERACTION_ARMS)))


def resolve_checkpoint(path):
    """Resolve a training directory once, to the latest numbered checkpoint."""
    path = Path(path).resolve()
    if path.is_dir():
        candidates = [p for p in path.glob('step_*.pt') if p.stem[5:].isdigit()]
        if not candidates:
            raise FileNotFoundError(f'No numbered checkpoints in {path}; pass a checkpoint file')
        # Interrupted saves can leave the highest numbered file empty/truncated.
        complete = [p for p in candidates if zipfile.is_zipfile(p)]
        if not complete:
            raise FileNotFoundError(f'No complete numbered checkpoints in {path}')
        path = max(complete, key=lambda p: int(p.stem[5:]))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class InferenceEngine:
    """Keep one model resident; switch original/C2 UND weights explicitly.

    Construct from an original native runtime. Neutral memory seeds stay fixed
    at the native prefill, matching the existing C2 deployment path.
    """

    def __init__(self, runtime, checkpoint):
        from ..interaction.checkpoint import KIND, apply
        from ..interaction.student import InteractionStudent
        from .record import validate_checkpoint
        from .visual import VisualStudent
        from .visual_record import FORMAT_VERSION, STAGES
        self.interaction = checkpoint.get('kind') == KIND
        parent = checkpoint['parent'] if self.interaction else checkpoint
        validate_checkpoint(parent, runtime, STAGES)
        if parent.get('format_version') != FORMAT_VERSION or parent['stage'] not in ('C2-observe', 'C2-feedback'):
            raise ValueError('Memory controls require a format-4 Observe/Feedback parent')
        original_dtypes = {id(p): p.dtype for p in runtime.model.parameters()}
        if self.interaction:
            self.original = {name: runtime.decoder.get_parameter(name).detach().cpu().clone()
                             if name.startswith('layers.') else None for name in parent['und']}
            # Non-layer parents can also include embedding/norm/head parameters.
            for name in self.original:
                if self.original[name] is None:
                    group, suffix = name.split('.', 1)
                    module = {'final_norm': runtime.decoder.norm, 'embeddings': runtime.decoder.embed_tokens,
                              'lm_head': runtime.model.language_model.lm_head}[group]
                    self.original[name] = module.get_parameter(suffix).detach().cpu().clone()
            self.student = InteractionStudent(runtime, parent, checkpoint['config'])
            apply(checkpoint, self.student)
            self.student.eval().requires_grad_(False)
        else:
            self.student = VisualStudent(runtime).eval().requires_grad_(False)
            self.original = {name: p.detach().cpu().to(original_dtypes[id(p)]).clone()
                             for name, p in self.student.native_parameters().items()}
        if self.original.keys() != parent['und'].keys():
            raise ValueError('Checkpoint UND tensors differ from the selected parameter configuration')
        self.trained = parent['und']
        self.runtime = runtime
        self.step, self.stage = checkpoint['step'], checkpoint['stage']
        self.parent_step, self.parent_stage = parent['step'], parent['stage']
        self._weights = None

    @classmethod
    def load(cls, model, checkpoint, device='npu:0'):
        import torch
        from ..bagel.backbone import load_native
        from .runtime import DiagnosticRuntime
        saved = torch.load(str(resolve_checkpoint(checkpoint)), map_location='cpu', weights_only=True, mmap=True)
        cfg = dict(saved.get('parent', saved)['config']) if saved.get('kind') else dict(saved['config'])
        runtime = DiagnosticRuntime(load_native(model, device, cfg['timestep_shift']), cfg, 'C')
        runtime.native_files = saved['parent']['native_files'] if saved.get('kind') else saved['native_files']
        return cls(runtime, saved)

    def _select(self, original):
        identity = 'bagel' if original else 'c2'
        if self._weights == identity:
            return
        for name, value in (self.original if original else self.trained).items():
            parameter = self.student.native_tensor(name)
            # Restore dtype as well as values: original BAGEL norms must keep
            # their original precision, while C2 uses its saved FP32 tensors.
            parameter.data = value.to(device=parameter.device).clone()
        self.runtime._neutral_context = None
        self._weights = identity

    def generate(self, prompt, seed=0, *, arm=None, early=20,
                 loop_layers=None, donor_seed=None, steps=None):
        from .shallow_sampling import sample
        arm = arm or ('interaction_memory' if self.interaction else 'c2_memory')
        if arm not in ALL_ARMS:
            raise ValueError(f'Unknown arm: {arm}')
        if arm.startswith('interaction_') and not self.interaction:
            raise ValueError('Interaction arms require an interaction checkpoint')
        if arm.endswith('_swap') and donor_seed is None:
            donor_seed = (seed+1) % (2**63-1)
        self._select(arm == 'bagel')
        if self.interaction:
            self.student.adapters_enabled = arm.startswith('interaction_')
        enabled = arm not in ('bagel', 'c2_no_memory', 'parent_no_memory')
        mode = 'masked' if arm.endswith('_masked') else 'swap' if arm.endswith('_swap') else 'correct'
        return sample(self.student, prompt, seed, 2 if enabled else 1,
                      early if enabled else 0, loop_layers=loop_layers,
                      memory_mode=mode, donor_seed=donor_seed, steps=steps)
