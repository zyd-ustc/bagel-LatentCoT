"""One inference API for native, UND-parent and interaction memory controls."""
from pathlib import Path
import zipfile


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


def load_runtime(model, parent, device):
    from ..bagel.backbone import load_native
    from .runtime import DiagnosticRuntime
    cfg = dict(parent['config'])
    cfg['source_layers'] = list(range(parent['depth']))
    runtime = DiagnosticRuntime(load_native(model, device, cfg['timestep_shift']), cfg, 'C')
    runtime.native_files = parent['native_files']
    return runtime


class InferenceEngine:
    """The same new loop, initialized from C2 or restored from a v3 checkpoint."""
    def __init__(self, runtime, parent, config, *, saved=None):
        from ..interaction.checkpoint import apply
        from ..interaction.student import InteractionStudent
        self.runtime = runtime
        self.student = InteractionStudent(runtime, parent, config)
        if saved is not None:
            apply(saved, self.student)
        self.student.eval().requires_grad_(False)

    @classmethod
    def load(cls, model, checkpoint, device='npu:0'):
        from ..interaction.checkpoint import read, KINDS
        saved = read(checkpoint)
        if saved.get('kind') not in KINDS:
            raise ValueError('Sampling requires a full-loop v3 or Stage A v4 checkpoint; use --parent for C2 initialization')
        runtime = load_runtime(model, saved['parent'], device)
        return cls(runtime, saved['parent'], saved['config'], saved=saved)

    @classmethod
    def from_parent(cls, model, parent, config, device='npu:0'):
        from ..interaction.checkpoint import read
        saved = read(parent)
        if saved.get('format_version') != 4 or saved.get('stage') not in ('C2-observe', 'C2-feedback'):
            raise ValueError('--parent requires a format-4 C2 Observe/Feedback checkpoint')
        runtime = load_runtime(model, saved, device)
        return cls(runtime, saved, config)

    def generate(self, prompt, seed=0, *, native=False, on_step=None, feedback_updates=None):
        from ..interaction.sampling import sample
        return sample(self.student, prompt, seed, native=native, on_step=on_step, feedback_updates=feedback_updates)
