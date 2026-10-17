"""Short native UND gradient and frozen GEN check."""
import torch
from .flow_training import conditional_gen
from .visual_objectives import semantic_loss
from .record import FrozenGuard, write_json


def check(student, rows, output):
    runtime = student.runtime
    guard = FrozenGuard(runtime.model, runtime.bundle.vae, exclude_trainable=True)
    original = student.native_state_dict()
    optimizer = torch.optim.AdamW(student.parameters(), lr=runtime.config['learning_rate'])
    records = []
    try:
        for index, row in enumerate(rows[:2]):
            optimizer.zero_grad(set_to_none=True)
            row = dict(row, task='observe')
            state = runtime.training_state(row, .15, index)
            with torch.no_grad():
                source = conditional_gen(runtime, state).source
            bank = student.write(source, state.t)
            loss = semantic_loss(runtime, row, bank)
            loss.backward()
            groups = {name: sum(float(p.grad.float().square().sum()) for p in parameters if p.grad is not None)**.5
                      for name, parameters in student.parameter_groups().items()}
            if not torch.isfinite(loss) or not all(value > 0 for value in groups.values()):
                raise RuntimeError('native UND loss or selected gradient is invalid')
            optimizer.step(); guard.check()
            records.append(dict(loss=float(loss.detach()), gradient_norms=groups))
    finally:
        student.load_native_state_dict(original)
        student.zero_grad(set_to_none=True)
    write_json(output/'e0.json', dict(status='passed', records=records,
                                    trainable_parameters=sum(p.numel() for p in student.parameters())))
