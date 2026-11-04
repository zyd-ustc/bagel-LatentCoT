"""Load one frozen native release and broadcast it across the worker group."""

import torch
import torch.distributed as dist
from . import distributed as group
from .runtime import Runtime


def load_runtime(model_path, source, config, *, distributed=False):
    """Rank0 loads release weights; other ranks materialize parameter shapes."""

    if not distributed or dist.get_rank() == 0:
        rt = Runtime.load(model_path, source, config, "cpu")
    else:
        from .runtime import upstream
        from transformers import AutoTokenizer
        from accelerate import init_empty_weights

        native = upstream(source)
        cfg = native.NEOChatConfig.from_pretrained(model_path, local_files_only=True)
        if cfg.is_moe_llm:
            raise ValueError("First correction implementation targets dense U1.5")
        with init_empty_weights(include_buffers=False):
            model = native.NEOChatModel(cfg)
        rt = Runtime(
            model,
            AutoTokenizer.from_pretrained(model_path, local_files_only=True),
            config,
        )
    if distributed:
        identity = [rt.identity]
        dist.broadcast_object_list(identity, src=0)
        rt.identity = identity[0]
    return rt


def load_native(model, source, config, device, distributed=False):
    # Link formation precedes the only full release disk read.
    group.barrier()
    rt = load_runtime(model, source, config, distributed=distributed)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    m = rt.model
    if distributed and group.rank() != 0:
        from accelerate.utils import set_module_tensor_to_device

        # Preserve real initialized buffers; to_empty() would erase their values.
        for name, p in list(m.named_parameters()):
            set_module_tensor_to_device(
                m,
                name,
                device,
                value=torch.empty(p.shape, device=device, dtype=dtype),
                dtype=dtype,
            )
        for module in m.modules():
            for name, b in module._buffers.items():
                if b is not None:
                    module._buffers[name] = b.to(
                        device=device, dtype=dtype if b.is_floating_point() else b.dtype
                    )
    else:
        m.to(device=device, dtype=dtype)
    if distributed:
        for p in m.parameters():
            dist.broadcast(p.data, src=0)
        for b in m.buffers():
            dist.broadcast(b, src=0)
    m.eval().requires_grad_(False)
    rt.device = device
    return rt
