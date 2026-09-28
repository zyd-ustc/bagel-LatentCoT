"""Synchronous data parallelism for the multi-entrypoint BAGEL LoRA trainer.

Only adapter gradients are communicated. Each rank keeps a frozen backbone and
performs its own native rollout, Read and local counterfactual replay. Explicit
all-reduce avoids routing frozen teacher/public-method calls through a DDP wrapper.
"""
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class RankInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ
        world = int(env.get("WORLD_SIZE", "1"))
        rank = int(env.get("RANK", "0"))
        local = int(env.get("LOCAL_RANK", "0"))
        if world < 1 or not 0 <= rank < world or local < 0:
            raise ValueError("invalid torchrun RANK/LOCAL_RANK/WORLD_SIZE")
        if world > 1 and any(key not in env for key in ("RANK", "LOCAL_RANK")):
            raise ValueError("multi-GPU training must be launched with torchrun")
        return cls(rank, local, world)


def distributed_contract(config, stage, record_count=None, *, info=None):
    """Pure configuration validation; --validate-only never initializes CUDA/NCCL."""
    info = info or RankInfo.from_env()
    batch = config["batch_size"]
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 2:
        raise ValueError("batch_size is per rank and must be an integer >=2")
    if info.world_size > 1:
        if stage not in ("reader", "writer", "loop"):
            raise ValueError("multi-GPU v2 currently supports reader/writer/loop, not GRPO or eval")
        if str(config.get("device", "auto")) not in ("auto", "cuda"):
            raise ValueError("multi-GPU requires device=auto or cuda; LOCAL_RANK chooses the GPU")
    global_batch = batch * info.world_size
    if record_count is not None and record_count < global_batch:
        raise ValueError(f"need at least global_batch_size={global_batch} distinct records; got {record_count}")
    return dict(world_size=info.world_size, per_rank_batch_size=batch,
                global_batch_size=global_batch, gradient_accumulation_steps=1,
                distributed_strategy="mean_lora_gradients" if info.world_size > 1 else "single_device",
                counterfactual_shuffle_scope="within_rank", learning_rate_scaled=False,
                data_sampling="epoch_shuffle_cyclic_padding" if config.get("shuffle_data", False)
                              else "sequential_cyclic")


class RankBatchSampler:
    """Disjoint equal-size rank batches; no dropped rank and no duplicate per global batch."""
    def __init__(self, size, batch_size, *, rank=0, world_size=1, seed=42, shuffle=False):
        if min(size, batch_size, world_size) < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid sampler dimensions/rank")
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size = rank, world_size
        self.global_batch = batch_size * world_size
        if size < self.global_batch:
            raise ValueError("dataset must contain at least one complete global batch")
        self.seed, self.shuffle = int(seed), bool(shuffle)
        self.epoch, self.order = None, None

    def indices(self, batch_index):
        if batch_index < 0:
            raise ValueError("batch_index must be nonnegative")
        if not self.shuffle:
            start = batch_index * self.global_batch + self.rank * self.batch_size
            return [(start + j) % self.size for j in range(self.batch_size)]
        batches_per_epoch = math.ceil(self.size / self.global_batch)
        epoch, offset = divmod(batch_index, batches_per_epoch)
        if epoch != self.epoch:
            self.order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
            self.epoch = epoch
        start = offset * self.global_batch + self.rank * self.batch_size
        return [self.order[(start + j) % self.size] for j in range(self.batch_size)]


class DistributedContext:
    def __init__(self, info=None, *, device="cpu"):
        self.info = info or RankInfo()
        self.device = torch.device(device)

    @property
    def primary(self):
        return self.info.rank == 0

    @property
    def active(self):
        return self.info.world_size > 1

    @classmethod
    def start(cls, config):
        info = RankInfo.from_env()
        if info.world_size == 1:
            return cls(info)
        if not torch.cuda.is_available() or info.local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK={info.local_rank} has no visible CUDA GPU")
        if not dist.is_nccl_available():
            raise RuntimeError("multi-GPU BAGEL requires a PyTorch build with NCCL")
        device = torch.device("cuda", info.local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", init_method="env://",
            timeout=timedelta(seconds=int(config.get("distributed_timeout_seconds", 1800))))
        return cls(info, device=device)

    def close(self):
        if self.active and dist.is_initialized():
            dist.destroy_process_group()

    def gather(self, value):
        if not self.active:
            return [value]
        values = [None] * self.info.world_size
        dist.all_gather_object(values, value)
        return values

    def fail_if(self, condition, message, error_type=RuntimeError):
        failed = bool(condition)
        if self.active:
            flag = torch.tensor(int(failed), device=self.device, dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item():
                messages = self.gather(f"rank {self.info.rank}: {message}" if failed else None)
                raise error_type("; ".join(value for value in messages if value is not None))
        elif failed:
            raise error_type(message)

    def primary_call(self, callback):
        """Only rank zero mutates shared artifacts; broadcast failures instead of hanging peers."""
        if not self.active:
            return callback()
        envelope = [None]
        if self.primary:
            try:
                envelope[0] = dict(ok=True, value=callback())
            except Exception as exc:
                envelope[0] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(envelope, src=0)
        if not envelope[0]["ok"]:
            raise RuntimeError(f"rank-zero artifact operation failed: {envelope[0]['error']}")
        return envelope[0]["value"]

    def verify_inputs(self, config, records):
        if not self.active:
            return
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
        for row in records:
            fingerprint.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
            fingerprint.update(b"\n")
        digests = self.gather(fingerprint.hexdigest())
        if len(set(digests)) != 1:
            raise RuntimeError("ranks loaded different configuration/data/order")

    @torch.no_grad()
    def synchronize_initial_adapters(self, model):
        if not self.active:
            return
        # All frozen base weights are loaded from the same verified checkpoint.
        # Broadcast adapters (including frozen routes) and deterministic m0 once.
        selected = [(name, p) for name, p in model.named_parameters()
                    if p.requires_grad or ".lora_A." in name or ".lora_B." in name
                    or name == "loop_memory"]
        signatures = self.gather([(name, list(p.shape), str(p.dtype), p.requires_grad) for name, p in selected])
        if any(s != signatures[0] for s in signatures):
            raise RuntimeError("adapter/trainable tensor layout differs across ranks")
        for _, parameter in selected:
            dist.broadcast(parameter, src=0)

    @torch.no_grad()
    def average_gradients(self, parameters):
        """Average before clipping/Adam; equal local batch sizes give the global mean loss gradient."""
        if not parameters:
            raise ValueError("no trainable parameters")
        self.fail_if(any(p.grad is None for p in parameters), "missing trainable gradient")
        self.fail_if(any(not bool(torch.isfinite(p.grad).all()) for p in parameters),
                     "non-finite local gradient", FloatingPointError)
        if not self.active:
            return
        # LoRA master weights/grads are FP32; fail rather than silently downcast.
        self.fail_if(any(p.grad.dtype != torch.float32 for p in parameters), "LoRA gradients must be FP32")
        flat = torch.cat([p.grad.reshape(-1) for p in parameters])
        # Divide before reduction to reduce the chance of overflow in SUM.
        flat.div_(self.info.world_size)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for p in parameters:
            count = p.numel()
            p.grad.copy_(flat[offset:offset + count].view_as(p))
            offset += count


def aggregate_rank_metrics(rows):
    """Equal-size rank means; preserve local counterfactual provenance explicitly."""
    if not rows:
        raise ValueError("metrics require at least one rank")
    if len(rows) == 1:
        return dict(rows[0])
    result = dict(rank_metrics=rows, metric_reduction="equal_local_batch_rank_mean",
                  relative_dv_reduction="mean_of_per_rank_relative_norms")
    for key in rows[0]:
        if key in ("rank", "step"):
            continue
        values = [row[key] for row in rows]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            result[key] = sum(values) / len(values)
        elif key == "round_errors":
            result[key] = [sum(v[j] for v in values) / len(values) for j in range(len(values[0]))]
    return result
