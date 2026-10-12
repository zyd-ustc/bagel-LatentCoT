"""Check gradient scaling and synchronization against one global-batch optimizer."""
import os
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from qwen_latent_cot.diagnostics import distributed as parallel
from qwen_latent_cot.diagnostics.record import FrozenGuard


def worker(rank, port):
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank),
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    parallel.initialize('cpu')
    try:
        torch.manual_seed(21+rank)
        writer = torch.nn.Linear(3, 2)
        parallel.broadcast_parameters(writer)
        reference = torch.nn.Linear(3, 2)
        reference.load_state_dict(writer.state_dict())
        frozen = torch.nn.Linear(2, 2).requires_grad_(False)
        guard = FrozenGuard(frozen)
        optimizer = torch.optim.AdamW(writer.parameters(), lr=0.01)
        expected = torch.optim.AdamW(reference.parameters(), lr=0.01)
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3)/10
        target = torch.zeros(4, 2)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            expected.zero_grad(set_to_none=True)
            for index in range(rank, 4, 2):
                (torch.nn.functional.mse_loss(writer(x[index]), target[index])/4).backward()
            torch.nn.functional.mse_loss(reference(x), target).backward()
            parallel.sum_gradients(writer)
            for actual, wanted in zip(writer.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, wanted.grad)
            torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
            optimizer.step()
            expected.step()
            for actual, wanted in zip(writer.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, wanted)
            guard.check()
        assert parallel.sum_statistics([rank+1, 2], 'cpu') == [3., 4.]
        gathered = parallel.gather_samples([{'rank': rank}])
        assert gathered == [{'rank': 0}, {'rank': 1}]
    finally:
        dist.destroy_process_group()


def test_joint_optimizer_matches_global_batch():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=2, join=True)


def visual_worker(rank, port):
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank),
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    parallel.initialize('cpu')
    try:
        module = torch.nn.ModuleDict({'visual':torch.nn.Linear(3, 2, bias=False),
                                      'writer':torch.nn.Linear(3, 2, bias=False).requires_grad_(False)})
        parallel.broadcast_parameters(module)
        frozen = module['writer'].weight.clone()
        # Rank one owns a frozen text-only replay sample: no visual graph exists.
        if rank == 0: (module['visual'](torch.ones(3)).sum()/2).backward()
        parallel.sum_gradients(module, allow_unused=True)
        torch.testing.assert_close(module['visual'].weight.grad, torch.full((2,3), .5))
        assert module['writer'].weight.grad is None
        torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=.01).step()
        torch.testing.assert_close(module['writer'].weight, frozen, rtol=0, atol=0)
        weights = [None,None];dist.all_gather_object(weights,module['visual'].weight.detach().tolist())
        assert weights[0] == weights[1]
    finally: dist.destroy_process_group()


def test_frozen_text_replay_rank_participates_in_joint_visual_optimizer():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0)); port=listener.getsockname()[1]
    mp.spawn(visual_worker,args=(port,),nprocs=2,join=True)
