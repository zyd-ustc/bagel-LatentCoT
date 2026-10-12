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


def visual_training_worker(rank, port, root):
    from pathlib import Path
    import json
    from PIL import Image
    from test_memory_topology import lm as lm_fixture, runtime as runtime_fixture
    from qwen_latent_cot.diagnostics.memory import MemoryWriter
    from qwen_latent_cot.diagnostics.visual import VisualStudent, train
    from qwen_latent_cot.diagnostics.data import visual_label
    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank),
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    parallel.initialize('cpu')
    try:
        workspace = Path(root)/f'worker_{rank}'; workspace.mkdir()
        runtime = runtime_fixture.__wrapped__(lm_fixture.__wrapped__(), workspace)
        runtime.writer = MemoryWriter(runtime.decoder,'C',16,4)
        cfg=json.loads(Path('configs/visual_memory_pilot.json').read_text())
        cfg.update({key:runtime.config[key] for key in ('resolution','rank','caption_instruction')})
        cfg.update(visual_rank=4,response_layers=[0,2],response_query_count=3,steps=1,
                   effective_batch=4,eval_every=1,max_caption_tokens=2,eval_t=[.75,1.],noise_seeds=[0])
        runtime.config=cfg
        rows=[]
        for split in ('train','val'):
            for index,color in enumerate(('red','blue')):
                path=workspace/f'{split}_{index}.png';Image.new('RGB',(32,32),color).save(path)
                rows.append(visual_label(dict(sample_id=f'{split}_{index}',scene_id=f'{split}_{index}',
                    split=split,caption='a token',facts=[],verified=True,image=str(path))))
        text_rows=[dict(row) for row in rows]
        student=VisualStudent(runtime);student.text_teacher_hash='test'
        output=Path(root)/'training'
        if rank==0:output.mkdir()
        parallel.barrier()
        train(student,rows,text_rows,'C2-observe',output,None)
        parallel.barrier()
        if rank==0:
            index=json.loads((output/'validation_000001/metrics_index.json').read_text())
            assert {r['rank'] for r in index}=={0,1}
            assert len(index)==4
            log=json.loads((output/'training.jsonl').read_text().splitlines()[0])
            assert log['world_size']==2 and log['actual_gen_entries_including_recomputation']>=4
            assert log['und_layer_calls']>0
            assert (output/'last.pt').exists()
    finally:dist.destroy_process_group()


def test_visual_training_distributes_validation_families(tmp_path):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    mp.spawn(visual_training_worker,args=(port,str(tmp_path)),nprocs=2,join=True)
