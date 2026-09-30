"""Two-rank CPU/gloo checks for exact averaged gradients and synchronized heads."""
from datetime import timedelta
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from qwen_latent_cot.bagel.reader_warmup import (
    average_reader_gradients,synchronize_reader_parameters)


def gradient_worker(rank, rendezvous, output):
    dist.init_process_group("gloo",init_method=rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        param=torch.nn.Parameter(torch.tensor([1.+rank,3.+rank]))
        synchronize_reader_parameters([param])
        assert torch.equal(param.detach(),torch.tensor([1.,3.]))
        optimizer=torch.optim.SGD([param],lr=.1)
        (param.square().sum()*(rank+1)).backward()
        average_reader_gradients([param])
        assert torch.allclose(param.grad,torch.tensor([3.,9.]))
        optimizer.step()
        assert torch.allclose(param,torch.tensor([.7,2.1]))
        param.grad=torch.tensor([float("nan") if rank else 0.,0.])
        try:
            average_reader_gradients([param])
        except FloatingPointError:
            pass
        else:
            raise AssertionError("nonfinite rank must fail all ranks")
        with open(output+str(rank)+".json","w") as target:
            json.dump(param.detach().tolist(),target)
    finally:
        dist.destroy_process_group()


def test_two_rank_average_matches_global_batch_and_rejects_nonfinite(tmp_path):
    mp.spawn(gradient_worker,args=("file://"+str(tmp_path/"rendezvous"),str(tmp_path/"rank")),
             nprocs=2,join=True)
    assert (tmp_path/"rank0.json").read_text()==(tmp_path/"rank1.json").read_text()


def runner_worker(rank,rendezvous,config,train,heldout):
    from test_reader_warmup_runner import TinyWarmupRuntime
    from qwen_latent_cot.bagel.reader_warmup import ReaderWarmupRuntime,train_warmup
    dist.init_process_group("gloo",init_method=rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        torch.manual_seed(42)
        runtime=TinyWarmupRuntime(config)
        ReaderWarmupRuntime.load_model=classmethod(lambda cls,config:runtime)
        train_warmup(config,train,heldout)
        params=torch.cat([p.detach().reshape(-1) for p in runtime.model.parameters()
                          if p.requires_grad])
        other=params.clone()
        dist.broadcast(other,src=0)
        assert torch.equal(params,other)
    finally:
        dist.destroy_process_group()


def test_two_rank_tiny_runner_writes_one_global_checkpoint_and_disjoint_samples(tmp_path):
    train=[dict(prompt_id=f"t{i}",prompt=f"{i+2} cubes",category="count",split="train")
           for i in range(4)]
    heldout=[dict(prompt_id=f"h{i}",prompt=f"{i+8} spheres",category="count",split="heldout")
             for i in range(2)]
    source=tmp_path/"train.jsonl"; validation=tmp_path/"heldout.jsonl"
    for path,rows in ((source,train),(validation,heldout)):
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    config=dict(model_path=str(tmp_path),prompt_data=str(source),heldout_prompt_data=str(validation),
        output_dir=str(tmp_path/"run"),height=32,width=32,num_steps=4,timestep_shift=3.,
        seed=42,states_per_rollout=2,num_loop_tokens=2,memory_loop_start_layer=1,
        memory_loop_end_layer=3,o_adapter_rank=2,o_adapter_alpha=2,
        learning_rate=1e-4,max_grad_norm=1.,max_steps=2,save_steps=1,eval_steps=1,
        eval_max_prompts=2)
    mp.spawn(runner_worker,args=("file://"+str(tmp_path/"rendezvous"),config,train,heldout),
             nprocs=2,join=True)
    from pathlib import Path
    output=Path(config["output_dir"])
    rows=[json.loads(line) for line in (output/"metrics.jsonl").read_text().splitlines()]
    assert len(rows)==2 and rows[0]["prompt_ids"]==["t0","t1"]
    assert rows[1]["prompt_ids"]==["t2","t3"]
    assert rows[0]["loss_reader_mse"]==sum(r["loss_reader_mse"]
        for r in rows[0]["per_rank_metrics"])/2
    assert json.loads((output/"run_manifest.json").read_text())["effective_batch_size"]==2
    assert json.loads((output/"status.json").read_text())["status"]=="complete"
