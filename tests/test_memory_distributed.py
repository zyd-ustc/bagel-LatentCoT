from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from qwen_latent_cot.bagel.memory_distributed import (
    DistributedContext, RankBatchSampler, RankInfo, aggregate_rank_metrics, distributed_contract,
)
from qwen_latent_cot.bagel.memory_stage_runner import checked_step, new_output, train_stage


@pytest.mark.parametrize("shuffle", [False,True])
@pytest.mark.parametrize("size", [16,17,31,115883])
def test_eight_rank_sampler_is_disjoint_reproducible_and_handles_tail(size, shuffle):
    samplers = [RankBatchSampler(size,2,rank=r,world_size=8,seed=42,shuffle=shuffle) for r in range(8)]
    for batch in (0,1,2,(size+15)//16-1,(size+15)//16):
        parts = [sampler.indices(batch) for sampler in samplers]
        flat = sum(parts,[])
        assert len(flat) == len(set(flat)) == 16
        assert all(0<=i<size for i in flat)
        again = RankBatchSampler(size,2,rank=3,world_size=8,seed=42,shuffle=shuffle)
        assert parts[3] == again.indices(batch)


def test_single_rank_sequential_sampling_keeps_previous_contract():
    sampler = RankBatchSampler(5,2)
    assert [sampler.indices(i) for i in range(4)] == [[0,1],[2,3],[4,0],[1,2]]


def test_rank_and_config_validation():
    assert RankInfo.from_env({}) == RankInfo()
    info = RankInfo.from_env(dict(WORLD_SIZE="8",RANK="3",LOCAL_RANK="3"))
    c = distributed_contract(dict(batch_size=2,device="auto"),"reader",32,info=info)
    assert c["global_batch_size"] == 16 and not c["learning_rate_scaled"]
    for bad in ({"WORLD_SIZE":"0"},{"WORLD_SIZE":"8"},
                {"WORLD_SIZE":"2","RANK":"2","LOCAL_RANK":"0"}):
        with pytest.raises(ValueError): RankInfo.from_env(bad)
    for stage in ("grpo","eval"):
        with pytest.raises(ValueError,match="not GRPO or eval"):
            distributed_contract(dict(batch_size=2),stage,32,info=info)
    with pytest.raises(ValueError,match="at least"):
        distributed_contract(dict(batch_size=2),"reader",15,info=info)
    with pytest.raises(ValueError,match="LOCAL_RANK"):
        distributed_contract(dict(batch_size=2,device="cuda:0"),"reader",16,info=info)


def test_metric_means_keep_local_donor_provenance():
    rows = [dict(rank=r,loss=float(r),sample_ids=[str(r)],donors=[1,0],round_errors=[float(r),2.]) for r in range(2)]
    result = aggregate_rank_metrics(rows)
    assert result["loss"] == .5 and result["round_errors"] == [.5,2.]
    assert "rank" not in result and result["rank_metrics"] == rows


def _tiny_mot_loss(model, rank, step):
    from test_memory_grounding import flow_kwargs
    torch.manual_seed(1000+rank+step*100)
    kwargs = flow_kwargs()
    with torch.no_grad():
        memory = model.forward_memory_read(**{k:v for k,v in kwargs.items() if k!="loop_memory"}).memory_read
    out = model.forward_loop_supervised(**kwargs,num_write_rounds=1,
        write_memory_override=memory,mask_prompt_kv_during_write=True)
    return (out.final_velocity.float()-1.).square().mean()


class _ToyTrainRuntime:
    stage = "reader"
    def __init__(self, rank):
        torch.manual_seed(80+rank)
        self.model = nn.Linear(1,1)
        self.rank = rank
    def states(self, record, seed, state_seed):
        return [SimpleNamespace(record=record)]
    def dependency(self, items, **kwargs):
        x = torch.tensor([[float(item.record["id"])] for item in items])
        loss = (self.model(x)-x*.5).square().mean()
        return loss, dict(loss=float(loss.detach()),sample_ids=[it.record["id"] for it in items],donors=[1,0])
    def save(self, output, step, optimizer):
        with (output/"saved_by_rank.jsonl").open("a") as f:
            f.write(json.dumps(dict(rank=self.rank,step=step))+"\n")
        torch.save(self.model.state_dict(),output/f"step{step}.pt")


def _multiprocess_worker(rank, store, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=f"file://{store}",rank=rank,world_size=2,
                            timeout=timedelta(seconds=90))
    context = DistributedContext(RankInfo(rank,rank,2),device="cpu")
    try:
        from test_memory_grounding import tiny_bagel
        from qwen_latent_cot.bagel.loop import configure_loop_trainable_routes
        model = tiny_bagel()
        configure_loop_trainable_routes(model,["q_proj_moe_gen"])
        reference = tiny_bagel()
        configure_loop_trainable_routes(reference,["q_proj_moe_gen"])
        # Prove initialization synchronization, not merely identical random seeds.
        if rank == 1:
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad: p.add_(.25)
        context.synchronize_initial_adapters(model)
        params = [p for p in model.parameters() if p.requires_grad]
        ref_params = [p for p in reference.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params,lr=5e-4,betas=(.9,.95),weight_decay=0.)
        ref_opt = torch.optim.AdamW(ref_params,lr=5e-4,betas=(.9,.95),weight_decay=0.)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            checked_step(SimpleNamespace(model=model),opt,_tiny_mot_loss(model,rank,step),{},context)
            ref_opt.zero_grad(set_to_none=True)
            reference_loss = sum(_tiny_mot_loss(reference,r,step) for r in range(2))/2
            reference_loss.backward()
            torch.nn.utils.clip_grad_norm_(ref_params,1.,error_if_nonfinite=True)
            ref_opt.step()
            for actual, expected in zip(params,ref_params):
                torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-6)
            local = torch.cat([p.detach().reshape(-1) for p in params])
            replicas = [torch.empty_like(local) for _ in range(2)]
            dist.all_gather(replicas,local)
            assert torch.equal(replicas[0],replicas[1])

        # Exercise the actual shared train loop and rank-zero artifact path.
        records = [dict(id=str(i),prompt=f"p{i}") for i in range(8)]
        config = dict(data_path=str(Path(output)/"data.jsonl"),output_dir=str(Path(output)/"run"),
            batch_size=2,global_batch_size=4,world_size=2,seed=42,shuffle_data=True,
            max_steps=2,save_steps=1,states_per_prompt=1,objective="native_teacher")
        context.verify_inputs(config,records)
        run_dir = Path(context.primary_call(lambda:str(new_output(config))))
        runtime = _ToyTrainRuntime(rank)
        context.synchronize_initial_adapters(runtime.model)
        train_stage(runtime,records,run_dir,config,context)
        checkpoints = [json.loads(l) for l in (run_dir/"saved_by_rank.jsonl").read_text().splitlines()]
        assert checkpoints == [dict(rank=0,step=1),dict(rank=0,step=2)]
        rows = [json.loads(l) for l in (run_dir/"metrics.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and all(r["world_size"]==2 for r in rows)
        for row in rows:
            ids = sum([m["sample_ids"] for m in row["rank_metrics"]],[])
            assert len(set(ids)) == 4
        assert json.loads((run_dir/"status.json").read_text())["status"]=="complete"
        # Every rank receives failures rather than waiting in a mismatched collective.
        with pytest.raises(RuntimeError,match="rank 1: injected"):
            context.fail_if(rank==1,"injected")
        def broken_save():
            raise OSError("injected disk failure")
        with pytest.raises(RuntimeError,match="injected disk failure"):
            context.primary_call(broken_save)
        params[0].grad = torch.full_like(params[0],float("nan") if rank==1 else 1.)
        with pytest.raises(FloatingPointError,match="non-finite local gradient"):
            context.average_gradients(params)
        (Path(output)/f"rank{rank}.json").write_text(json.dumps(dict(
            equivalence_passed=True,checkpoint_writer=0,failure_propagation=True)))
    finally:
        context.close()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),reason="Gloo unavailable")
def test_real_two_process_tiny_mot_equivalence_and_training_artifacts(tmp_path):
    (tmp_path/"data.jsonl").write_text('{"prompt":"fixture"}\n')
    mp.spawn(_multiprocess_worker,args=(str(tmp_path/"gloo-store"),str(tmp_path)),nprocs=2,join=True)
    assert all(json.loads((tmp_path/f"rank{r}.json").read_text())["equivalence_passed"] for r in range(2))
