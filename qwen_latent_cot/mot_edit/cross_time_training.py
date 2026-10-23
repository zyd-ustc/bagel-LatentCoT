"""Actual rollouts; sample one write step per replay round for bounded-memory gradients."""
import torch
from torch.nn import functional as F
from .cross_time import detach_bank,native_segment
from .model import project_safe
from ..correction.sampling import times


def edit_pair(model,x0,flow,seed,write_index,batch):
    cfg=model.config;grid=times(cfg);early=cfg['rollback_index'];read=cfg['read_index']
    epsilon=torch.randn(x0.shape,generator=torch.Generator().manual_seed(seed)).to(x0.device)
    # Training entrance is a paired early-time flow state; every later state is a real native/editor rollout.
    anchor=(1-grid[early])*x0.float()+grid[early]*epsilon
    late=native_segment(model,anchor.clone(),flow,early,read);previous=None
    stats=[0.]*7
    for r in range(cfg['rounds']):
        plan=model.plan(late,grid[read],flow,previous)
        frozen=detach_bank(plan);state=anchor.clone()
        for index in range(early,read):
            t=grid[index];dt=t-grid[index+1]
            if index==write_index:
                vg,w,vc,delta=model.velocity(state.detach(),t,flow,plan)
                truth=(state.detach().float()-x0.float())/t;g=vg.float()-vc.float()
                target_delta=project_safe(truth-vc.float(),g)
                loss=F.mse_loss(w,vc.float()+target_delta)
                (loss/(cfg['rounds']*batch)).backward()
                stats[0]+=float(loss.detach())/cfg['rounds'];stats[1]+=float(target_delta.square().mean())/cfg['rounds']
                stats[2+r]+=float(loss.detach())
                coefficient=float((delta.detach()*g).sum()/g.square().sum().clamp_min(1e-12))
                stats[4]+=float(coefficient < -1e-6)/cfg['rounds']
                stats[5]+=float((vg.float()+delta.detach()-truth).square().mean())/cfg['rounds']
                stats[6]+=float(delta.detach().square().mean().sqrt())/cfg['rounds']
                state=state-dt*vg-dt*delta.detach()
                del w,vc,delta,truth,g,target_delta,loss,vg
            else:
                with torch.no_grad():
                    vg,_,_,delta=model.velocity(state,t,flow,frozen)
                    state=state-dt*vg-dt*delta
        late=state.detach();previous=frozen
        del plan
    return stats
