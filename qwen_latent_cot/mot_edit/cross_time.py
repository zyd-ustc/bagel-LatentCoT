"""Late UND reading, saved-state rollback and a persistent native GEN prefix."""
import torch
from .model import MoTEdit,project_safe
from .roles import KV
from ..correction.attention import rope_for,rotate
from ..correction.numerics import autocast_for
from ..correction.sampling import times


def detach_bank(bank):
    return tuple(KV(x.keys.detach(),x.values.detach(),x.positions) for x in bank)


class CrossTimeEdit(MoTEdit):
    checkpoint_kind='bagel_mot_cross_time_v3'
    def __init__(self,runtime,config):
        super().__init__(runtime,config)
        early,read=config['rollback_index'],config['read_index']
        if not 0<=early<read<config['sampler_steps']-1:raise ValueError('Invalid rollback/read interval')
        instruction='Describe the image briefly. State objects, colors, counts, and spatial relations.\n'
        self.caption_instruction=runtime.bundle.tokenizer.encode(instruction,add_special_tokens=False)

    def text_bank(self,flow):
        cache=flow['past_key_values'];n=len(cache.key_cache[0])
        positions=torch.arange(n,device=self.runtime.device);result=[]
        for i in range(len(self.runtime.decoder.layers)):
            k,v=cache.key_cache[i],cache.value_cache[i]
            k=k.float();cos,sin=rope_for(self.runtime.decoder,k,positions)
            result.append(KV(rotate(k,(cos,-sin)).to(v.dtype),v,positions))
        return tuple(result)

    def feedback_bank(self,observed,flow,previous=None):
        goal=self.text_bank(flow);offset=len(goal[0].keys)
        def relocate(bank,start):
            return tuple(KV(x.keys,x.values,torch.arange(start,start+len(x.keys),device=x.keys.device)) for x in bank)
        sources=[goal,relocate(observed,offset)];offset+=len(observed[0].keys)
        if previous is not None:
            sources.append(relocate(previous,offset));offset+=len(previous[0].keys)
        positions=torch.arange(offset,offset+self.config['slots'],device=self.runtime.device)
        with autocast_for(self.runtime.device):
            return self.feedback(self.slots(),positions,tuple(sources),checkpointing=self.config['activation_checkpointing'],export=True)[1]

    def plan(self,state,t,flow,previous=None):
        vg,_,_=self.runtime.proposal(state.detach(),t,flow,capture=False,with_conditional=True)
        observed=self.observation_bank((state.detach().float()-t*vg.float()).detach())
        return self.feedback_bank(observed,flow,previous)

    def edit_velocity(self,state,t,feedback,flow=None):
        if flow is None:raise ValueError('Cross-time editor requires the original prompt')
        rt=self.runtime;m=rt.model;goal=self.text_bank(flow);n=len(goal[0].keys)
        prefix=tuple(KV(torch.cat((g.keys,f.keys)),torch.cat((g.values,f.values)),
            torch.arange(n+len(f.keys),device=rt.device)) for g,f in zip(goal,feedback))
        image_indexes=flow['packed_vae_token_indexes'];text_indexes=flow['packed_text_indexes']
        positions=torch.full((len(state)+len(text_indexes),),n+len(feedback[0].keys),device=rt.device,dtype=torch.long)
        with autocast_for(rt.device):
            image=m.vae2llm(state.to(m.vae2llm.weight.dtype))
            image=image+m.time_embedder(torch.full((len(image),),t,device=rt.device))+m.latent_pos_embed(flow['packed_vae_position_ids'])
            hidden=rt.decoder.embed_tokens(flow['packed_text_ids']).new_zeros((len(positions),rt.decoder.config.hidden_size))
            hidden[text_indexes]=rt.decoder.embed_tokens(flow['packed_text_ids']);hidden[image_indexes]=image.to(hidden.dtype)
            hidden,_=self.editor(hidden,positions,(prefix,),checkpointing=self.config['activation_checkpointing'],text_indexes=text_indexes)
            return m.llm2vae(rt.decoder.norm_moe_gen(hidden[image_indexes])).float()

    def velocity(self,state,t,flow,plan):
        vg,_,vc=self.runtime.proposal(state.detach(),t,flow,capture=False,with_conditional=True)
        w=self.edit_velocity(state,t,plan,flow)
        delta=project_safe(w-vc.float(),vg.float()-vc.float())
        return vg,w,vc,delta

    def caption_hidden(self,tokens,bank):
        n=self.config['slots'];device=self.runtime.device
        prefix=tuple(KV(x.keys,x.values,torch.arange(n,device=device)) for x in bank)
        with autocast_for(device):
            hidden=self.runtime.decoder.embed_tokens(tokens)
            hidden,_=self.observation(hidden,torch.arange(n,n+len(tokens),device=device),(prefix,),
                causal=True,checkpointing=self.config['activation_checkpointing'])
            return self.runtime.decoder.norm(hidden)

    def caption_loss(self,latent,text):
        from torch.nn import functional as F
        from torch.utils.checkpoint import checkpoint
        bank=self.observation_bank(latent);rt=self.runtime
        instruction=[rt.bundle.token_ids['bos_token_id']]+self.caption_instruction
        answer=rt.bundle.tokenizer.encode(text,add_special_tokens=False)+[rt.bundle.token_ids['eos_token_id']]
        tokens=torch.tensor(instruction+answer,device=rt.device)
        hidden=self.caption_hidden(tokens[:-1],bank)[len(instruction)-1:]
        targets=tokens[len(instruction):];losses=[]
        for start in range(0,len(hidden),64):
            target=targets[start:start+64]
            def ce(h,target=target):
                return F.cross_entropy(rt.model.language_model.lm_head(h).float(),target,reduction='sum')
            h=hidden[start:start+64]
            losses.append(checkpoint(ce,h,use_reentrant=False) if torch.is_grad_enabled() else ce(h))
        return torch.stack(losses).sum()/len(targets)

    @torch.no_grad()
    def caption_text(self,proposal,max_tokens=64):
        bank=self.observation_bank(proposal);rt=self.runtime
        ids=[rt.bundle.token_ids['bos_token_id']]+self.caption_instruction;start=len(ids)
        for _ in range(max_tokens):
            h=self.caption_hidden(torch.tensor(ids,device=rt.device),bank)
            with autocast_for(rt.device):token=int(rt.model.language_model.lm_head(h[-1:]).float().argmax(-1))
            if token==rt.bundle.token_ids['eos_token_id']:break
            ids.append(token)
        return rt.bundle.tokenizer.decode(ids[start:],skip_special_tokens=True)


@torch.no_grad()
def native_segment(model,state,flow,start,end):
    grid=times(model.config)
    for i in range(start,end):
        vg,_,_=model.runtime.proposal(state,grid[i],flow,capture=False,with_conditional=True)
        state=state-(grid[i]-grid[i+1])*vg
    return state


@torch.no_grad()
def sample(model,prompt,seed,rounds=2):
    rt=model.runtime;flow=rt.flow(prompt);n=len(flow['packed_vae_token_indexes'])
    channels=rt.model.latent_channel*rt.model.latent_patch_size**2
    state=torch.randn((n,channels),generator=torch.Generator().manual_seed(seed)).to(rt.device)
    grid=times(model.config);early=model.config['rollback_index'];read=model.config['read_index'];trace=[]
    anchor=native_segment(model,state,flow,0,early)
    late=native_segment(model,anchor.clone(),flow,early,read);plan=None
    for r in range(rounds):
        plan=model.plan(late,grid[read],flow,plan);state=anchor.clone()
        trace.append(dict(event='rollback',round=r+1,read_index=read,read_t=grid[read],write_index=early,write_t=grid[early],anchor_error_rms=float((state-anchor).float().square().mean().sqrt())))
        for i in range(early,read):
            vg,w,vc,delta=model.velocity(state,grid[i],flow,plan);dt=grid[i]-grid[i+1]
            raw=w-vc.float();g=vg.float()-vc.float()
            state=state-dt*vg-dt*delta
            trace.append(dict(event='feedback_step',index=i,t=grid[i],round=r+1,delta_rms=float(delta.square().mean().sqrt()),
                raw_delta_rms=float(raw.square().mean().sqrt()),delta_to_base_rms=float(delta.square().mean().sqrt()/vg.float().square().mean().sqrt().clamp_min(1e-12)),
                clean_edit_shift_rms=float(grid[i]*delta.square().mean().sqrt()),committed_shift_rms=float(dt*delta.square().mean().sqrt()),
                applied_guidance_coefficient=float((delta*g).sum()/g.square().sum().clamp_min(1e-12))))
            if not torch.isfinite(state).all():raise FloatingPointError('Nonfinite cross-time state')
        late=state
    result=native_segment(model,late,flow,read,len(grid)-1)
    trace.append(dict(event='complete',euler_updates=len(grid)-1+rounds*(read-early),rounds=rounds))
    return rt.image(result),trace
