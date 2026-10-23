"""No new output heads: UND native prefix state, GEN native llm2vae prediction."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .roles import Role,KV,native_und_forward
from ..correction.attention import project
from ..correction.numerics import autocast_for


def project_safe(delta,g):
    # Caller handles one image, shape [patches,channels]. No cross-sample reduction.
    g=g.detach().float();d=delta.float()
    a=(d*g).sum()/g.square().sum().clamp_min(1e-12)
    return d-a.clamp_max(0)*g


def apply_edit(x,t,vc,vg,w):
    delta=project_safe(w.float()-vc.float(),vg.float()-vc.float())
    # clean=x-t*vg, noise=x+(1-t)*vg; edit clean by -t*delta.
    # Re-noise the edited clean estimate with the SAME estimated noise:
    # (1-t)*(clean-t*delta)+t*noise = x-t*(1-t)*delta.
    # Keep the state graph: later-round losses can train the earlier edit.
    return x.float()-t*(1-t)*delta,delta


class MoTEdit(nn.Module):
    def __init__(self,runtime,config):
        super().__init__();object.__setattr__(self,'runtime',runtime);self.config=dict(config)
        d=runtime.decoder;rank=config['lora_rank']
        self.observation=Role(d,rank);self.feedback=Role(d,rank);self.editor=Role(d,rank,generation=True)
        self.stage=None;self.set_stage('observation')
        ids=runtime.bundle.tokenizer.encode('Observe the image carefully and describe the objects, attributes and relations.',add_special_tokens=False)
        ids=(ids*((config['slots']+len(ids)-1)//len(ids)))[:config['slots']]
        self.register_buffer('slot_ids',torch.tensor(ids,device=runtime.device),persistent=False)
        self.neutral_flow=runtime.flow('An image.')
    def set_stage(self,stage):
        if stage not in ('observation','edit'):raise ValueError(stage)
        self.stage=stage
        self.observation.set_active(True);self.feedback.set_active(stage=='edit');self.editor.set_active(stage=='edit')
    def all_adapters(self):
        return {f'{role}.{n}':p for role in ('observation','feedback','editor')
                for n,p in getattr(self,role).adapters().items()}
    def trainable(self):return {n:p for n,p in self.all_adapters().items() if p.requires_grad}
    @torch.no_grad()
    def observe_image(self,proposal):
        rt=self.runtime;source={};handles=[];flow=self.neutral_flow
        def capture(index):
            def hook(module,args,kwargs):
                if kwargs.get('past_key_values') is not flow['past_key_values']:return
                indexes=kwargs['packed_vae_token_indexes'];normalized=kwargs['packed_query_sequence'][indexes]
                _,k,v=project(module,normalized,generation=True)
                pos=flow['packed_position_ids'][indexes]
                source[index]=KV(k.detach(),v.detach(),pos)
            return hook
        try:
            for i,layer in enumerate(rt.decoder.layers):handles.append(layer.self_attn.register_forward_pre_hook(capture(i),with_kwargs=True))
            rt.predict(proposal.detach(),0.,flow,capture=False)
        finally:
            for h in handles:h.remove()
        if len(source)!=len(rt.decoder.layers):raise RuntimeError('Incomplete clean-image source')
        return tuple(source[i] for i in range(len(source)))
    def slots(self):return self.runtime.decoder.embed_tokens(self.slot_ids)
    def observation_bank(self,proposal):
        source=self.observe_image(proposal);position=int(source[0].positions.max())+1
        pos=torch.arange(position,position+self.config['slots'],device=self.runtime.device)
        with autocast_for(self.runtime.device):
            return self.observation(self.slots(),pos,(source,),checkpointing=self.config['activation_checkpointing'],export=True)[1]
    def feedback_bank(self,observed,flow):
        # Export only feedback slots. Goal text cache remains private to UND.
        cache=flow['past_key_values'];length=len(cache.key_cache[0])
        goal=[]
        # Native text keys are already rotated; recover original projected K with inverse RoPE.
        from ..correction.attention import rope_for,rotate
        positions=torch.arange(length,device=self.runtime.device)
        for i in range(len(observed)):
            k=cache.key_cache[i].float();cos,sin=rope_for(self.runtime.decoder,k,positions)
            goal.append(KV(rotate(k,(cos,-sin)).to(k.dtype),cache.value_cache[i],positions))
        op=torch.arange(length,length+self.config['slots'],device=self.runtime.device)
        relocated=tuple(KV(kv.keys,kv.values,op) for kv in observed)
        fp=torch.arange(length+self.config['slots'],length+2*self.config['slots'],device=self.runtime.device)
        with autocast_for(self.runtime.device):
            return self.feedback(self.slots(),fp,(tuple(goal),relocated),checkpointing=self.config['activation_checkpointing'],export=True)[1]
    def edit_velocity(self,state,t,feedback):
        rt=self.runtime;m=rt.model
        # Private editor sequence contains feedback prefix and image segment, never the goal cache.
        pos=torch.arange(self.config['slots'],device=rt.device)
        relocated=tuple(KV(kv.keys,kv.values,pos[:len(kv.keys)]) for kv in feedback)
        # Preserve native image boundary tokens and their UND expert routing.
        # The editor has no original prompt cache; its semantic prefix is feedback only.
        flow=self.neutral_flow;image_indexes=flow['packed_vae_token_indexes'];text_indexes=flow['packed_text_indexes']
        positions=torch.full((len(state)+len(text_indexes),),len(feedback[0].keys),device=rt.device,dtype=torch.long)
        with autocast_for(rt.device):
            image=m.vae2llm(state.to(m.vae2llm.weight.dtype))
            image=image+m.time_embedder(torch.full((len(image),),t,device=rt.device))
            image=image+m.latent_pos_embed(flow['packed_vae_position_ids'])
            text=rt.decoder.embed_tokens(flow['packed_text_ids'])
            hidden=text.new_zeros((len(positions),text.shape[-1]))
            hidden[text_indexes]=text;hidden[image_indexes]=image.to(text.dtype)
            hidden,_=self.editor(hidden,positions,(relocated,),checkpointing=self.config['activation_checkpointing'],text_indexes=text_indexes)
            return m.llm2vae(rt.decoder.norm_moe_gen(hidden[image_indexes])).float()
    def round(self,state,t,flow):
        # Read a clean estimate, but predict velocity from the actual noisy state.
        vg,_,vc=self.runtime.proposal(state.detach(),t,flow,capture=False,with_conditional=True)
        proposal=(state.detach().float()-t*vg.float()).detach()
        observed=self.observation_bank(proposal)
        feedback=self.feedback_bank(observed,flow)
        w=self.edit_velocity(state,t,feedback)
        edited,delta=apply_edit(state,t,vc,vg,w)
        return edited,w,vc,vg,delta
    def caption_loss(self,latent,text):
        bank=self.observation_bank(latent)
        ids=self.runtime.bundle.tokenizer.encode(text,add_special_tokens=False)
        ids=[self.runtime.bundle.token_ids['bos_token_id']]+ids+[self.runtime.bundle.token_ids['eos_token_id']]
        tokens=torch.tensor(ids,device=self.runtime.device,dtype=torch.long)
        n=self.config['slots'];positions=torch.arange(n,n+len(tokens)-1,device=tokens.device)
        bank=tuple(KV(kv.keys,kv.values,torch.arange(n,device=tokens.device)) for kv in bank)
        with autocast_for(self.runtime.device):
            hidden=self.runtime.decoder.embed_tokens(tokens[:-1])
            hidden=native_und_forward(self.runtime.decoder,hidden,positions,bank,self.config['activation_checkpointing'])
            losses=[]
            for start in range(0,len(hidden),64):
                h=hidden[start:start+64];target=tokens[1+start:1+start+len(h)]
                def ce(value,target=target):
                    return F.cross_entropy(self.runtime.model.language_model.lm_head(value).float(),target,reduction='sum')
                losses.append(checkpoint(ce,h,use_reentrant=False) if torch.is_grad_enabled() else ce(h))
            return torch.stack(losses).sum()/(len(tokens)-1)

    @torch.no_grad()
    def caption_text(self,proposal,max_tokens=64):
        bank=self.observation_bank(proposal);n=self.config['slots'];device=self.runtime.device
        bank=tuple(KV(kv.keys,kv.values,torch.arange(n,device=device)) for kv in bank)
        ids=[self.runtime.bundle.token_ids['bos_token_id']]
        for _ in range(max_tokens):
            tokens=torch.tensor(ids,device=device)
            with autocast_for(device):
                hidden=self.runtime.decoder.embed_tokens(tokens)
                hidden=native_und_forward(self.runtime.decoder,hidden,torch.arange(n,n+len(ids),device=device),bank,False)
                token=int(self.runtime.model.language_model.lm_head(hidden[-1:]).float().argmax(-1))
            if token==self.runtime.bundle.token_ids['eos_token_id']:break
            ids.append(token)
        return self.runtime.bundle.tokenizer.decode(ids[1:],skip_special_tokens=True)
