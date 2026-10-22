"""UND reads ONLY generated hidden tokens; frozen base has one parameter owner."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .runtime import autocast


class LoRA(nn.Module):
    def __init__(self, linear, rank):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank,linear.in_features,dtype=torch.float32,device=linear.weight.device))
        self.b = nn.Parameter(torch.zeros(linear.out_features,rank,dtype=torch.float32,device=linear.weight.device))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))

    def forward(self, hidden):
        return F.linear(F.linear(hidden,self.a.to(hidden.dtype)),self.b.to(hidden.dtype))


class SemanticReader(nn.Module):
    def __init__(self, runtime, config):
        super().__init__(); object.__setattr__(self,'runtime',runtime)
        from sensenova_u1.models.neo_unify.modeling_qwen3 import U15_UND_LORA
        if not U15_UND_LORA: raise ValueError('Apply the U1.5 NPU/reader source patch first')
        d = runtime.decoder.config.hidden_size
        self.config = dict(config)
        self.input = nn.Linear(d,d,bias=False,device=runtime.device,dtype=torch.float32)
        nn.init.eye_(self.input.weight)
        self.state = nn.Sequential(nn.Linear(2,d),nn.SiLU(),nn.Linear(d,d)).to(runtime.device)
        self.adapters = nn.ModuleList(nn.ModuleDict({name:LoRA(getattr(layer.self_attn,name),config['lora_rank'])
            for name in ('q_proj','k_proj','v_proj','o_proj')}) for layer in runtime.decoder.layers)

    def forward(self, observation, t, text_ids=None):
        """Same-layer generated sources; no target text enters the visual stream."""
        rt=self.runtime
        if len(observation)!=len(self.adapters):raise ValueError('GEN observations must cover every UND layer')
        n=observation[-1].shape[1]
        state=torch.tensor([[float(t),rt.noise_scale()]],device=rt.device)
        hidden=self.input(observation[-1].detach().float())+self.state(state)[:,None]
        source_positions=rt.model._build_t2i_image_indexes(*rt.grid,0,device=rt.device)
        query_positions=rt.model._build_t2i_image_indexes(*rt.grid,1,device=rt.device)
        if text_ids is not None:
            text=rt.decoder.embed_tokens(text_ids).detach()
            hidden=torch.cat((hidden,text.float()),1)
            positions=torch.zeros(3,text.shape[1],dtype=torch.long,device=rt.device)
            positions[0]=torch.arange(2,text.shape[1]+2,device=rt.device)
            query_positions=torch.cat((query_positions,positions),1)
        indexes=torch.cat((source_positions,query_positions),1)
        total=n+hidden.shape[1]
        allowed=torch.ones(total,total,dtype=torch.bool,device=rt.device).tril()
        allowed[:n,:n]=True;allowed[:n,n:]=False
        allowed[n:2*n,:2*n]=True;allowed[n:2*n,2*n:]=False
        mask=torch.zeros(total,total,device=rt.device).masked_fill(~allowed,float('-inf'))[None,None]
        dtype=next(rt.decoder.parameters()).dtype;hidden=hidden.to(dtype)
        kwargs=dict(image_gen_indicators=None,exist_non_image_gen_tokens=True,exist_image_gen_tokens=False,
                    indexes=indexes,attention_mask=mask,past_key_values=None,use_cache=False)
        with autocast(rt.device):
            for layer,adapter,source in zip(rt.decoder.layers,self.adapters,observation):
                def block(x, source, layer=layer, adapter=adapter):
                    source=self.input(source.detach().float()).to(x.dtype)
                    combined=torch.cat((source,x),1)
                    return layer(combined,**kwargs,und_lora=adapter)[:,n:]
                hidden=checkpoint(block,hidden,source,use_reentrant=False) if self.config['activation_checkpointing'] and torch.is_grad_enabled() else block(hidden,source)
            hidden=rt.decoder.norm(hidden)
        return hidden[:,:n],hidden[:,n:]

    def caption_loss(self, observation, t, caption):
        tokenizer = self.runtime.tokenizer
        ids = tokenizer.encode('Describe the generated image.\n',add_special_tokens=False)
        answer = tokenizer.encode(caption,add_special_tokens=False)+[tokenizer.eos_token_id]
        all_ids = torch.tensor([ids+answer],device=self.runtime.device)
        _, states = self(observation,t,all_ids[:,:-1])
        weight=self.runtime.model.language_model.lm_head.weight
        logits = F.linear(states.to(weight.dtype),weight).float()
        target = all_ids[:,1:].clone(); target[:,:len(ids)-1] = -100
        return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1))

    @torch.no_grad()
    def describe(self, observation, t, max_tokens=128):
        rt = self.runtime; ids = rt.tokenizer.encode('Describe the generated image.\n',add_special_tokens=False)
        begin = len(ids)
        for _ in range(max_tokens):
            _, hidden = self(observation,t,torch.tensor([ids],device=rt.device))
            token = int(F.linear(hidden[:,-1].to(rt.model.language_model.lm_head.weight.dtype),rt.model.language_model.lm_head.weight).argmax(-1))
            if token == rt.tokenizer.eos_token_id: break
            ids.append(token)
        return rt.tokenizer.decode(ids[begin:],skip_special_tokens=True)
