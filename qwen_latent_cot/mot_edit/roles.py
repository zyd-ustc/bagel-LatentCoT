"""Role-local adapters over immutable native expert weights, never global swapping."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..correction.attention import rope_for,rotate
from ..correction.model import LoRALinear
from ..correction.numerics import autocast_for


@dataclass(frozen=True)
class KV:
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor


def attention(q,k,v,causal=False):
    groups=q.shape[1]//k.shape[1]
    qi=q.transpose(0,1)[None]
    ki=k.transpose(0,1)[None].repeat_interleave(groups,1)
    vi=v.transpose(0,1)[None].repeat_interleave(groups,1)
    mask=None
    if causal:
        nq,nk=len(q),len(k)
        mask=torch.arange(nk,device=q.device)[None]<=torch.arange(nq,device=q.device)[:,None]+nk-nq
    if q.device.type=='cpu':qi,ki,vi=qi.float(),ki.float(),vi.float()
    with torch.autocast('cpu',enabled=False) if q.device.type=='cpu' else autocast_for(q.device):
        out=F.scaled_dot_product_attention(qi,ki,vi,attn_mask=mask,dropout_p=0.)
    return out[0].transpose(0,1).reshape(len(q),-1).to(v.dtype)


class Block(nn.Module):
    def __init__(self,native,rank,generation=False):
        super().__init__();object.__setattr__(self,'native',native)
        self.generation=generation;suffix='_moe_gen' if generation else ''
        self.projections=nn.ModuleDict({name:LoRALinear(getattr(native.self_attn,name+suffix),rank)
                                       for name in ('q_proj','k_proj','v_proj','o_proj')})
    def parts(self,hidden):
        layer=self.native;a=layer.self_attn;s='_moe_gen' if self.generation else ''
        h=getattr(layer,'input_layernorm'+s)(hidden)
        q=self.projections['q_proj'](h).view(-1,a.num_heads,a.head_dim)
        k=self.projections['k_proj'](h).view(-1,a.num_key_value_heads,a.head_dim)
        v=self.projections['v_proj'](h).view_as(k)
        return getattr(a,'q_norm'+s)(q.float()),getattr(a,'k_norm'+s)(k.float()),v
    def forward(self,hidden,decoder,positions,sources=(),causal=False):
        q,k,v=self.parts(hidden);bank=KV(k,v,positions)
        q=rotate(q,rope_for(decoder,q,positions)).to(hidden.dtype)
        keys=[rotate(x.keys,rope_for(decoder,x.keys,x.positions)).to(hidden.dtype) for x in sources]
        values=[x.values.to(hidden.dtype) for x in sources]
        keys.append(rotate(k,rope_for(decoder,k,positions)).to(hidden.dtype));values.append(v.to(hidden.dtype))
        out=attention(q,torch.cat(keys),torch.cat(values),causal)
        hidden=hidden+self.projections['o_proj'](out.to(hidden.dtype))
        s='_moe_gen' if self.generation else ''
        hidden=hidden+getattr(self.native,'mlp'+s)(getattr(self.native,'post_attention_layernorm'+s)(hidden))
        return hidden,bank


class Role(nn.Module):
    def __init__(self,decoder,rank,*,generation=False,adapt=True):
        super().__init__();object.__setattr__(self,'decoder',decoder)
        self.blocks=nn.ModuleList(Block(layer,rank,generation) for layer in decoder.layers) if adapt else nn.ModuleList()
    def adapters(self):
        return {n:p for n,p in self.named_parameters() if n.endswith('.a') or n.endswith('.b')}
    def set_active(self,active):
        for p in self.adapters().values():p.requires_grad_(active)
    def forward(self,hidden,positions,sources=(),*,causal=False,checkpointing=True,export=False):
        bank=[]
        for i,block in enumerate(self.blocks):
            src=tuple(source[i] for source in sources)
            if export:
                # Export EXACT layer-input KV; final Q/O does not influence a prefix's last-layer KV.
                def run(h,block=block,src=src):
                    out,kv=block(h,self.decoder,positions,src,causal)
                    return out,kv.keys,kv.values
                if checkpointing and torch.is_grad_enabled():
                    hidden,k,v=checkpoint(run,hidden,use_reentrant=False)
                else:hidden,k,v=run(hidden)
                bank.append(KV(k,v,positions))
            else:
                def run(h,block=block,src=src):return block(h,self.decoder,positions,src,causal)[0]
                hidden=checkpoint(run,hidden,use_reentrant=False) if checkpointing and torch.is_grad_enabled() else run(hidden)
        return hidden,tuple(bank)


def native_und_forward(decoder,hidden,positions,source,checkpointing=True):
    """Frozen language decoder with gradients into continuous prefix; native causal mask."""
    for i,layer in enumerate(decoder.layers):
        def block(h,layer=layer,kv=source[i]):
            a=layer.self_attn;norm=layer.input_layernorm(h)
            q=a.q_norm(a.q_proj(norm).view(-1,a.num_heads,a.head_dim).float())
            k=a.k_norm(a.k_proj(norm).view(-1,a.num_key_value_heads,a.head_dim).float())
            v=a.v_proj(norm).view_as(k)
            q=rotate(q,rope_for(decoder,q,positions)).to(h.dtype)
            k=rotate(k,rope_for(decoder,k,positions)).to(h.dtype)
            prefix=rotate(kv.keys,rope_for(decoder,kv.keys,kv.positions)).to(h.dtype)
            out=attention(q,torch.cat((prefix,k)),torch.cat((kv.values.to(v.dtype),v)),True)
            h=h+a.o_proj(out.to(h.dtype))
            return h+layer.mlp(layer.post_attention_layernorm(h))
        hidden=checkpoint(block,hidden,use_reentrant=False) if checkpointing and torch.is_grad_enabled() else block(hidden)
    return decoder.norm(hidden)
