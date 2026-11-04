"""Target-image supervision; frozen visual features retain image gradients."""
import torch
from torch.nn import functional as F
from .runtime import autocast


def weighted(error,weight):
    return (error*weight).sum()/(weight.sum()*error.shape[1]).clamp_min(1)


def features(rt,image):
    m=rt.model
    mean=image.new_tensor([.485,.456,.406])[None,:,None,None]
    std=image.new_tensor([.229,.224,.225])[None,:,None,None]
    rgb=((image+1)/2-mean)/std
    gh,gw=rt.height//m.patch_size,rt.width//m.patch_size
    pixels=m.patchify(rgb,m.patch_size,channel_first=True).reshape(gh*gw,-1)
    pixels=pixels.to(m.vision_model.embeddings.patch_embedding.weight.dtype)
    tok=rt.tokenizer
    m.img_context_token_id=tok.convert_tokens_to_ids('<IMG_CONTEXT>')
    m.img_start_token_id=tok.convert_tokens_to_ids('<img>')
    query=m._build_t2i_query('<img>'+'<IMG_CONTEXT>'*rt.tokens+'</img>',append_text='')
    ids=tok(query,return_tensors='pt')['input_ids'].to(rt.device)[0]
    positions=(ids==m.img_context_token_id).nonzero().flatten()
    with autocast(rt.device):
        hidden,indexes,mask=m._build_it2i_inputs(tok,query,pixels,torch.tensor([[gh,gw]],device=rt.device))
        from .native_mot import LayerKV
        from torch.utils.checkpoint import checkpoint
        maps=[]
        selected={len(rt.decoder.layers)//3-1,2*len(rt.decoder.layers)//3-1,len(rt.decoder.layers)-1}
        for index,layer in enumerate(rt.decoder.layers):
            def block(x,index=index,layer=layer):
                return layer(x,indexes=indexes,attention_mask=mask['full_attention'],
                             image_gen_indicators=None,exist_non_image_gen_tokens=True,
                             exist_image_gen_tokens=False,past_key_values=LayerKV(index),
                             use_cache=True,update_cache=True)
            hidden=checkpoint(block,hidden,use_reentrant=False) if torch.is_grad_enabled() else block(hidden)
            if index in selected:
                tokens=hidden[:,positions].float()
                maps.append(tokens.transpose(1,2).reshape(1,tokens.shape[-1],*rt.grid))
    return maps


@torch.no_grad()
def target_weights(rt,reference,target,valid,zero,cfg):
    target_f=features(rt,target)
    if zero:
        return torch.zeros_like(valid),valid,target_f
    reference_f=features(rt,reference)
    semantic=(F.normalize(target_f[-1],dim=1)-F.normalize(reference_f[-1],dim=1)).square().mean(1,keepdim=True).sqrt()
    semantic=F.interpolate(semantic,size=target.shape[-2:],mode='bilinear',align_corners=False)
    diff=cfg['weight_pixel_mix']*(target-reference).abs().mean(1,keepdim=True)+(1-cfg['weight_pixel_mix'])*semantic
    average=(diff*valid).sum()/valid.sum()
    edit=(diff/(average+1e-6)).clamp(.25,3)*valid
    keep=(1/(edit+.5)).clamp(.3,2)*valid
    return edit,keep,target_f


def image_loss(rt,pred,target,valid,edit,keep,target_f,cfg):
    current=features(rt,pred)
    struct=pred.new_zeros(()); preserve=pred.new_zeros(())
    for a,b in zip(current,target_f):
        error=(F.normalize(a,dim=1)-F.normalize(b,dim=1)).square()
        struct=struct+weighted(error,F.interpolate(edit,size=a.shape[-2:],mode='area'))/len(current)
        preserve=preserve+weighted(error,F.interpolate(keep,size=a.shape[-2:],mode='area'))/len(current)
    low=F.avg_pool2d(pred-target,4)
    struct=struct+weighted((low.square()+1e-6).sqrt()-1e-3,F.avg_pool2d(edit,4))
    preserve=preserve+weighted((pred-target).abs(),keep)
    border=weighted((pred-target).square(),1-valid)
    return struct+cfg['lambda_keep']*preserve+border
