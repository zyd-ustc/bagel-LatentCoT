"""Independent parent layout/initialization for parity checks only."""
import torch


def legacy_kwargs(kwargs, slots=3):
    # Independent parent layout/initialization: do not call production helpers.
    h = kwargs['packed_query_sequence']; pos = kwargs['packed_query_position_ids']
    native=[]; memory=[]; positions=[]; query=[]; cached=[]; states=[]
    oldoff=qoff=merged=0
    noise=1e-4*torch.randn(slots,h.shape[-1],generator=torch.Generator().manual_seed(0)).to(h)
    for g,p in zip(kwargs['query_lens'].tolist(),kwargs['key_values_lens'].tolist()):
        native += [qoff]+list(range(qoff+slots+1,qoff+g+slots))
        memory += list(range(qoff+1,qoff+1+slots))
        positions += [int(pos[oldoff])]*(slots+1)+pos[oldoff+1:oldoff+g].tolist()
        query += list(range(merged+p,merged+p+g+slots)); cached += list(range(merged,merged+p))
        states.append(h[[oldoff,oldoff+g-1]].float().mean(0).to(h).unsqueeze(0)+noise)
        oldoff+=g;qoff+=g+slots;merged+=p+g+slots
    ids=lambda x:pos.new_tensor(x)
    native,memory=ids(native),ids(memory)
    expanded=h.new_empty(qoff,h.shape[-1]); expanded[native]=h;expanded[memory]=torch.cat(states)
    kw={**kwargs,'packed_query_sequence':expanded,'packed_query_position_ids':ids(positions),
        'query_lens':kwargs['query_lens']+slots,'packed_query_indexes':ids(query),
        'packed_key_value_indexes':ids(cached),'packed_text_indexes':torch.cat([native[kwargs['packed_text_indexes']],memory]),
        'packed_vae_token_indexes':native[kwargs['packed_vae_token_indexes']], 'packed_memory_token_indexes':memory}
    return kw,native

