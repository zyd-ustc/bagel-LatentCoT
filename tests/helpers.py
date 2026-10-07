from types import SimpleNamespace
import torch
from qwen_latent_cot.bagel.internal_loop import LoopConfig, InternalLoopRuntime
from qwen_latent_cot.bagel.modeling.bagel.qwen2_navit import Qwen2Config, Qwen2ForCausalLM, NaiveCache


def fixture(device='cpu', batch=2):
    torch.manual_seed(123)
    cfg = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=4,
        layer_module='Qwen2MoTDecoderLayer', pad_token_id=0)
    llm = Qwen2ForCausalLM(cfg).eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    model = SimpleNamespace(language_model=llm)
    cache = NaiveCache(4)
    glen, plen = ([4, 7], [2, 3]) if batch == 2 else ([6], [3])
    for index in range(4):
        cache.key_cache[index] = torch.randn(sum(plen), 2, 8, device=device, dtype=torch.bfloat16)
        cache.value_cache[index] = torch.randn_like(cache.key_cache[index])
    positions, text, image, queries, cached = [], [], [], [], []
    qoff = merged = 0
    for g, p in zip(glen, plen):
        positions.extend(range(p, p+g))
        text.extend([qoff, qoff+g-1]); image.extend(range(qoff+1, qoff+g-1))
        cached.extend(range(merged, merged+p)); queries.extend(range(merged+p, merged+p+g))
        qoff += g; merged += p+g
    ids = lambda x: torch.tensor(x, device=device, dtype=torch.long)
    kwargs = dict(packed_query_sequence=torch.randn(sum(glen), 32, device=device, dtype=torch.bfloat16),
        query_lens=ids(glen).int(), packed_query_position_ids=ids(positions),
        packed_query_indexes=ids(queries), past_key_values=cache, key_values_lens=ids(plen).int(),
        packed_key_value_indexes=ids(cached), update_past_key_values=False, is_causal=False,
        mode='gen', packed_text_indexes=ids(text), packed_vae_token_indexes=ids(image))
    return model, kwargs

def prepared(batch=2, empty_first=False, rounds=1, mode='LAYERWISE_UND_STATE_REPLACE', device='cpu', start=1, special_tokens=False):
    model, kwargs = fixture(batch=batch,device=device)
    decoder = model.language_model.model
    runtime = InternalLoopRuntime(model, LoopConfig(mode=mode, extra_rounds=rounds,
        start_layer=start, end_layer=3), diagnostics=True)
    lengths = kwargs['key_values_lens'].tolist()
    tokens = torch.tensor([5,6,7,8,9] if batch==2 else [5,6,7],device=device)
    if empty_first: tokens[:lengths[0]]=0
    if special_tokens:
        offset=0
        for n in lengths:
            tokens[offset]=0;tokens[offset+n-1]=1;offset+=n
    cache = NaiveCache(4)
    positions = torch.tensor([j for n in lengths for j in range(n)],device=device)
    runtime.begin_prefill(cache, tokens, lengths, {0,1})
    try:
        decoder.forward_inference(packed_query_sequence=decoder.embed_tokens(tokens),
            query_lens=torch.tensor(lengths,dtype=torch.int32,device=device), packed_query_position_ids=positions,
            packed_query_indexes=torch.arange(len(tokens),device=device), past_key_values=cache,
            key_values_lens=torch.zeros(batch,dtype=torch.int32,device=device),
            packed_key_value_indexes=torch.empty(0,dtype=torch.long,device=device),
            update_past_key_values=True, is_causal=True, mode='und')
    finally: runtime.end_prefill()
    return model, {**kwargs,'past_key_values':cache}, runtime
