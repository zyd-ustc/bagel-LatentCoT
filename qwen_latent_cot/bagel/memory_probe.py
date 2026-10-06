"""Offline native UND/LM-head QA over layer-indexed Memory KV.

No prompt KV or image enters the Memory-only path. All decoder layers execute
at their original depth. Seed/empty/VIT controls share the same answer protocol.
"""
from contextlib import nullcontext
import json
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from .modeling.bagel.qwen2_navit import NaiveCache
from .inferencer import to_device


class ProbeCapture:
    def __init__(self,steps):
        self.steps=set(steps);self.layers={};self.images={};self.timesteps={}

    def record(self,step,layer,payload):
        if sum(payload['lengths'])==0:raise ValueError('nonempty prompt content is required for Memory QA')
        self.layers.setdefault(step,{})[layer]={k:v.detach().cpu().contiguous().clone() if isinstance(v,torch.Tensor) else v for k,v in payload.items()}

    def save(self,outdir,metadata):
        output=Path(outdir);output.mkdir(parents=True,exist_ok=True)
        if set(self.layers)!=self.steps or set(self.images)!=self.steps:
            raise ValueError('probe steps not captured; steps must fall inside active loop progress')
        expected=set(metadata.get('read_layers',range(metadata['start_layer'],metadata['end_layer'])))
        snapshots=[]
        for step in sorted(self.steps):
            layers=self.layers[step]
            if set(layers)!=expected:raise ValueError('incomplete per-layer probe snapshot')
            tensors={f'{layer}.{name}':state[name] for layer,state in layers.items()
                     for name in ('dynamic_k','dynamic_v','seed_k','seed_v','source_indexes')}
            path=output/f'step_{step:03d}.safetensors';save_file(tensors,str(path))
            imagepath=output/f'step_{step:03d}_x0.png';self.images[step].save(imagepath)
            snapshots.append({'step':step,'timestep':self.timesteps[step],
                'sampling_progress':step/max(metadata['num_timesteps']-2,1),
                'tensor_path':str(path.resolve()),'image_path':str(imagepath.resolve()),
                'question_position_start':max(s['question_position_start'] for s in layers.values()),
                'layers':sorted(layers),'observed_image_kind':'guided_one_step_x0_proxy'})
        selected_prompt_seed=metadata.get('seed_reference')=='selected_native_prompt_layer_input_kv'
        record={**metadata,'snapshots':snapshots,'prompt_kv_exported':selected_prompt_seed,
                'full_prompt_kv_exported':False,'selected_prompt_seed_kv_exported':selected_prompt_seed,
                'memory_carries_across_layers_in_generation':metadata.get('memory_carries_across_layers_in_generation',True),
                'memory_state':metadata.get('memory_state','body_end_hidden'),
                'seed_reference':metadata.get('seed_reference','strict_read_layer_input_kv')}
        (output/'capture.json').write_text(json.dumps(record,indent=2)+'\n')
        return record


NUMBER_WORDS=dict(zip('zero one two three four five six seven eight nine ten eleven twelve'.split(),map(str,range(13))))


def canonical_answer(value):
    s=str(value).strip().lower().rstrip('.')
    return NUMBER_WORDS.get(s,s)


def probe_questions(row,maximum=4,max_count=12):
    if maximum<1 or max_count<0:raise ValueError('invalid question/count limit')
    if 'probe_questions' in row:
        questions=row['probe_questions']
    else:
        raw=row.get('vqa_list',list(zip(row.get('yn_question_list',[]),row.get('yn_answer_list',[]))))
        skills=row.get('skills',['other']*len(raw));questions=[]
        if len(skills)!=len(raw):raise ValueError('question/skill lengths differ')
        # A deterministic stratified subset, no image-dependent question selection.
        order=list(range(len(raw)))
        chosen=[];seen=set()
        for i in order:
            if skills[i] not in seen:chosen.append(i);seen.add(skills[i])
        chosen+=(i for i in order if i not in chosen)
        for i in chosen[:maximum]:
            q,a=raw[i];count=q.lower().startswith('how many')
            candidates=[str(n) for n in range(max_count+1)]+['unknown'] if count else ['yes','no','unknown']
            questions.append({'question_id':str(i),'question':q,'desired_answer':canonical_answer(a),
                'candidates':candidates,'skill':skills[i]})
    if not questions:raise ValueError('probe requires vqa_list/TIIF/probe_questions')
    out=[]
    for i,question in enumerate(questions[:maximum]):
        q=dict(question);q.setdefault('question_id',str(i));q.setdefault('skill','other')
        q['question_id']=str(q['question_id'])
        if not isinstance(q.get('question'),str) or not q['question'].strip():raise ValueError('empty probe question')
        q['candidates']=[canonical_answer(a) for a in q['candidates']]
        if 'unknown' not in q['candidates'] or len(set(q['candidates']))!=len(q['candidates']):
            raise ValueError('probe candidates must be distinct and include unknown')
        q['desired_answer']=canonical_answer(q.get('desired_answer','unknown'))
        if q['desired_answer'] not in q['candidates']:raise ValueError('desired answer outside fixed candidates')
        out.append(q)
    if len({q['question_id'] for q in out})!=len(out):raise ValueError('duplicate question IDs')
    return out


class NativeMemoryQA:
    def __init__(self,bundle):
        self.bundle=bundle;self.model=bundle.model
        self.decoder=self.model.language_model.model;self.device=next(self.model.parameters()).device

    def autocast(self):
        return torch.autocast('cuda',dtype=torch.bfloat16) if self.device.type=='cuda' else nullcontext()

    @torch.no_grad()
    def score(self,question,candidates,layer_kv,position_start):
        tok=self.bundle.tokenizer;ids=self.bundle.token_ids
        prefix=[ids['bos_token_id']]+tok.encode(question+' Answer with one word or number; use unknown if not discernible.',add_special_tokens=False)+[ids['eos_token_id'],ids['bos_token_id']]
        sequences=[];answer_ids=[];positions=[];prediction_indexes=[];offset=0
        for candidate in candidates:
            answer=tok.encode(candidate,add_special_tokens=False)
            if not answer:raise ValueError('empty candidate encoding')
            sequence=prefix+answer[:-1];sequences.extend(sequence);answer_ids.extend(answer)
            prediction_indexes.extend(range(offset+len(prefix)-1,offset+len(prefix)-1+len(answer)))
            positions.extend(range(position_start,position_start+len(sequence)));offset+=len(sequence)
        lens=[len(prefix)+len(tok.encode(a,add_special_tokens=False))-1 for a in candidates]
        tokens=torch.tensor(sequences,device=self.device,dtype=torch.long)
        pos=torch.tensor(positions,device=self.device,dtype=torch.long)
        hidden=self.decoder.embed_tokens(tokens)
        with self.autocast():
            cos,sin=self.decoder.rotary_emb(hidden,pos.unsqueeze(0))
            rope=(cos.squeeze(0),sin.squeeze(0))
            for index,layer in enumerate(self.decoder.layers):
                cache=NaiveCache(len(self.decoder.layers));prefix_len=0
                if index in layer_kv:
                    k,v=layer_kv[index];prefix_len=len(k)
                    cache.key_cache[index]=k.to(self.device).repeat(len(candidates),1,1)
                    cache.value_cache[index]=v.to(self.device).repeat(len(candidates),1,1)
                query_indexes=[];cache_indexes=[];start=0
                for length in lens:
                    cache_indexes.extend(range(start,start+prefix_len))
                    query_indexes.extend(range(start+prefix_len,start+prefix_len+length))
                    start+=prefix_len+length
                hidden,_=layer.forward_inference(packed_query_sequence=hidden,
                    query_lens=torch.tensor(lens,dtype=torch.int32,device=self.device),
                    packed_query_position_embeddings=rope,
                    packed_query_indexes=torch.tensor(query_indexes,device=self.device,dtype=torch.long),
                    past_key_values=cache,key_values_lens=torch.tensor([prefix_len]*len(lens),device=self.device,dtype=torch.int32),
                    packed_key_value_indexes=torch.tensor(cache_indexes,device=self.device,dtype=torch.long),
                    update_past_key_values=False,is_causal=True,mode='und')
            chosen=self.decoder.norm(hidden[torch.tensor(prediction_indexes,device=self.device)])
            logits=self.model.language_model.lm_head(chosen).float().log_softmax(-1)
            target=torch.tensor(answer_ids,device=self.device,dtype=torch.long)
            logp=logits.gather(1,target[:,None]).squeeze(1)
        lengths=[len(tok.encode(a,add_special_tokens=False)) for a in candidates]
        values=[float(x.mean()) for x in logp.split(lengths)]
        probability=torch.tensor(values,dtype=torch.float64).softmax(0).tolist()
        return {'prediction':candidates[max(range(len(values)),key=values.__getitem__)],
                'mean_token_log_likelihood':dict(zip(candidates,values)),
                'choice_probability':dict(zip(candidates,probability)),
                'probability_kind':'softmax_of_length_normalized_candidate_log_likelihood_not_calibrated'}

    @torch.no_grad()
    def image_cache(self,image):
        from .modeling.native_image_transform import ImageTransform
        transform=ImageTransform(980,224,14)
        cache=NaiveCache(len(self.decoder.layers))
        inputs,_,_=self.model.prepare_vit_images([0],[0],[image.convert('RGB')],transform,self.bundle.token_ids)
        with self.autocast():cache=self.model.forward_cache_update_vit(cache,**to_device(inputs,self.device))
        return {i:(cache.key_cache[i],cache.value_cache[i]) for i in range(len(self.decoder.layers))}


def snapshot_sources(snapshot):
    data=load_file(snapshot['tensor_path'])
    sources={}
    for name,prefix in [('DYNAMIC','dynamic'),('SEED','seed')]:
        sources[name]={i:(data[f'{i}.{prefix}_k'],data[f'{i}.{prefix}_v']) for i in snapshot['layers']}
    sources['EMPTY']={}
    return sources


def finalize_capture(path):
    from ..evaluation.io import sha256
    value=json.loads(Path(path).read_text())
    for snapshot in value['snapshots']:
        snapshot['tensor_sha256']=sha256(snapshot['tensor_path'])
        snapshot['image_sha256']=sha256(snapshot['image_path'])
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def validate_capture(path,expected_hash):
    from ..evaluation.io import sha256
    if sha256(path)!=expected_hash:raise ValueError('Memory capture changed')
    value=json.loads(Path(path).read_text())
    if value.get('prompt_kv_exported') is not False:
        if (value.get('full_prompt_kv_exported') is not False or
            value.get('seed_reference')!='selected_native_prompt_layer_input_kv' or
            value.get('selected_prompt_seed_kv_exported') is not True):
            raise ValueError('only the explicit selected-prompt SEED reference may enter QA')
    for snapshot in value['snapshots']:
        if sha256(snapshot['tensor_path'])!=snapshot['tensor_sha256'] or sha256(snapshot['image_path'])!=snapshot['image_sha256']:
            raise ValueError('snapshot tensor/image changed')
    return value
