"""Resolve explicit family splits, then normalize annotations and index tasks."""
import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import mmap
from pathlib import Path
import struct
import os
import time

REF = struct.Struct('<IQB')
ROLES = ('observe_target', 'observe_input', 'generation', 'repair', 'hold', 'trajectory', 'conditional', 'observe_state')
FORMAT = 'annotation_index_v2'
TOKEN_FIELDS = ('goal_prompt', 'target_caption', 'input_caption', 'edit_instruction', 'observed_caption')


def json_records(path):
    """Read JSONL or a JSON array with bounded memory, including chunk boundaries."""
    with Path(path).open(encoding='utf-8') as stream:
        char = stream.read(1)
        while char and char.isspace(): char = stream.read(1)
        if char != '[':
            stream.seek(0)
            for line in stream:
                if line.strip(): yield json.loads(line)
            return
        decoder = json.JSONDecoder(); buffer = ''; position = 0; eof = False; need_value = True
        while True:
            while position < len(buffer) and buffer[position].isspace(): position += 1
            if position < len(buffer) and buffer[position] == ']':
                if buffer[position+1:].strip() or stream.read().strip(): raise ValueError('trailing JSON array data')
                return
            if not need_value and position < len(buffer) and buffer[position] == ',':
                position += 1; need_value = True
                continue
            if position < len(buffer):
                if not need_value: raise ValueError('missing JSON array comma')
                try: value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof: raise
                else:
                    yield value; position = end; need_value = False; continue
            if eof: raise ValueError('unterminated JSON array')
            chunk = stream.read(1024 * 1024); eof = not chunk
            buffer = buffer[position:] + chunk; position = 0


def clean_instruction(text):
    text = text.replace('<image>', '').strip()
    import re
    return re.sub(r'^(?:Generate an image (?:of|for me)|Generate an image for me|Create (?:a picture showing|an image for)|'
                  r'Image generation|Visualize the following|Edit the image|Editing task|Image modification|'
                  r'Picture modification|Alter the image|Image edition|Image edit|Modify the image|Transform this image|Edit the picture|'
                  r'Visual depiction of|Create an image showing|Generate a picture of)\s*:\s*', '', text, flags=re.I).strip()


def resolve_path(path, source, roots):
    if not path: return None
    path = str(path)
    for old, new in sorted(roots.items(), key=lambda item: -len(item[0])):
        if path.startswith(old): path = str(new) + path[len(old):]; break
    return os.path.abspath(os.path.join(os.path.dirname(source), path))


def normalized_records(row, source, roots):
    """Preserve labels' scope; a next edited image is not necessarily a gold target."""
    name, kind, path = source['name'], source['format'], source['path']
    relocate = lambda p: resolve_path(p, path, roots)
    sid = str(row.get('id', row.get('sample_id', row.get('source_index', ''))))
    if not sid: sid = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:24]
    namespace = source.get('family_namespace', name)
    family = str(row.get('source_sample_id', row.get('origin_sample_id', row.get('trajectory_id', row.get('family_id', sid)))))
    common = dict(record_id=f'{name}:{sid}', source=name, family_node=f'{namespace}:{family}',
                  anchor=source.get('split', row.get('origin_split', row.get('split')) if source.get('inherit_split', False) else None),
                  diagnostic=source.get('diagnostic', False) or not source.get('enabled', True), annotation=dict(origin='source_annotation', scope='endpoint'),
                  edit_type=row.get('edit_type', ['other']), condition_kind='none',
                  original=dict(path=path, id=sid, annotation=row), input_images=[], input_caption='', target_image=None,
                  target_caption='', goal_prompt='', edit_instruction='', target_is_goal=False)
    if kind == 'unimoe':
        human = '\n'.join(c['value'] for c in row['conversations'] if c['from'] in ('human', 'user'))
        assistant = '\n'.join(c['value'] for c in row['conversations'] if c['from'] in ('gpt', 'assistant'))
        import re
        tags = re.findall(r'\[TASK([0-4])\]', assistant)
        if not tags: raise ValueError('unimoe row has no task marker')
        task = int(tags[-1]); goal = clean_instruction(human)
        common.update(target_image=relocate(row['image_gen']), target_is_goal=True,
                      raw_human=human, task_type=('generate' if task < 2 else 'edit' if task == 2 else 'conditional'))
        if task < 2:
            common.update(goal_prompt=goal, target_caption=goal)
            common['planning_text'] = assistant.split('</think>')[0]+'</think>' if task == 1 and '</think>' in assistant else ''
        else:
            common['input_images'] = [relocate(p) for p in row['image']]
            common['edit_instruction'] = goal
            if task == 2:
                common.update(goal_prompt=row.get('output_caption', '').strip(),
                              target_caption=row.get('output_caption', '').strip(), input_caption=row.get('input_caption', '').strip())
                common['feedback_available'] = bool(common['goal_prompt'] and common['goal_prompt'] != common['input_caption'])
            else:
                lower=human.lower()
                condition='depth' if 'depth' in lower else 'segmentation' if 'segmentation' in lower else 'pose' if 'pose' in lower else 'canny'
                common.update(goal_prompt=goal, condition_kind=condition if task == 3 else 'degraded')
        # Real scene identities, not repeated black/white placeholders.
        if common['input_images']:
            p = Path(common['input_images'][0]); parts = p.parts
            if 'MagicBrush' in name: common['family_node'] = f'MagicBrush:{p.parent.name}'
            elif 'UnicEdit' in name: common['family_node'] = f'UnicEdit:{p.parent.parent.name}'
            elif 'ImgEdit' in name:
                common['family_node'] = 'ImgEdit:'+p.stem.split('_segment_')[0]
        yield common; return
    if kind == 'cort':
        images = [relocate(p) for p in row.get('image_paths', [])]; steps = row.get('steps', [])
        goal = (row.get('prompt') or '').strip()
        if row.get('adapter_role') == 'stage12_stage2_visual_state_only':
            common.update(input_images=images, goal_prompt=goal, task_type='asset'); yield common; return
        terminal = bool(steps and images and len(steps)==len(images) and not steps[-1].get('fix') and
                        steps[-1].get('analysis', '').strip().lower() == 'the image matches the prompt.')
        for i, step in enumerate(steps):
            correction = (step.get('fix') or '').strip()
            if correction and i < len(images):
                yield dict(common, record_id=f'{name}:{sid}:step{i}', task_type='edit', goal_prompt=goal,
                           input_images=[images[i]], target_image=images[-1] if terminal else None,
                           target_caption=goal if terminal else '', target_is_goal=terminal,
                           edit_instruction=correction, source_analysis=step.get('analysis', ''), feedback_available=True)
        if terminal:
            yield dict(common, record_id=f'{name}:{sid}:terminal', task_type='generate',
                       goal_prompt=goal, target_caption=goal, target_image=images[-1], target_is_goal=True, is_noop=True)
        return
    if kind == 'structured':
        noop = row.get('is_noop', False)
        common.update(task_type='edit', goal_prompt=row.get('target_prompt', ''),
                      input_images=[relocate(row['source_image'])], target_image=relocate(row.get('target_image')),
                      edit_instruction='' if noop else row.get('instruction', ''), feedback_available=not noop,
                      preserve_constraints=row.get('preserve_constraints', []), original_reflection=row.get('reflection', ''))
        if noop:
            common.update(target_image=common['input_images'][0], target_caption=common['goal_prompt'], target_is_goal=True, is_noop=True)
        yield common; return
    if kind in ('paired', 'trajectory', 'text'):
        feedback = row.get('feedback')
        common.update(task_type='trajectory' if row.get('state') else 'edit' if feedback else 'generate',
                      target_image=relocate(row.get('image')), target_caption=row.get('target_caption', row.get('caption', '')),
                      goal_prompt=row.get('prompt') or row.get('target_caption', row.get('caption', '')),
                      target_is_goal=not feedback or feedback.get('satisfied', False), facts=row.get('facts', row.get('target_facts', [])))
        if common['goal_prompt'] == 'An image.' and not source.get('diagnostic'):
            common['goal_prompt'] = common['target_caption']
        if row.get('annotation'):common['annotation']=row['annotation']
        if row.get('label_kind')=='observation_proxy':
            common.update(observed_caption=row.get('observation_caption',''),
                          annotation=dict(origin='manual_checked' if row.get('human_reviewed') else 'source_annotation',scope='preview'),
                          bound_observation=True,t=row['t'],noise_seed=row['noise_seed'],state_hash=row.get('observation_state_hash'))
        if feedback:
            common.update(edit_instruction=feedback.get('correction', ''), feedback=feedback,
                          annotation=dict(origin=feedback.get('origin', 'source_annotation'), scope='state' if row.get('state') else 'endpoint'))
            if not row.get('state'):
                if feedback.get('satisfied'):
                    common.update(is_noop=True)
                else:
                    common.update(input_images=[common['target_image']], target_image=None, feedback_available=True)
        if row.get('state'):
            common['annotation']=dict(origin=common['annotation']['origin'],scope='state' if not common.get('bound_observation') else 'preview')
            common.update(state=relocate(row['state']), teacher=relocate(row.get('teacher')),
                          t=row['t'], noise_seed=row['noise_seed'], pass_index=row.get('round', 0),
                          state_hash=row.get('observation_state_hash',row.get('state_hash')), trajectory_checkpoint_hash=row.get('trajectory_checkpoint_hash'))
        yield common; return
    if kind in ('legacy_manifest', 'echo_raw', 'entry', 'nofix'):
        goal = row.get('prompt', row.get('instruction', ''))
        target = row.get('gt_image_path', row.get('img_gt_path', row.get('output_image')))
        if kind in ('entry','nofix'):
            steps=row.get('steps',[])
            if kind=='entry':
                base=Path(source['images_root'])/sid
                images=[str(base/f'img{i}.png') for i in range(max(len(steps),1))]
                # Entry annotations are aliases. Only explicit terminal markers claim a goal.
                terminal=bool(steps and (steps[-1].get('analysis','').strip().lower() in
                    ('the image matches the prompt.','image matches the prompt.') or '<|cort_end|>' in steps[-1].get('analysis','')) and not steps[-1].get('fix'))
            else:
                model=row.get('generator_model','januspro7b');model='janus_pro' if model=='januspro7b' else model
                base=Path(source.get('images_root',Path(path).parent.parent/'intermediate'))/row.get('source','sharegpt4o')/model/sid
                images=[str(base/step['input_image']) for step in steps if step.get('input_image')]
                terminal=row.get('stop_reason')=='no_fix' and bool(steps) and not steps[-1].get('fix')
            if terminal and steps: steps=[dict(step,analysis='The image matches the prompt.' if i==len(steps)-1 else step.get('analysis','')) for i,step in enumerate(steps)]
            if not steps:
                common.update(input_images=[relocate(p) for p in images],task_type='asset',goal_prompt=goal);yield common;return
            yield from normalized_records(dict(row,image_paths=images,steps=steps),dict(source,format='cort'),roots);return
        common.update(task_type='generate', goal_prompt=goal, target_caption=goal,
                      target_image=relocate(target), input_images=[relocate(row['img0_path'])] if row.get('img0_path') else [],
                      target_is_goal=bool(target))
        yield common; return
    raise ValueError('unknown source format: '+kind)


class Shards:
    def __init__(self, root, limit):
        self.root = Path(root); self.root.mkdir(parents=True); self.limit = limit; self.files = []
        self.stream = None; self.count = 0

    def append(self, row):
        if self.stream is None or self.count == self.limit:
            self.close(); name = f'{len(self.files):06d}.jsonl'; self.stream = (self.root/name).open('wb')
            self.files.append(dict(path=name, rows=0)); self.count = 0; self.sha = hashlib.sha256()
        offset = self.stream.tell(); data = (json.dumps(row, ensure_ascii=False)+'\n').encode()
        self.stream.write(data); self.sha.update(data); self.count += 1; self.files[-1]['rows'] += 1
        return len(self.files)-1, offset

    def close(self):
        if self.stream:
            self.stream.close(); self.files[-1]['sha256'] = self.sha.hexdigest(); self.stream = None


def family_anchors(registry):
    """Collect explicit splits before aliases are indexed, regardless of source order."""
    anchors = {}
    for source in registry['sources']:
        if not source.get('split') and not source.get('inherit_split'): continue
        for raw in json_records(source['path']):
            for row in normalized_records(raw, source, registry.get('roots', {})):
                node, split = row['family_node'], row['anchor']
                if split is None: continue
                if split not in ('train', 'val', 'test'): raise ValueError('unknown family split: '+str(split))
                if node in anchors and anchors[node] != split:
                    raise ValueError(f'conflicting explicit splits for family {node}: {anchors[node]} and {split}')
                anchors[node] = split
    return anchors


def build(registry, output, *, tokenizer=None, batch_tokenizer=None, shard_size=100000,
          batch_size=1024, progress_every=100000):
    """Build every eligible annotation directly; never stat, open or hash image assets."""
    registry = dict(registry); out = Path(output)
    anchors = family_anchors(registry)
    out.mkdir(parents=True, exist_ok=False)
    records = Shards(out/'records', shard_size)
    counts = Counter(); indexes = {}; handles = {}; start = time.monotonic()
    roots = registry.get('roots', {})

    def progress(source, raw_count, complete=False):
        seconds = time.monotonic()-start
        print(json.dumps(dict(event='source_complete' if complete else 'progress', source=source,
              source_raw=raw_count, normalized=counts['normalized'],
              indexed=sum(item['count'] for item in indexes.values()),
              seconds=round(seconds, 1), rows_per_second=round(counts['normalized']/max(seconds,.001),1))), flush=True)

    def add(task, row, role, shard, offset):
        budget = registry.get('context_token_limit', 4096); lengths = row.get('tokens', {})
        output_tokens = 8 if role==4 or (role==5 and row['feedback'].get('satisfied')) else lengths.get('observed_caption' if role==7 else 'input_caption' if role==1 else 'edit_instruction' if role in (3,5) else 'target_caption', 0)
        prompt_tokens = lengths.get('edit_instruction',0) if role==6 and row['edit_instruction'] else lengths.get('goal_prompt',0)
        semantic = task in ('observe','endpoint','trajectory')
        if (semantic and output_tokens>max(registry.get('output_token_budgets',[48,128,512]))) or (
            semantic and output_tokens+64>budget) or (task!='observe' and prompt_tokens+64>=registry.get('writer_position_start',2048)):
            counts['context_ineligible:'+task] += 1; return
        if role==3 and lengths.get('goal_prompt',0)+output_tokens+64>=registry.get('writer_position_start',2048):
            counts['context_ineligible:'+task] += 1; return
        name = f'{task}/{row["source"]}/{row["split"]}.idx'
        if name not in handles:
            path = out/'views'/name; path.parent.mkdir(parents=True, exist_ok=True); handles[name] = path.open('wb')
            indexes[name] = dict(task=task, source=row['source'], split=row['split'], count=0, hash=hashlib.sha256())
        data = REF.pack(shard, offset, role)
        handles[name].write(data); indexes[name]['hash'].update(data); indexes[name]['count'] += 1
        counts['indexed:'+task] += 1

    def append(row):
        node = row.pop('family_node')
        row['family_id'] = hashlib.sha256(node.encode()).hexdigest()
        bucket = int(row['family_id'][:16],16)%10000
        row.pop('anchor')
        row['split'] = anchors.get(node) or ('test' if bucket<10 else 'val' if bucket<20 else 'train')
        row['episode_id'] = hashlib.sha256(row['record_id'].encode()).hexdigest()
        shard, offset = records.append(row); counts['normalized'] += 1
        if row['diagnostic']: return
        if row.get('state') or row.get('bound_observation'):
            if row['t']>=1 and not registry.get('pure_noise_trajectory',False):
                counts['pure_noise_diagnostic']+=1; return
            if row.get('feedback'): add('trajectory',row,5,shard,offset)
            elif row.get('observed_caption'): add('observe',row,7,shard,offset)
            return
        target = bool(row['target_image']); source = bool(row['input_images'])
        if target and row['target_caption']: add('observe',row,0,shard,offset)
        if source and row['input_caption']: add('observe',row,1,shard,offset)
        if target and row['target_is_goal'] and row['goal_prompt'] and row['condition_kind']=='none':
            add('generation',row,2,shard,offset)
            if row.get('is_noop') or (source and row.get('feedback_available',True) and row['edit_instruction']):
                add('endpoint',row,4,shard,offset)
        if source and row.get('feedback_available',True) and row['goal_prompt'] and row['edit_instruction']:
            add('endpoint',row,3,shard,offset)
        if target and source and (row['edit_instruction'] or row['goal_prompt']):
            add('conditional',row,6,shard,offset)

    def flush(batch):
        if tokenizer or batch_tokenizer:
            texts = list(dict.fromkeys(row.get(key,'') for row in batch for key in TOKEN_FIELDS))
            if batch_tokenizer:
                lengths = batch_tokenizer(texts)
            else:
                lengths = [len(tokenizer(text)) for text in texts]
            sizes = dict(zip(texts,lengths))
            for row in batch: row['tokens'] = {key:sizes[row.get(key,'')] for key in TOKEN_FIELDS}
        for row in batch: append(row)
        batch.clear()

    try:
        for source in registry['sources']:
            name = source['name']; batch = []; raw_count = 0
            print(json.dumps(dict(event='source_start',source=name,path=source['path'])),flush=True)
            for raw_count, raw in enumerate(json_records(source['path']),1):
                counts['raw:'+name] += 1
                for index,row in enumerate(normalized_records(raw,source,roots)):
                    # Occurrence IDs retain repeated annotations as separate training examples.
                    row['record_id'] += f':row{raw_count}:item{index}'
                    batch.append(row)
                    if len(batch)>=batch_size: flush(batch)
                if progress_every and raw_count%progress_every==0:
                    if batch: flush(batch)
                    progress(name,raw_count)
            if batch: flush(batch)
            progress(name,raw_count,complete=True)
    finally:
        records.close()
        for handle in handles.values(): handle.close()
    for item in indexes.values(): item['sha256'] = item.pop('hash').hexdigest()
    manifest = dict(format=FORMAT, registry=registry, image_access='training_only', deduplication='none',
                    split_policy='explicit_family_anchors_v1',
                    source_sampling='sqrt_task_records', counts=dict(counts), shards=records.files,
                    views=[dict(path=key,**value) for key,value in sorted(indexes.items())])
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


class TaskView:
    def __init__(self, corpus, info):
        self.corpus, self.info = corpus, info; self.path = corpus.root/'views'/info['path']
        self.stream = self.path.open('rb'); self.index = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
    def __len__(self): return self.info['count']
    def __getitem__(self, index):
        if not 0 <= index < len(self): raise IndexError(index)
        shard, offset, role = REF.unpack_from(self.index, index*REF.size)
        return self.corpus.materialize(self.corpus.record(shard, offset), role)
    def close(self): self.index.close(); self.stream.close()


class Corpus:
    def __init__(self, root):
        self.root = Path(root); self.manifest = json.loads((self.root/'manifest.json').read_text()); self.handles = OrderedDict()
        if self.manifest.get('format') != FORMAT: raise ValueError('rebuild corpus with the annotation index builder')
        self.views = [TaskView(self, info) for info in self.manifest['views'] if info['count']]
    @property
    def fingerprint(self): return self.manifest['fingerprint']
    def record(self, shard, offset):
        if shard not in self.handles:
            self.handles[shard] = (self.root/'records'/self.manifest['shards'][shard]['path']).open('rb')
            if len(self.handles)>8: self.handles.popitem(last=False)[1].close()
        stream=self.handles[shard];self.handles.move_to_end(shard);stream.seek(offset);return json.loads(stream.readline())
    def groups(self, split, tasks=None):
        return [v for v in self.views if v.info['split']==split and (tasks is None or v.info['task'] in tasks)]
    def select(self, split='val', limit=64, tasks=None, *, max_rows=128):
        """Round-robin bounded index reads across source/task views."""
        if limit<=0 or max_rows<=0: raise ValueError('validation requires a positive bounded budget')
        chosen=set();result=[];views=self.groups(split,tasks);seen=set()
        for position in range(max_rows*4):
            for view in views:
                if position>=len(view): continue
                row=view[position]
                if row['sample_id'] in seen: continue
                family=row['family_id']
                if family not in chosen and len(chosen)>=limit: continue
                chosen.add(family);seen.add(row['sample_id']);result.append(row)
                if len(result)>=max_rows:return result
            if all(position>=len(v) for v in views):break
        return result
    def materialize(self, row, role):
        relocate=lambda path:resolve_path(path,row['original']['path'],self.manifest['registry'].get('roots',{}))
        task='observe' if role in (0,1,7) else 'feedback' if role in (3,4,5) else 'generation'
        image = relocate(row['input_images'][0] if role in (1,3) else row['target_image'])
        caption=row.get('observed_caption','') if role==7 else row['input_caption'] if role==1 else row['target_caption']
        result=dict(sample_id=hashlib.sha256(f'{row["record_id"]}:{role}'.encode()).hexdigest()[:24],
                    scene_id=row['family_id'], family_id=row['family_id'], pair_id=row['episode_id'], split=row['split'], source=row['source'],
                    task=task, image=image, prompt=row['goal_prompt'], caption=caption,
                    target_caption=row['target_caption'], facts=row.get('facts',[]), label_type='observed' if role in (5,7) else 'clean',
                    label_kind='feedback' if task=='feedback' else 'target_clean', annotation=row['annotation'],
                    flow_supervision=role in (2,4,6), condition_images=[relocate(p) for p in row['input_images']] if role==6 else [],
                    condition_kind=row['condition_kind'] if role==6 else 'none', edit_type=row.get('edit_type',['other']),
                    output_tokens=8 if role==4 or (role==5 and row['feedback'].get('satisfied')) else row.get('tokens',{}).get('observed_caption' if role==7 else 'input_caption' if role==1 else 'edit_instruction' if role in (3,5) else 'target_caption',0),
                    verified=True, confirmed_label=row['annotation']['origin']=='manual_checked')
        if role==6:result['prompt']=row['edit_instruction'] or row['goal_prompt']
        if role in (3,4,5):
            feedback=row.get('feedback') if role==5 else dict(observation='',gap='',correction=row['edit_instruction'] if role==3 else '',
                         satisfied=role==4,origin=row['annotation']['origin'])
            result['feedback']=feedback;result['caption']=feedback['correction'] if not feedback['satisfied'] else 'No change is needed.'
        if role in (5,7):
            result.update({k:row.get(k) for k in ('state','teacher','t','noise_seed','pass_index','state_hash','trajectory_checkpoint_hash') if row.get(k) is not None})
            result['flow_supervision']=False
        return result
    def close(self):
        for view in self.views:view.close()
        for handle in self.handles.values():handle.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['build','inspect'])
    parser.add_argument('--registry');parser.add_argument('--output',required=True);parser.add_argument('--model')
    parser.add_argument('--shard-size',type=int,default=100000)
    parser.add_argument('--batch-size',type=int,default=1024);parser.add_argument('--progress-every',type=int,default=100000)
    args=parser.parse_args(argv)
    if args.command=='inspect': print((Path(args.output)/'manifest.json').read_text());return
    batch_tokenizer=None
    if args.model:
        from transformers import AutoTokenizer
        native=AutoTokenizer.from_pretrained(args.model)
        batch_tokenizer=lambda texts:[len(ids) for ids in native(texts,add_special_tokens=False)['input_ids']]
    if not args.registry:parser.error('build requires --registry')
    manifest=build(json.loads(Path(args.registry).read_text()),args.output,batch_tokenizer=batch_tokenizer,
                   shard_size=args.shard_size,batch_size=args.batch_size,progress_every=args.progress_every)
    print(json.dumps(dict(counts=manifest['counts'],views=manifest['views'],fingerprint=manifest['fingerprint']),indent=2))


if __name__=='__main__':main()
