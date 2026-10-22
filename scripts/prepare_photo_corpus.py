"""Build a separate human-caption photo corpus; original records and family splits stay read-only."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import uuid
from PIL import Image
from qwen_latent_cot.correction.data import Corpus,REF

SOURCES={'unimoe2.5_coco_fixed','unimoe2.5_flickr30k_fixed'}


def caption_problem(value):
    text=(value.get('target_caption') or '').strip()
    if not text or not value.get('target_image'):return 'missing_pair'
    if value.get('edit_instruction') or value.get('input_images'):return 'edit_condition'
    if re.search(r'<think>|</think>|<image>|<\|',text):return 'control_markup'
    if re.match(r'^(add|remove|replace|change|convert|swap|delete|erase|transform|edit)\b',text,re.I):return 'edit_imperative'
    if re.search(r'\b(original image|input image|as before|unchanged)\b',text,re.I):return 'relative_caption'
    return None


def build(source,output,*,workers=16,min_size=128):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    corpus=Corpus(source);selected=[v for v in corpus.views if v.info.get('source') in SOURCES]
    if {v.info['source'] for v in selected}!=SOURCES:raise ValueError('Both COCO and Flickr30k are required')
    rows=[];reasons=Counter();seen=set()
    for view in selected:
        for i in range(len(view)):
            shard,offset,role=REF.unpack_from(view.index,i*REF.size)
            value=corpus.record(shard,offset);problem=caption_problem(value)
            if problem:reasons[problem]+=1;continue
            row=view[i]
            pair=(row['split'],row['image'],row['caption'])
            if pair in seen:reasons['duplicate_pair']+=1;continue
            seen.add(pair);rows.append((view.info,row,bytes(view.index[i*REF.size:(i+1)*REF.size])))
        print(json.dumps(dict(event='metadata',source=view.info['source'],split=view.info['split'],retained=len(rows))),flush=True)
    paths=list(dict.fromkeys(row['image'] for _,row,_ in rows))
    def inspect(path):
        try:
            with Image.open(path) as image:
                if min(image.size)<min_size:return path,'small_image'
                image.load()
            return path,None
        except (OSError,ValueError,Image.DecompressionBombError):return path,'unreadable_image'
    invalid={}
    with ThreadPoolExecutor(workers) as pool:
        for i,(path,problem) in enumerate(pool.map(inspect,paths)):
            if problem:invalid[path]=problem
            if (i+1)%10000==0:print(json.dumps(dict(event='images',checked=i+1,total=len(paths))),flush=True)
    # Keep held-out records in their original splits; remove their conflicting train records.
    image_splits={};family_splits={}
    for _,row,_ in rows:
        image_splits.setdefault(row['image'],set()).add(row['split'])
        family_splits.setdefault(row['family_id'],set()).add(row['split'])
    output.mkdir();(output/'views').mkdir();(output/'records').symlink_to(source/'records',target_is_directory=True)
    streams={};counts=Counter();previews=[]
    for info,row,reference in rows:
        problem=invalid.get(row['image'])
        priority={'train':0,'val':1,'test':2}
        preferred=max(image_splits[row['image']]|family_splits[row['family_id']],key=priority.__getitem__)
        if row['split']!=preferred:problem='cross_split_conflict'
        if problem:reasons[problem]+=1;continue
        name=info['path']
        if name not in streams:
            path=output/'views'/name;path.parent.mkdir(parents=True,exist_ok=True);streams[name]=path.open('wb')
        streams[name].write(reference);counts[name]+=1
        if sum(p['split']==row['split'] for p in previews)<32:previews.append(row)
    for stream in streams.values():stream.close()
    manifest=dict(corpus.manifest)
    manifest['views']=[dict({k:value for k,value in v.info.items() if k!='sha256'},count=counts[v.info['path']]) for v in selected if counts[v.info['path']]]
    manifest['fingerprint']='photo-human-v2-'+str(uuid.uuid4())
    manifest['selection']=dict(parent=str(source),parent_fingerprint=corpus.manifest['fingerprint'],sources=sorted(SOURCES),
        policy='human-caption photos; independent captions; full image decode; unique pairs; no split conflicts',min_image_side=min_size,
        sampling='pair_count',split_conflict_policy='test > val > train; discard lower-priority records without relabeling',semantic_scope='source restriction and structural screening; not exhaustive object/count verification')
    # Photographic source weighting follows accepted pair counts.
    manifest['sampling']='pair_count'
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(status='complete',unique_images_checked=len(paths),invalid_images=len(invalid),removed=dict(reasons),
        pairs_by_split={split:sum(v['count'] for v in manifest['views'] if v['split']==split) for split in ('train','val','test')},
        views=manifest['views'],previews=previews,selection=manifest['selection'])
    (output/'selection.json').write_text(json.dumps(report,indent=2)+'\n');corpus.close()
    print(json.dumps({k:v for k,v in report.items() if k not in ('previews','views')},indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=16);a=p.parse_args();build(a.source,a.output,workers=a.workers)
