"""Indexed teacher pairs, grouped by original prompt before train/val/test splitting."""
from collections import Counter
import json
import math
from pathlib import Path
import random
import sqlite3
import uuid
import zlib
from PIL import Image, ImageOps
import numpy as np
import torch


def dimensions(width, height, short=512, long=1024, multiple=16):
    # Do not stretch/crop a counted scene. Extreme ratios need another recipe.
    if min(width, height) < short or max(width,height)/min(width,height) > long/short:
        raise ValueError('image cannot meet short/long-side bounds without upsampling or cropping')
    scale = min(1., long/max(width,height))
    w = max(multiple, int(width*scale)//multiple*multiple)
    h = max(multiple, int(height*scale)//multiple*multiple)
    if min(w,h)<short: raise ValueError('rounded image is below minimum short side')
    return w,h


def build(pairs, output, tokenizer, config, seed=17, *, excluded_prompts=(), excluded_benchmarks=()):
    pairs=Path(pairs).resolve(); output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect(output/'pairs.sqlite')
    db.executescript('''CREATE TABLE pairs(id INTEGER PRIMARY KEY,uid TEXT UNIQUE,image TEXT UNIQUE,
      prompt TEXT,source_file TEXT,source_line INTEGER,benchmark TEXT,width INTEGER,height INTEGER,
      tokens INTEGER,family TEXT,split TEXT);
      CREATE INDEX source_lookup ON pairs(source_file,source_line);''')
    excluded={' '.join(x.split()).casefold() for x in excluded_prompts}
    excluded_benchmarks={x.casefold() for x in excluded_benchmarks}
    removed=Counter(); total=0
    with pairs.open() as stream:
        for total,line in enumerate(stream,1):
            try:
                row=json.loads(line)
                if str(row['benchmark']).casefold() in excluded_benchmarks:
                    removed['evaluation_benchmark']+=1;continue
                prompt=row['prompt'].strip()
                if not prompt: raise ValueError('empty prompt')
                image=Path(row['image']).resolve(strict=True)
                with Image.open(image) as im:
                    im=ImageOps.exif_transpose(im);im.load()
                    w,h=dimensions(*im.size,config['min_short_side'],config['max_long_side'])
                if config.get('expected_geometry') and (w,h)!=tuple(config['expected_geometry']):
                    removed['different_geometry']+=1;continue
                tokens=len(tokenizer.encode(prompt))+4
                if tokens+(h//16)*(w//16)>config['max_sequence_length']:
                    removed['overlength']+=1;continue
                db.execute('INSERT INTO pairs(uid,image,prompt,source_file,source_line,benchmark,width,height,tokens) VALUES(?,?,?,?,?,?,?,?,?)',
                    (row['uid'],str(image),prompt,str(Path(row['source_file']).resolve()),int(row['source_line']),row['benchmark'],w,h,tokens))
            except sqlite3.IntegrityError:removed['duplicate_uid_or_path']+=1
            except (ValueError,KeyError,OSError,Image.DecompressionBombError):removed['invalid_pair_or_size']+=1
            if total%10000==0:
                db.commit();print(json.dumps(dict(event='prepare',read=total,removed=dict(removed))),flush=True)
    db.commit()
    # Recover raw prompts once per original source shard, not via O(n²) line lookups.
    for (source,) in db.execute('SELECT DISTINCT source_file FROM pairs').fetchall():
        wanted={line:uid for line,uid in db.execute('SELECT source_line,uid FROM pairs WHERE source_file=?',(source,))}
        with Path(source).open() as stream:
            for number,line in enumerate(stream,1):
                if number not in wanted:continue
                row=json.loads(line)
                if row['uid']!=wanted[number]:raise ValueError(f'Source UID mismatch: {source}:{number}')
                family=' '.join(row['raw_prompt'].split()).casefold()
                if not family:raise ValueError('Empty original prompt')
                if family in excluded:
                    db.execute('DELETE FROM pairs WHERE source_file=? AND source_line=?',(source,number))
                    removed['evaluation_prompt_family']+=1;del wanted[number];continue
                bucket=zlib.crc32((str(seed)+'\n'+family).encode())%10000
                split='test' if bucket<100 else 'val' if bucket<200 else 'train'
                db.execute('UPDATE pairs SET family=?,split=? WHERE source_file=? AND source_line=?',(family,split,source,number))
                del wanted[number]
        if wanted:raise ValueError(f'Missing source lines in {source}')
        db.commit()
    if db.execute('SELECT COUNT(*) FROM pairs WHERE split IS NULL').fetchone()[0]:raise ValueError('Unassigned family')
    db.execute('CREATE INDEX split_lookup ON pairs(split,id)');db.commit()
    counts=dict(db.execute('SELECT split,COUNT(*) FROM pairs GROUP BY split'))
    if not counts.get('train') or not counts.get('val'):raise ValueError('Empty train/validation split')
    manifest=dict(format='bagel_t2i_pairs_v1',identity=str(uuid.uuid4()),source=str(pairs),source_bytes=pairs.stat().st_size,
                  read=total,counts=counts,removed=dict(removed),seed=seed,
                  grouping='normalized original prompt; global across benchmark labels; duplicate paths removed',
                  screening='full image decode, resolution, token length; not a semantic quality score',
                  min_short_side=config['min_short_side'],max_long_side=config['max_long_side'],max_sequence_length=config['max_sequence_length'])
    if excluded or excluded_benchmarks:
        manifest['evaluation_exclusion']=dict(prompt_families=len(excluded),benchmark_labels=sorted(excluded_benchmarks))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');db.close()
    return manifest


class Pairs:
    def __init__(self, root):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if self.manifest['format']!='bagel_t2i_pairs_v1':raise ValueError('Not a T2I pair corpus')
        self.db=sqlite3.connect(f'file:{self.root / "pairs.sqlite"}?mode=ro',uri=True)
        self.db.row_factory=sqlite3.Row
    def ids(self,split):return [r[0] for r in self.db.execute('SELECT id FROM pairs WHERE split=? ORDER BY id',(split,))]
    def row(self,index):return dict(self.db.execute('SELECT * FROM pairs WHERE id=?',(int(index),)).fetchone())
    def close(self):self.db.close()


class Sampler:
    def __init__(self, ids, seed):
        if not ids:raise ValueError('Empty training corpus')
        self.ids=ids;self.rng=random.Random(seed);self.cursor=0;self.epoch=0;self._shuffle()
    def _shuffle(self):
        n=len(self.ids);self.a=self.rng.randrange(1,max(2,n))
        while math.gcd(self.a,n)!=1:self.a=self.a%n+1
        self.b=self.rng.randrange(n)
    def take(self,n):
        result=[]
        for _ in range(n):
            result.append(self.ids[(self.a*self.cursor+self.b)%len(self.ids)]);self.cursor+=1
            if self.cursor==len(self.ids):self.cursor=0;self.epoch+=1;self._shuffle()
        return result
    def state_dict(self):return dict(n=len(self.ids),rng=self.rng.getstate(),cursor=self.cursor,epoch=self.epoch,a=self.a,b=self.b)
    def load_state_dict(self,s):
        if s['n']!=len(self.ids):raise ValueError('Sampler inventory changed')
        self.rng.setstate(s['rng'])
        for k in ('cursor','epoch','a','b'):setattr(self,k,s[k])


def pixels(row, device):
    with Image.open(row['image']) as image:
        image=ImageOps.exif_transpose(image).convert('RGB').resize((row['width'],row['height']),Image.Resampling.LANCZOS)
        array=np.array(image)
    return torch.from_numpy(array).permute(2,0,1)[None].to(device=device,dtype=torch.float32)/127.5-1
