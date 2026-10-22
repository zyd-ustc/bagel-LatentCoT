"""Exact supplied JSONL index; deterministic prompt-family split and resumable sampling."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
import uuid
import zlib
from ..t2i.data import Sampler


def prepare(source,root,seed=17):
    source=Path(source).resolve();root=Path(root);root.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect(root/'pairs.sqlite')
    db.execute('CREATE TABLE pairs(id INTEGER PRIMARY KEY,uid TEXT UNIQUE,image TEXT UNIQUE,prompt TEXT,family TEXT,split TEXT,source_line INTEGER,benchmark TEXT,source_file TEXT,original_source_line INTEGER)')
    removed=Counter();read=0
    with source.open() as stream:
        for read,line in enumerate(stream,1):
            row=json.loads(line);prompt=row['prompt'].strip()
            if not prompt:removed['empty_prompt']+=1;continue
            family=' '.join(prompt.split()).casefold()
            bucket=zlib.crc32((str(seed)+'\n'+family).encode())%10000
            split='test' if bucket<100 else 'val' if bucket<200 else 'train'
            try:
                db.execute('INSERT INTO pairs(uid,image,prompt,family,split,source_line,benchmark,source_file,original_source_line) VALUES(?,?,?,?,?,?,?,?,?)',
                    (row['uid'],row['image'],prompt,family,split,read,row.get('benchmark',''),row.get('source_file',''),row.get('source_line')))
            except sqlite3.IntegrityError:removed['duplicate_uid_or_image']+=1
            if read%20000==0:
                db.commit();print(json.dumps(dict(event='index',read=read)),flush=True)
    db.execute('CREATE INDEX split_lookup ON pairs(split,id)');db.commit()
    counts=dict(db.execute('SELECT split,COUNT(*) FROM pairs GROUP BY split'))
    if not counts.get('train') or not counts.get('val'):raise ValueError('Empty train or validation split')
    manifest=dict(format='bagel_mot_edit_pairs_v1',identity=str(uuid.uuid4()),source=str(source),source_bytes=source.stat().st_size,
        source_rows=read,counts=counts,removed=dict(removed),seed=seed,grouping='normalized supplied prompt; duplicate UID and image removed',
        image_validation='decode at consumption; preflight sample image audit',
        benchmark_contamination='source includes benchmark-derived prompts; post-training scores on those sources are in-domain')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');db.close();return manifest


class Corpus:
    def __init__(self,root):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.db=sqlite3.connect(f'file:{self.root / "pairs.sqlite"}?mode=ro',uri=True);self.db.row_factory=sqlite3.Row
    def ids(self,split):return [r[0] for r in self.db.execute('SELECT id FROM pairs WHERE split=? ORDER BY id',(split,))]
    def row(self,index):return dict(self.db.execute('SELECT * FROM pairs WHERE id=?',(int(index),)).fetchone())
    def close(self):self.db.close()
