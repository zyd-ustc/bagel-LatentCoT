"""Export 128 blinded pairs; keep identities outside the reviewer directory."""
import csv
import json
from pathlib import Path
import random
import shutil
from .io import identity,sha256


def make_blind_pack(rows,outdir,maximum=128,seed=20261005):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    base={identity(r):r for r in rows if r['arm']=='BASE'}
    candidates=[r for r in rows if r['arm']!='BASE']
    rng=random.Random(seed);rng.shuffle(candidates);candidates=candidates[:maximum]
    mapping=[];semantic=[]
    csvpath=out/'review.csv'
    if csvpath.exists():return
    with csvpath.open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['pair_id','image_A','image_B','preference_A_B_tie','A_invalid','B_invalid'])
        for i,row in enumerate(candidates):
            a=base[identity(row)];b=row
            swap=bool(rng.randrange(2));a,b=(b,a) if swap else (a,b)
            pairid=f'{i:04d}'
            filenames=[]
            for label,r in (('A',a),('B',b)):
                name=f'{pairid}_{label}.png';shutil.copyfile(r['path'],out/name);filenames.append(name)
            writer.writerow([pairid,*filenames,'','',''])
            semantic.append({'pair_id':pairid,'prompt':row['prompt'],
                'questions':row.get('semantic_questions'),'A_constraint_judgments':None,'B_constraint_judgments':None})
            mapping.append({'pair_id':pairid,'A':a,'B':b,'A_sha256':sha256(out/filenames[0]),'B_sha256':sha256(out/filenames[1])})
    (out/'semantic_review.json').write_text(json.dumps(semantic,indent=2)+'\n')
    # This file is deliberately outside the blinded directory.
    (out.parent/'blind_review_private_key.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (out/'README.md').write_text('Compare visible quality only. Enter A, B or tie. Enter true/false for invalid images. Complete quality review before opening semantic_review.json. That file contains prompts and constraints for a separate semantic audit. Arm identities stay hidden.\n')


def import_blind_review(csvpath,keypath):
    key={r['pair_id']:r for r in json.loads(Path(keypath).read_text())}
    results=[];seen=set()
    with open(csvpath) as f:
        for row in csv.DictReader(f):
            pairid=row['pair_id']
            if pairid in seen or pairid not in key:raise ValueError('unknown or duplicate blind pair')
            seen.add(pairid)
            if not row['preference_A_B_tie']:continue
            if row['preference_A_B_tie'] not in ('A','B','tie'):raise ValueError('invalid blind preference')
            for label in ('A','B'):
                if sha256(Path(csvpath).parent/row['image_'+label])!=key[pairid][label+'_sha256']:
                    raise ValueError('blind image changed')
            for label in ('A','B'):
                if row[label+'_invalid'].lower() not in ('true','false'):raise ValueError('reviewed pair requires A/B invalid labels')
            winner=row['preference_A_B_tie'];own=key[pairid]
            results.append({'pair_id':pairid,'prompt_id':own['A']['prompt_id'],
                'seed':own['A']['seed'],'arm':next(own[l]['arm'] for l in ('A','B') if own[l]['arm']!='BASE'),
                'preference':0 if winner=='tie' else (1 if own[winner]['arm']!='BASE' else -1),
                'A_invalid':row['A_invalid'].lower()=='true','B_invalid':row['B_invalid'].lower()=='true'})
    return {'reviewed_pairs':len(results),'total_pairs':len(key),'complete':len(results)==len(key),'results':results}
