"""Read-only image-path coverage and a bounded full-decode audit of the supplied index."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sqlite3
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--output',required=True);a=p.parse_args()
db=sqlite3.connect('file:'+str(Path(a.corpus)/'pairs.sqlite')+'?mode=ro',uri=True)
rows=db.execute('SELECT id,image FROM pairs ORDER BY id').fetchall()
def exists(item):
 i,path=item
 return None if Path(path).is_file() else dict(id=i,image=path)
missing=[]
with ThreadPoolExecutor(24) as pool:
 for result in pool.map(exists,rows,chunksize=128):
  if result is not None:missing.append(result)
selected=random.Random(17).sample(rows,min(128,len(rows)));decoded=[];bad=[]
for i,path in selected:
 try:
  with Image.open(path) as image:image.load();decoded.append(dict(id=i,size=image.size,mode=image.mode))
 except Exception as error:bad.append(dict(id=i,image=path,error=repr(error)))
record=dict(indexed=len(rows),missing_count=len(missing),missing=missing,decoded=decoded,decode_errors=bad)
Path(a.output).write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ('missing','decoded')}))
if missing or bad:raise SystemExit(1)
