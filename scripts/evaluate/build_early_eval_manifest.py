#!/usr/bin/env python
"""Combine existing hard/ordinary prompts. No fixed-size confirmation set."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--hard',required=True);p.add_argument('--ordinary',required=True);p.add_argument('--output',required=True)
a=p.parse_args();seen=set();rows=[]
for bucket,path in [('structural',a.hard),('ordinary',a.ordinary)]:
 for i,line in enumerate(Path(path).read_text().splitlines()):
  if not line.strip():continue
  r=json.loads(line);normalized=' '.join(r['prompt'].casefold().split())
  if normalized in seen:continue
  seen.add(normalized)
  r.update(prompt_id=f'{bucket}_{i:05d}',bucket=bucket,source=str(Path(path).resolve()))
  rows.append(r)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
text=''.join(json.dumps(r)+'\n' for r in rows)
if out.exists() and out.read_text()!=text:raise ValueError('existing evaluation manifest differs; choose a new path')
out.write_text(text)
print(f'{len(rows)} prompts written to {out}; sha256={hashlib.sha256(text.encode()).hexdigest()}')
