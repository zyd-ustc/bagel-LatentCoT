"""Read pinned MoT run progress,64 workers, finite metrics, checkpoints and images."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import shlex
import subprocess
import time


def status(run):
    root=Path(run);launch=Path(run+'_launch');manifest=json.loads((launch/'run.json').read_text());rows=[]
    if (root/'train.jsonl').exists():
        for line in (root/'train.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
    def node(item):
        i,host=item;log=launch/f'node{i}.log';text=log.read_text(errors='replace') if log.exists() else ''
        source=f'''import json; from pathlib import Path
workers=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=p.joinpath('cmdline').read_bytes().decode().split('\\0')
 except (OSError,UnicodeError):continue
 if 'qwen_latent_cot.mot_edit.cli' in args and {run!r} in args and 'torch.distributed.run' not in args:workers.append(int(p.name))
print(json.dumps(workers))'''
        py='/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python'
        args=[py,'-'] if i==0 else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','2222','root@'+host,shlex.join([py,'-'])]
        p=subprocess.run(args,input=source,text=True,capture_output=True,timeout=60)
        loaded=sum('"event": "loaded"' in line for line in text.splitlines())
        errors=[l for l in text.splitlines() if any(s in l for s in ('Traceback (most recent call last)','OutOfMemoryError','FloatingPointError','RuntimeError:'))]
        exitfile=launch/f'node{i}.exit.json'
        return dict(node=i,host=host,loaded=loaded,workers=json.loads(p.stdout) if p.returncode==0 else None,
                    probe_error=p.stderr[-500:] if p.returncode else None,errors=errors[-8:],exit=json.loads(exitfile.read_text()) if exitfile.exists() else None,
                    log_tail=text.splitlines()[-4:])
    with ThreadPoolExecutor(4) as pool:nodes=list(pool.map(node,enumerate(manifest['hosts'])))
    finite=all(all(not isinstance(v,float) or math.isfinite(v) for v in row.values()) for row in rows)
    sets={}
    for directory in sorted(root.glob('semantic_[0-9]*')):
        record=directory/'records.json';sets[directory.name]=dict(records=record.exists(),cases=len(list(directory.glob('case_*.json'))),pngs=len(list(directory.glob('*.png'))))
    final=root/'semantic_008000/records.json'
    complete=bool(rows and rows[-1]['step']==8000 and (root/'step_008000.pt').exists() and final.exists()
                  and len(json.loads(final.read_text()))==32
                  and all((root/path).exists() for row in json.loads(final.read_text()) for path in row['arms'].values())
                  and all(n['exit']=={'returncode':0} for n in nodes))
    return dict(run=run,checked=time.time(),latest=rows[-1] if rows else None,finite=finite,
        negative_fraction_max=max((r.get('applied_negative_fraction',0) for r in rows),default=0),
        recent_step_seconds=sum(r['step_seconds'] for r in rows[-20:])/len(rows[-20:]) if rows else None,
        metric_age_seconds=time.time()-(root/'train.jsonl').stat().st_mtime if rows else None,
        checkpoints=[p.name for p in sorted(root.glob('step_*.pt'))],semantic=sets,nodes=nodes,complete=complete)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    print(json.dumps(status(a.run),indent=2))
