"""Audit/stop obsolete project training and launch four detached 16-NPU workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

BASE='/data/zyd_workspace/bagel-LatentCoT'
PYTHON=BASE+'/venvs/bagel-NPU/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--code',required=True);p.add_argument('--output',required=True)
    p.add_argument('--corpus',required=True);p.add_argument('--config');p.add_argument('--audit-only',action='store_true')
    p.add_argument('--master-port',default='29641');a=p.parse_args()
    hosts=os.environ['VC_WORKER_HOSTS'].split(',')
    if len(hosts)!=4:raise ValueError('Requires exactly four ModelArts nodes')
    code=str(Path(a.code).resolve());run=str(Path(a.output).resolve());launch=Path(run+'_launch');launch.mkdir(exist_ok=True)
    config=str(Path(a.config or code+'/configs/correction.json').resolve())
    def remote(host,source):
        command=[PYTHON,'-'] if host==hosts[0] else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','2222','root@'+host,shlex.join([PYTHON,'-'])]
        value=subprocess.run(command,input=source,text=True,capture_output=True,timeout=180)
        result=dict(host=host,returncode=value.returncode,stdout=value.stdout,stderr=value.stderr[-5000:])
        return result
    def record(name,value):
        (launch/(name+'.json')).write_text(json.dumps(value,indent=2));print(name,json.dumps(value),flush=True)
    scan='''
import os,json,signal,time
from pathlib import Path
required=REQUIRED

def targets():
 result=[]
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  try:args=path.joinpath('cmdline').read_bytes().decode().split('\\0')
  except (OSError,UnicodeError):continue
  training=('qwen_latent_cot.interaction.cli' in args and any(stage in args for stage in ('sft','stage_a'))) or ('qwen_latent_cot.diagnostics.visual_cli' in args and 'train' in args)
  if training and any(arg.startswith('/data/zyd_workspace/bagel-LatentCoT/outputs/') for arg in args):
   result.append((int(path.name),'torch.distributed.run' in args))
 return result
before=targets()
for pid,launcher in before:
 if launcher:
  try:os.kill(pid,signal.SIGTERM)
  except ProcessLookupError:pass
for _ in range(20):
 if not targets():break
 time.sleep(1)
for pid,_ in targets():
 try:os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:pass
for _ in range(15):
 if not targets():break
 time.sleep(1)
remaining=targets()
print(json.dumps(dict(host=os.uname().nodename,before=before,remaining=remaining,paths={p:Path(p).exists() for p in required},
 revision=Path(CODE+'/DEPLOYED_COMMIT').read_text().strip())))
if remaining:raise RuntimeError('Old project training has not stopped')
'''.replace('REQUIRED',repr([PYTHON,code+'/qwen_latent_cot/correction/cli.py',a.corpus+'/manifest.json',BASE+'/models/Bagel-7B-MoT/ema.safetensors',config])).replace('CODE',repr(code))
    with ThreadPoolExecutor(4) as pool:audits=list(pool.map(lambda h:remote(h,scan),hosts))
    record('preflight_stop',audits)
    if any(v['returncode'] for v in audits):raise RuntimeError('Node audit/stop failed')
    values=[json.loads(v['stdout']) for v in audits]
    if not all(all(v['paths'].values()) for v in values):raise RuntimeError('Missing node dependencies')
    if len({v['revision'] for v in values})!=1:raise RuntimeError('Node code revisions differ')
    if a.audit_only:return
    if Path(run).exists():raise FileExistsError('Main training requires a fresh run path')
    cfg=json.loads(Path(config).read_text());(launch/'config.json').write_text(json.dumps(cfg,indent=2))
    record('run',dict(code=code,revision=values[0]['revision'],output=run,hosts=hosts,world=64,config=cfg,corpus=a.corpus))
    def start(item):
        rank,host=item
        env=dict(NPROC_PER_NODE='16',NNODES='4',NODE_RANK=str(rank),MASTER_ADDR=hosts[0],MASTER_PORT=a.master_port,
                 ASCEND_RT_VISIBLE_DEVICES=','.join(map(str,range(16))))
        args=['bash',code+'/scripts/train_correction_npu.sh',run,'--corpus',a.corpus,'--config',str(launch/'config.json')]
        source='''
import os,subprocess,json
from pathlib import Path
env=os.environ.copy();env.update(ENV)
with open(LOG,'xb') as log,open('/dev/null','rb') as null:
 process=subprocess.Popen(ARGS,stdin=null,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=CODE,start_new_session=True)
Path(PID).write_text(str(process.pid)+'\\n')
print(json.dumps(dict(host=os.uname().nodename,pid=process.pid,node_rank=RANK)))
'''
        import re
        mappings=dict(ENV=env,LOG=str(launch/f'node{rank}.log'),ARGS=args,CODE=code,PID=str(launch/f'node{rank}.pid'),RANK=rank)
        for key in sorted(mappings,key=len,reverse=True):source=re.sub(r'\b'+key+r'\b',lambda _:repr(mappings[key]),source)
        return remote(host,source)
    with ThreadPoolExecutor(4) as pool:launched=list(pool.map(start,enumerate(hosts)))
    record('launch',launched)
    if any(v['returncode'] for v in launched):raise RuntimeError('Partial launch; inspect named node logs and PIDs')


if __name__=='__main__':main()
