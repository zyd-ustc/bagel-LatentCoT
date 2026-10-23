"""Launch exactly this run on four sixteen-NPU nodes, without evicting other tasks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess

BASE=Path('/data/zyd_workspace/bagel-LatentCoT');PYTHON=str(BASE/'venvs/bagel-NPU/bin/python')


def remote(host,source,master):
    args=[PYTHON,'-'] if host==master else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-p','2222','root@'+host,shlex.join([PYTHON,'-'])]
    p=subprocess.run(args,input=source,text=True,capture_output=True,timeout=180)
    return dict(host=host,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr[-4000:])


def main():
    p=argparse.ArgumentParser();p.add_argument('--code',required=True);p.add_argument('--run',required=True);p.add_argument('--corpus',required=True)
    p.add_argument('--config',default='configs/mot_edit_8000.json');p.add_argument('--resume');p.add_argument('--master-port',default='29653');a=p.parse_args()
    hosts=os.environ['VC_WORKER_HOSTS'].split(',')
    if len(hosts)!=4:raise ValueError('Requires four nodes')
    run=Path(a.run);launch=Path(a.run+'_launch')
    if run.exists() and not a.resume:raise FileExistsError('Fresh run output already exists')
    launch.mkdir(exist_ok=bool(a.resume))
    paths=[a.code+'/qwen_latent_cot/mot_edit/cli.py',a.corpus+'/manifest.json',str(BASE/'models/Bagel-7B-MoT/ema.safetensors')]
    audit='''
import json,subprocess,re
from pathlib import Path
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=p.joinpath('cmdline').read_bytes().decode().split('\\0')
 except (OSError,UnicodeError):continue
 if any(x in args for x in ('qwen_latent_cot.mot_edit.cli','qwen_latent_cot.correction.cli','vllm.entrypoints.openai.api_server')):active.append((p.name,args))
smi=subprocess.run(['npu-smi','info'],text=True,capture_output=True,check=True).stdout
occupied=re.findall(r'^\\|\\s+\\d+\\s+\\d+\\s+(\\d+)\\s+\\S+',smi,re.M)
print(json.dumps(dict(paths={p:Path(p).exists() for p in PATHS},active=active,npu_processes=occupied)))
if occupied or active or not all(Path(p).exists() for p in PATHS):raise RuntimeError('Node occupied or missing dependency')
'''.replace('PATHS',repr(paths))
    with ThreadPoolExecutor(4) as pool:audits=list(pool.map(lambda h:remote(h,audit,hosts[0]),hosts))
    (launch/'preflight.json').write_text(json.dumps(audits,indent=2))
    if any(r['returncode'] for r in audits):raise RuntimeError('Four-node audit failed; no process was stopped')
    manifest=dict(run=a.run,code=a.code,corpus=a.corpus,hosts=hosts,world=64,master_port=a.master_port,resume=a.resume,
                  revision=(Path(a.code)/'DEPLOYED_COMMIT').read_text().strip())
    (launch/'run.json').write_text(json.dumps(manifest,indent=2))
    def start(item):
        rank,host=item
        args=['bash',a.code+'/scripts/train_mot_edit_npu.sh',a.run,'--corpus',a.corpus]
        if a.resume:args+=['--resume',a.resume]
        env=dict(NPROC_PER_NODE='16',NNODES='4',NODE_RANK=str(rank),MASTER_ADDR=hosts[0],MASTER_PORT=a.master_port,
                 MOT_CONFIG=str(Path(a.code)/a.config),ASCEND_RT_VISIBLE_DEVICES=','.join(map(str,range(16))))
        source=f'''
import json,os,subprocess
from pathlib import Path
env=dict(os.environ,**{env!r})
wrapper="import subprocess,json; from pathlib import Path; p=subprocess.run("+repr({args!r})+"); Path("+repr({str(launch/f'node{rank}.exit.json')!r})+").write_text(json.dumps(dict(returncode=p.returncode))); raise SystemExit(p.returncode)"
with open({str(launch/f'node{rank}.log')!r},'ab') as log:
 p=subprocess.Popen([{PYTHON!r},'-c',wrapper],cwd={a.code!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
Path({str(launch/f'node{rank}.pid')!r}).write_text(str(p.pid))
print(json.dumps(dict(host=os.uname().nodename,pid=p.pid,node_rank={rank})))
'''
        return remote(host,source,hosts[0])
    with ThreadPoolExecutor(4) as pool:launched=list(pool.map(start,enumerate(hosts)))
    (launch/'launch.json').write_text(json.dumps(launched,indent=2));print(json.dumps(launched,indent=2))
    if any(r['returncode'] for r in launched):raise RuntimeError('Partial launch: inspect named run logs')


if __name__=='__main__':main()
