"""Launch this U1.5 stage on four idle ModelArts nodes, without evicting jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('code','python','source','model','corpus','config','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--stage',choices=['t2i','caption','correct'],required=True);p.add_argument('--reader');p.add_argument('--gen-weights');p.add_argument('--resume')
    p.add_argument('--master-port',default='29651');a=p.parse_args()
    hosts=os.environ['VC_WORKER_HOSTS'].split(',')
    if len(hosts)!=4:raise ValueError('Requires exactly four nodes')
    if a.stage=='correct' and not a.reader:raise ValueError('--reader required')
    launch=Path(a.output+'_launch')/str(time.time_ns());launch.mkdir(parents=True,exist_ok=False)
    def remote(host,code):
        args=[a.python,'-'] if host==hosts[0] else ['ssh','-o','BatchMode=yes','-p','2222','root@'+host,shlex.join([a.python,'-'])]
        result=subprocess.run(args,input=code,text=True,capture_output=True,timeout=120)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        return json.loads(result.stdout)
    paths=[a.python,a.code+'/scripts/train_u15_npu.sh',a.source+'/NPU_PATCH',a.model+'/config.json',a.corpus+'/manifest.json',a.config]+([a.reader] if a.reader else [])+([a.gen_weights] if a.gen_weights else [])+([a.resume+'/manifest.json'] if a.resume else [])
    check=f'''import json,subprocess
from pathlib import Path
paths={paths!r}
value=subprocess.run(['npu-smi','info'],text=True,capture_output=True,check=True).stdout
print(json.dumps(dict(paths={{p:Path(p).exists() for p in paths}},idle=value.count('No running processes found in NPU ')==8,npu_smi=value)))
'''
    with ThreadPoolExecutor(4) as pool:audit=list(pool.map(lambda h:remote(h,check),hosts))
    (launch/'preflight.json').write_text(json.dumps(audit,indent=2))
    if not all(x['idle'] and all(x['paths'].values()) for x in audit):raise RuntimeError('Missing assets or busy NPU; inspect preflight, no process was stopped')
    def start(pair):
        rank,host=pair;env=dict(NPROC_PER_NODE='16',NNODES='4',NODE_RANK=str(rank),MASTER_ADDR=hosts[0],MASTER_PORT=a.master_port,MODEL_PYTHON=a.python,ASCEND_RT_VISIBLE_DEVICES=','.join(map(str,range(16))))
        args=['bash',a.code+'/scripts/train_u15_npu.sh',a.stage,'--source',a.source,'--model',a.model,'--corpus',a.corpus,'--config',a.config,'--output',a.output]
        for name in ('reader','gen_weights','resume'):
            if getattr(a,name):args+=['--'+name.replace('_','-'),getattr(a,name)]
        source=f'''import subprocess,os,json
from pathlib import Path
env=os.environ.copy();env.update({env!r})
with open({str(launch/f'node{rank}.log')!r},'xb') as log:
 p=subprocess.Popen({args!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,cwd={a.code!r},start_new_session=True)
print(json.dumps(dict(node={rank},pid=p.pid)))
'''
        return remote(host,source)
    with ThreadPoolExecutor(4) as pool:started=list(pool.map(start,enumerate(hosts)))
    (launch/'launch.json').write_text(json.dumps(started,indent=2));print(json.dumps(started))


if __name__=='__main__':main()
