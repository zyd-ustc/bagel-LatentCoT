"""Release only this thread's finished, idle base-evaluation judge for64-NPU training."""
import json
import os
from pathlib import Path
import signal
import time
from urllib.request import urlopen

RUN=Path('/data/zyd_workspace/bagel-LatentCoT/outputs/bagel_base_sixbench_20261022')
DEST=Path('/data/zyd_workspace/bagel-LatentCoT/outputs/mot_edit_judge_release_20261023.json')


def snapshot(pid):
    path=Path('/proc')/str(pid)
    argv=[x.decode() for x in path.joinpath('cmdline').read_bytes().split(b'\0') if x]
    allowed={'PATH','LD_LIBRARY_PATH','PYTHONPATH','TMPDIR','TRITON_CACHE_DIR','ASCEND_RT_VISIBLE_DEVICES',
             'ASCEND_HOME_PATH','ASCEND_OPP_PATH','ASCEND_AICPU_PATH','ASCEND_TOOLKIT_HOME',
             'VLLM_USE_V1','VLLM_WORKER_MULTIPROC_METHOD','VLLM_ASCEND_ENABLE_NZ','HCCL_CONNECT_TIMEOUT','OMP_NUM_THREADS'}
    env={}
    for part in path.joinpath('environ').read_bytes().split(b'\0'):
        key,sep,value=part.partition(b'=')
        if sep and key.decode() in allowed:env[key.decode()]=value.decode()
    return dict(pid=pid,pgid=os.getpgid(pid),argv=argv,cwd=str(path.joinpath('cwd').resolve()),env=env)


def main():
    with urlopen('http://127.0.0.1:18364/health',timeout=10) as response:health=json.load(response)
    progress=json.loads((RUN/'scoring_qwen27b_v1/control/progress.json').read_text())
    if health['namespace']!='bagel_native_base_sixbench_20261022_v1' or health['inflight']!=0:
        raise RuntimeError('Dedicated base judge is occupied or another namespace')
    if progress['running'] or progress['waiting_generation'] or len(progress['completed'])!=6:
        raise RuntimeError('Base evaluation has active work')
    candidates=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=p.joinpath('cmdline').read_bytes().decode().split('\0')
        except (OSError,UnicodeError):continue
        if 'vllm.entrypoints.openai.api_server' in args and '18363' in args:candidates.append(int(p.name))
    if len(candidates)!=1:raise RuntimeError('Ambiguous dedicated backend')
    specs=[snapshot(health['pid']),snapshot(candidates[0])]
    if str(RUN/'judge/cache') not in specs[0]['argv'] or '/data/ltj/modelscope/Qwen3.8-27B' not in specs[1]['argv']:
        raise RuntimeError('Dedicated service command changed')
    if any(s['pid']!=s['pgid'] for s in specs):raise RuntimeError('Dedicated services need own process group')
    DEST.write_text(json.dumps(dict(health=health,services=specs,released_for='bagel_mot_edit_64npu_20261023_8000'),indent=2))
    for spec in specs:os.killpg(spec['pgid'],signal.SIGTERM)
    time.sleep(12)
    print(json.dumps(dict(released=[s['pid'] for s in specs],recovery=str(DEST))))


if __name__=='__main__':main()
