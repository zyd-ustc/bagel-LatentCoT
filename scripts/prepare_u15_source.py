"""Create a pinned, explicitly patched NPU source copy; never edit refs in place."""
import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    revision=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()
    if not revision.startswith('c35c83b'):raise ValueError('This NPU patch targets the audited c35c83b source')
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(a.source/'src',a.output/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(a.source/'LICENSE',a.output/'LICENSE')
    patch=Path(__file__).resolve().parents[1]/'patches/sensenova-u1-npu.patch'
    subprocess.run(['patch','-p1','--batch','--forward','-i',str(patch)],cwd=a.output,check=True)
    (a.output/'SOURCE_REVISION').write_text(revision+'\n')
    (a.output/'NPU_PATCH').write_text('sensenova-u1-npu-v2\n')
    print(a.output.resolve())


if __name__=='__main__':main()
