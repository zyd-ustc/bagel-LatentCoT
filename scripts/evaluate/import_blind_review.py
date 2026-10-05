#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from qwen_latent_cot.evaluation.blind_review import import_blind_review
p=argparse.ArgumentParser(description='Validate image hashes and import completed anonymous A/B judgments')
p.add_argument('--csv',required=True);p.add_argument('--private-key',required=True);p.add_argument('--output',required=True)
a=p.parse_args();result=import_blind_review(a.csv,a.private_key)
Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
