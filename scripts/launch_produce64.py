"""Produce teacher targets on all 64 chips across the four job workers.

``prepare_coupled`` is one NPU per process, so parallelism comes from shards
rather than from torchrun: each of the 64 logical chips owns a disjoint slice of
the request list and its own card. This launcher mirrors ``launch_shards64.py``
but targets the v2 module and the parameters this build needs, and it keeps one
log per shard so a dead chip is visible instead of hiding in a shared stream.

Usage::

    python scripts/launch_produce64.py --requests requests_full.jsonl \\
        --output v2_full --nodes 4 --per-node 16
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
PY = "/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python"
MODEL = "/data/zyd_workspace/models/SenseNova-U1.5-8B-MoT"
SOURCE = "/data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b"
CONFIG = "configs/u15_isochronic_state/terminal_active.json"
ROOT = "/data/zyd_workspace/isochronic_data_20261107"

SSH_ARGS = [
    "ssh", "-n", "-p", "2222",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=20",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
]


def _hosts(nodes):
    hosts = [h for h in os.environ.get("VC_WORKER_HOSTS", "").split(",") if h]
    if len(hosts) != nodes:
        raise ValueError("Requires exactly %d workers in VC_WORKER_HOSTS" % nodes)
    return hosts


def _shard_command(args, shard, total, device):
    parts = [
        "env",
        "PYTHONPATH=" + shlex.quote(str(args.code)),
        "PYTHONUNBUFFERED=1",
        "ASCEND_RT_VISIBLE_DEVICES=%d" % device,
        "HCCL_NPU_SOCKET_PORT_RANGE=auto",
        "OMP_NUM_THREADS=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        shlex.quote(args.python),
        "-m", "scripts.prepare_coupled",
        "--model", shlex.quote(args.model),
        "--source", shlex.quote(args.source),
        "--config", args.config,
        "--requests", shlex.quote(str(args.requests)),
        "--source-cache", shlex.quote(args.source_cache_a), shlex.quote(args.source_cache_b),
        "--output", shlex.quote(str(args.output)),
        "--shard-index", str(shard),
        "--shard-count", str(total),
        "--device", "npu:0",
    ]
    if args.limit:
        parts += ["--limit", str(args.limit)]
    if args.skip_existing:
        parts += ["--skip-existing"]
    return " ".join(parts)


def _node_script(args, shards, log_dir):
    lines = ["set -u", "cd " + shlex.quote(str(args.code)),
             "mkdir -p " + shlex.quote(str(log_dir)), "status=0", "pids=()"]
    for shard, device in zip(shards, range(len(shards))):
        log = log_dir / ("shard_%03d.log" % shard)
        lines.append("(%s) > %s 2>&1 &" % (_shard_command(args, shard, args.total, device),
                                           shlex.quote(str(log))))
        lines.append("pids+=($!)")
    lines += ['for pid in "${pids[@]}"; do wait "$pid" || status=1; done',
              "echo $status > %s" % shlex.quote(str(log_dir / "node.exit")),
              "exit $status"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source-cache-a", default=ROOT + "/active/mt")
    ap.add_argument("--source-cache-b", default=ROOT + "/active/mt_b")
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--per-node", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--code", default=str(CODE))
    ap.add_argument("--python", default=PY)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.requests = Path(args.requests).resolve()
    args.output = Path(args.output).resolve()
    args.total = args.nodes * args.per_node
    args.log_dir = args.output.with_name(args.output.name + "_logs")
    hosts = _hosts(args.nodes)

    if not args.requests.exists():
        raise SystemExit("requests file missing: " + str(args.requests))

    launches = []
    for index, host in enumerate(hosts):
        shards = list(range(index * args.per_node, (index + 1) * args.per_node))
        log_dir = args.log_dir / ("node%d" % index)
        script = _node_script(args, shards, log_dir)
        command = ["bash", "-lc", script] if index == 0 else \
            [*SSH_ARGS, "root@" + host, "bash -lc " + shlex.quote(script)]
        launches.append(dict(node=index, host=host,
                             shards=[shards[0], shards[-1]],
                             log_dir=str(log_dir), command=command))

    print(json.dumps(dict(module="scripts.prepare_coupled",
                          shard_total=args.total,
                          requests=str(args.requests),
                          output=str(args.output),
                          launches=[dict(node=x["node"], host=x["host"], shards=x["shards"])
                                    for x in launches]), indent=2))
    if args.dry_run:
        for item in launches:
            print(item["node"], shlex.join(item["command"])[:180], file=sys.stderr)
        return

    for item in launches:
        Path(item["log_dir"]).mkdir(parents=True, exist_ok=True)
        with open(Path(item["log_dir"]) / "launch.err", "ab") as errors:
            subprocess.Popen(item["command"], stdin=subprocess.DEVNULL,
                             stdout=errors, stderr=errors, start_new_session=True)
    # Give the fleet time to start loading before the caller looks at it.
    time.sleep(20)
    print(json.dumps(dict(launched=len(launches), log_dir=str(args.log_dir))))


if __name__ == "__main__":
    main()
