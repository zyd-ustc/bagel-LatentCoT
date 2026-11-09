"""Run any sharded python module on four 16-chip ModelArts workers.

Every shard owns one logical NPU and one disjoint slice, exactly like
``prepare-state-pairs``. Usage::

    launch_shards64.py --module qwen_latent_cot...produce_r_targets -- \
        --model ... --requests ... --output ...

The launcher appends ``--shard-index`` and ``--shard-count`` itself, waits for
each node to write its supervisor PID, and records ``node.exit`` per node.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

CODE = Path("/data/zyd_workspace/bagel-LatentCoT/code/isochronic-state-flow-20261107")
PY = "/data/zyd_workspace/bagel-LatentCoT/venvs/u15-NPU/bin/python"

SSH_ARGS = [
    "ssh", "-n", "-p", "2222",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=20",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
]


def _hosts(nodes):
    hosts = [item for item in os.environ.get("VC_WORKER_HOSTS", "").split(",") if item]
    if len(hosts) != nodes:
        raise ValueError(f"Requires exactly {nodes} workers in VC_WORKER_HOSTS")
    return hosts


def _node(hosts, index):
    return None if index == 0 else hosts[index]


def _remote(host, command, *, detached):
    if host is None:
        return ["bash", "-lc", command]
    if detached:
        return ["bash", "-lc", shlex.join([*SSH_ARGS, f"root@{host}", command])]
    return [*SSH_ARGS, f"root@{host}", command]


def _shard_command(args, shard, total, device):
    parts = [
        "env",
        f"PYTHONPATH={shlex.quote(str(args.code))}",
        "PYTHONUNBUFFERED=1",
        f"ASCEND_RT_VISIBLE_DEVICES={device}",
        "HCCL_NPU_SOCKET_PORT_RANGE=auto",
        "OMP_NUM_THREADS=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        shlex.quote(args.python),
        "-m", args.module,
    ]
    parts += [shlex.quote(item) for item in args.argv]
    parts += ["--shard-index", str(shard), "--shard-count", str(total)]
    return " ".join(parts)


def _node_script(args, shards, log_dir):
    lines = [
        "set -u",
        f"cd {shlex.quote(str(args.code))}",
        f"mkdir -p {shlex.quote(str(log_dir))}",
        "status=0",
        "pids=()",
    ]
    for shard, device in zip(shards, range(len(shards))):
        log = log_dir / f"shard_{shard:03d}.log"
        lines.append(f"({_shard_command(args, shard, args.shard_total, device)}) > {shlex.quote(str(log))} 2>&1 &")
        lines.append("pids+=($!)")
    lines += [
        'for pid in "${pids[@]}"; do wait "$pid" || status=1; done',
        f"echo $status > {shlex.quote(str(log_dir / 'node.exit'))}",
        "exit $status",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--per-node", type=int, default=16)
    parser.add_argument("--code", default=str(CODE))
    parser.add_argument("--python", default=PY)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = list(args.argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        parser.error("Pass the module arguments after --")
    args.argv = argv
    args.shard_total = args.nodes * args.per_node
    args.log_dir = Path(args.log_dir).resolve()
    hosts = _hosts(args.nodes)

    launches = []
    for index, host in enumerate(hosts):
        shards = list(range(index * args.per_node, (index + 1) * args.per_node))
        log_dir = args.log_dir / f"node{index}"
        script = _node_script(args, shards, log_dir)
        supervisor = [
            shlex.quote(args.python),
            shlex.quote(str(Path(args.code) / "scripts/start_train_node.py")),
            "--log", shlex.quote(str(log_dir / "supervisor.log")),
            "--pid-file", shlex.quote(str(log_dir / "supervisor.pid")),
            "--", "bash", "-lc", shlex.quote(script),
        ]
        shell = f"mkdir -p {shlex.quote(str(log_dir))}; " + " ".join(supervisor)
        launches.append(dict(node=index, host=host, first=shards[0], last=shards[-1],
                             log_dir=str(log_dir),
                             command=_remote(_node(hosts, index), shell, detached=True)))
    print(json.dumps(dict(module=args.module, shard_total=args.shard_total,
                          log_dir=str(args.log_dir),
                          launches=[dict(node=item["node"], host=item["host"],
                                         shards=[item["first"], item["last"]])
                                    for item in launches]), indent=2))
    if args.dry_run:
        for item in launches:
            print(shlex.join(item["command"]), file=sys.stderr)
        return

    pids = []
    for item in launches:
        Path(item["log_dir"]).mkdir(parents=True, exist_ok=True)
        with open(Path(item["log_dir"]) / "launch.err", "ab") as errors:
            subprocess.Popen(item["command"], stdin=subprocess.DEVNULL,
                             stdout=errors, stderr=errors, start_new_session=True)
        pid_path = Path(item["log_dir"]) / "supervisor.pid"
        deadline = time.monotonic() + 240
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.5)
        if not pid_path.exists():
            raise RuntimeError(f"Node {item['node']} did not write a supervisor PID")
        pids.append(dict(node=item["node"], host=item["host"],
                         pid=pid_path.read_text().strip(), log=item["log_dir"]))
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "pids.json").write_text(json.dumps(pids, indent=2) + "\n")
    print(json.dumps(dict(launched=pids), indent=2))


if __name__ == "__main__":
    main()
