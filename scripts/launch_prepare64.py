"""Launch sharded I/R cache production on four 16-chip ModelArts workers.

``prepare-state-pairs`` cannot use torchrun: every shard owns one chip and one
disjoint slice of the manifest. Each node therefore runs ``per-node`` detached
processes and waits for all of them; the controller only starts those node
supervisors and records their PIDs.

``--limit`` slices the corpus *before* sharding, so ``--limit 32`` on 64 shards
gives one record on each of the first 32 shards. For a single-chip smoke test
use ``--nodes 1 --per-node 1 --limit 32``. Use ``--status`` to read progress
without touching the job.
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
SOURCE = "/data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b"
MODEL = "/data/zyd_workspace/models/SenseNova-U1.5-8B-MoT"

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


def _remote(host, command, *, detached):
    """Node 0 runs locally; the other nodes are reached over the job network.

    The command is passed raw: ``bash -c`` and ``ssh`` already treat their last
    argument as a shell program, so quoting it would make the shell look for a
    program named after the whole script.
    """
    if host is None:
        return ["bash", "-lc", command]
    if detached:
        return ["bash", "-lc", shlex.join([*SSH_ARGS, f"root@{host}", command])]
    return [*SSH_ARGS, f"root@{host}", command]


def _node(hosts, index):
    return None if index == 0 else hosts[index]


def _preflight(hosts, output, *, skip, per_node, python):
    """Require enough idle *logical* NPUs, not just idle cards.

    ``npu-smi`` reports physical cards while torch_npu may expose two logical
    devices per card, so the required idle card count follows the observed
    devices-per-card ratio of this job.
    """
    if skip:
        return
    if (output / "manifest.json").exists():
        raise ValueError("Output already holds a single manifest: " + str(output))
    probe = (
        shlex.quote(python) + " -c 'import torch, torch_npu; print(torch.npu.device_count())' "
        "2>/dev/null; npu-smi info | grep -c 'No running processes found in NPU '"
    )
    for index, host in enumerate(hosts):
        result = subprocess.run(
            _remote(_node(hosts, index), probe, detached=False),
            capture_output=True, text=True, timeout=240,
        )
        if result.returncode:
            raise RuntimeError(
                f"NPU preflight failed on {host}\n{result.stdout}{result.stderr}"
            )
        fields = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        devices = int(fields[-2]) if len(fields) >= 2 and fields[-2].isdigit() else 0
        idle_cards = int(fields[-1]) if fields and fields[-1].isdigit() else 0
        if devices < per_node:
            raise RuntimeError(f"Only {devices} logical NPUs on {host}, need {per_node}")
        per_card = max(1, devices // max(1, idle_cards))
        need = -(-per_node // per_card)
        if idle_cards < need:
            raise RuntimeError(
                f"NPU not fully idle on {host}: idle cards={idle_cards} of {need} "
                f"(devices={devices}, per_node={per_node})"
            )
        print(json.dumps(dict(node=index, host=host, devices=devices,
                              idle_cards=idle_cards, need_cards=need)), flush=True)


def _shard_command(args, shard, shard_total, local_device):
    parts = [
        "env",
        f"PYTHONPATH={shlex.quote(str(args.code))}",
        "PYTHONUNBUFFERED=1",
        f"ASCEND_RT_VISIBLE_DEVICES={local_device}",
        "HCCL_NPU_SOCKET_PORT_RANGE=auto",
        "HCCL_CONNECT_TIMEOUT=3600",
        "HCCL_EXEC_TIMEOUT=3600",
        "OMP_NUM_THREADS=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        shlex.quote(args.python),
        "-m", "qwen_latent_cot.u15_target_correction.cli", "prepare-state-pairs",
        "--model", shlex.quote(str(args.model)),
        "--source", shlex.quote(str(args.source)),
        "--config", shlex.quote(str(args.config)),
        "--mode", args.mode,
        "--output", shlex.quote(str(args.output)),
        "--device", "npu:0",
        "--shard-index", str(shard),
        "--shard-count", str(shard_total),
    ]
    if args.mode == "inverted":
        parts += ["--linear", shlex.quote(str(args.linear))]
    else:
        parts += ["--prompts", shlex.quote(str(args.prompts))]
        if args.targets:
            parts += ["--targets", shlex.quote(str(args.targets))]
        if args.refresh_needs_edit:
            parts.append("--refresh-needs-edit")
        if args.keep_share:
            parts += ["--keep-share", str(args.keep_share)]
        if args.reconstruct_parent:
            parts.append("--reconstruct-parent")
    if args.limit:
        parts += ["--limit", str(args.limit)]
    if args.skip_existing:
        parts.append("--skip-existing")
    if args.cache_base_prediction:
        parts.append("--cache-base-prediction")
    if args.no_lpips:
        parts.append("--no-lpips")
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
        command = _shard_command(args, shard, args.shard_total, device)
        log = log_dir / f"shard_{shard:03d}.log"
        lines.append(f"({command}) > {shlex.quote(str(log))} 2>&1 &")
        lines.append("pids+=($!)")
    lines += [
        'for pid in "${pids[@]}"; do wait "$pid" || status=1; done',
        f"echo $status > {shlex.quote(str(log_dir / 'node.exit'))}",
        "exit $status",
    ]
    return "\n".join(lines)


def _status(args, hosts, nodes, per_node):
    manifests_seen = 0
    for index, host in enumerate(hosts):
        root = shlex.quote(str(args.output))
        log_dir = shlex.quote(str(args.log_dir / f"node{index}"))
        command = "; ".join(
            [
                f"ls {root}/progress 2>/dev/null | wc -l",
                f"ls {root}/manifest.shard_*.json 2>/dev/null | wc -l",
                f"cat {root}/production.jsonl 2>/dev/null | wc -l",
                f"cat {log_dir}/node.exit 2>/dev/null || echo running",
            ]
        )
        result = subprocess.run(
            _remote(_node(hosts, index), command, detached=False),
            capture_output=True, text=True, timeout=120,
        )
        fields = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        while len(fields) < 4:
            fields.append("0")
        markers, manifests, produced, exit_state = fields[:4]
        manifests_seen += int(manifests or 0)
        print(
            json.dumps(
                dict(node=index, host=host, shards=per_node, markers=int(markers or 0),
                     manifests=int(manifests or 0), produced=int(produced or 0),
                     node_exit=exit_state),
                ensure_ascii=False,
            ),
            flush=True,
        )
    print(json.dumps(dict(manifests_expected=nodes, manifests_present=manifests_seen)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inverted", "native-parent"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--linear")
    parser.add_argument("--prompts")
    parser.add_argument("--targets")
    parser.add_argument("--config", default=str(CODE / "configs/u15_isochronic_state/stage2_36.json"))
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--per-node", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--refresh-needs-edit", action="store_true")
    parser.add_argument("--keep-share", type=float, default=0.0)
    parser.add_argument("--reconstruct-parent", action="store_true")
    parser.add_argument("--cache-base-prediction", action="store_true")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--code", default=str(CODE))
    parser.add_argument("--python", default=PY)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if args.mode == "inverted" and not args.linear:
        parser.error("Inverted production needs --linear")
    if args.mode == "native-parent" and not args.prompts:
        parser.error("Native-parent production needs --prompts")
    args.shard_total = args.nodes * args.per_node
    args.output = Path(args.output).resolve()
    args.log_dir = args.output.with_name(args.output.name + "_logs")
    hosts = _hosts(args.nodes)

    if args.status:
        _status(args, hosts, args.nodes, args.per_node)
        return

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
        launches.append(
            dict(
                node=index,
                host=host,
                first=shards[0],
                last=shards[-1],
                log_dir=str(log_dir),
                command=_remote(_node(hosts, index), shell, detached=True),
            )
        )
    print(
        json.dumps(
            dict(
                output=str(args.output),
                mode=args.mode,
                shard_total=args.shard_total,
                per_node=args.per_node,
                config=str(args.config),
                launches=[
                    dict(node=item["node"], host=item["host"],
                         shards=[item["first"], item["last"]], log_dir=item["log_dir"])
                    for item in launches
                ],
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        for item in launches:
            print("# node %d" % item["node"], file=sys.stderr)
            print(shlex.join(item["command"]), file=sys.stderr)
        return

    _preflight(hosts, args.output, skip=args.skip_preflight,
               per_node=args.per_node, python=args.python)
    pids = []
    for item in launches:
        Path(item["log_dir"]).mkdir(parents=True, exist_ok=True)
        with open(Path(item["log_dir"]) / "launch.err", "ab") as errors:
            subprocess.Popen(
                item["command"], stdin=subprocess.DEVNULL,
                stdout=errors, stderr=errors,
                start_new_session=True,
            )
        pid_path = Path(item["log_dir"]) / "supervisor.pid"
        deadline = time.monotonic() + 240
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.5)
        if not pid_path.exists():
            raise RuntimeError(f"Node {item['node']} did not write a supervisor PID")
        pids.append(dict(node=item["node"], host=item["host"],
                         pid=pid_path.read_text().strip(), log=item["log_dir"]))
    (args.log_dir / "pids.json").write_text(json.dumps(pids, indent=2) + "\n")
    print(json.dumps(dict(launched=pids), indent=2))


if __name__ == "__main__":
    main()
