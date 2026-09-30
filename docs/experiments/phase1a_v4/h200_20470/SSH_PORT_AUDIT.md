# Registered H200 SSH endpoints — 2026-09-30

Read-only inspection of `/Users/zyd/.ssh/config`, followed by actual batch-mode
SSH connection and `nvidia-smi` on every configured `vr.turbo-ai.com` endpoint.
These are all registered endpoints in this config, not an exhaustive account
inventory or a claim about unregistered external ports.

| SSH port | Alias | Connected hostname | Visible GPUs | Used VRAM per GPU, snapshot |
|---|---|---|---|---|
| 20344 | avgen-v1 | dedicated-developjob-zhizhou-avgen-1-m8eyg | 8 H200 | 65.9–115.1 GiB |
| 20470 | node-zyd1 | dedicated-developjob-zhizhou-avgen-js-public-jp3j7 | 8 H200 | 48.8–54.0 GiB |
| 20474 | node-12 | dedicated-developjob-js-public-huvlx | 8 H200 | 101.6–123.0 GiB |

Every endpoint is reachable; none is an idle 8-card allocation. Utilization
and memory are dynamic, and zero utilization does not mean memory is free.
Retain the explicitly authorized training target 20470, which has the most
consistent VRAM headroom. Do not kill or move unrelated jobs.

Other configured entries: 18026 is `MyGPU-ydzhi` on a different host and its
GPU type is unverified: batch-mode connection to `ydzhi@110.157.241.3:18026`
failed with `Permission denied (publickey,password)`. 32692 is the known
ModelArts NPU endpoint. No arbitrary external port scan was performed.

Internal TCP listeners were also inspected via `/proc/net/tcp{,6}` because
`ss` is absent. Every container listens on internal SSH port 22; external
ports 20344/20470/20474 are platform mappings. Internal service/ephemeral
listeners do not prove that an external forwarded port is available.
The existing HTML servers occupy 127.0.0.1:18765 on 20470 and
127.0.0.1:18766 on 20474; neither service was changed.

## Latest full-card audit — 2026-09-30 11:39 UTC

Memory cells below are **used / free GiB**, converted from nvidia-smi MiB.
All 24 cards are H200. This supersedes the earlier busy-node snapshot above.

| GPU index | 20344 | 20470 | 20474 |
|---|---|---|---|
| 0 | 0.0 / 139.8 | 113.7 / 26.1 | 0.0 / 139.8 |
| 1 | 0.0 / 139.8 | 114.0 / 25.8 | 0.0 / 139.8 |
| 2 | 0.0 / 139.8 | 113.9 / 25.9 | 0.0 / 139.8 |
| 3 | 0.0 / 139.8 | 113.9 / 25.9 | 0.0 / 139.8 |
| 4 | 0.0 / 139.8 | 113.8 / 26.0 | 0.0 / 139.8 |
| 5 | 0.0 / 139.8 | 113.9 / 25.9 | 0.0 / 139.8 |
| 6 | 0.0 / 139.8 | 113.6 / 26.2 | 0.0 / 139.8 |
| 7 | 0.0 / 139.8 | 113.6 / 26.3 | 0.0 / 139.8 |

20344/20474 each show 0% utilization on every GPU and no compute-app processes.
20470 shows 100% utilization on every GPU, including the current BAGEL main
workers and unrelated WAN/style jobs. Do not stop unrelated processes.

Shared-storage verification: all three report device 4013033953 and identical
inode/size for the r2 inferencer and exported train JSONL. Inferencer SHA256:
`185990d18b481597981ffe1e646e3e92ee9bd65226c81ef7a2de183aec37ad17`.
The same absolute model/code/data/output paths are visible on every endpoint;
moving compute would not require another code or dataset upload.
