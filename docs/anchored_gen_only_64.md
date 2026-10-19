# GEN-only 64 卡部署

这条入口训练的是 Anchored Dual-State Loop 的 GEN-only 对照：冻结原生 BAGEL，只更新无状态残差和 gate。它不读 Memory，也不训练 Dual-Correct。合同见 [Anchored Dual-State Loop](anchored_dual_state_loop.md)。同一批图文对上的两阶段日程是 `scripts/anchored_train_64.sh`，说明见 [两阶段日程](anchored_curriculum_64.md)。那条入口还在本地，不在下面的 `ed073a9` 树里。

当前代码是 `ed073a9`（`ed073a9b962d4e25028a0e6c85a69f57c7bbb9fe`）。集群上的树在共享盘：

```text
/data/zyd_workspace/bagel-LatentCoT/code/bagel-LatentCoT-anchored
```

同目录的 `REVISION` 记录这个提交。`bagel-LatentCoT-interaction_20261017` 仍是 2026-10-17 的 `master`（`a71e101`），不要拿它启动这条入口。

## 多实例作业怎么跑

ModelArts 这一组是 4 个实例，每个实例 16 张昇腾卡，一共 64 张。平台在每个实例上执行同一条启动命令。worker-0 不会去 SSH 另外三个实例，另外三个实例的进程也不是由本机 `torchrun` 拉起来的。

每个实例只看见自己的 16 张卡，设备号仍是 `0` 到 `15`。跨实例的会合用平台注入的 `VC_WORKER_HOSTS`：逗号分隔的四个主机名。脚本用本机 `hostname` 做前缀匹配，得到 node rank，再在本机执行：

```text
torch.distributed.run --nnodes=4 --node_rank=<0..3> --nproc_per_node=16
```

主节点地址是 `VC_WORKER_HOSTS` 的第一项，端口默认 `29500`。进程组后端是 HCCL。全局 rank 是 0 到 63。世界大小不是 64 时，脚本和 Python 都会拒绝启动。

笔记本上的调试 SSH（`modelarts-64`，端口 31527）只登到 worker-0。作业启动脚本里有 `/etc/init.d/ssh start`，每个实例的 sshd 听在 **2222**，不听 22。从 worker-0 用 `VC_WORKER_HOSTS` 里的完整主机名可以登到另外三个实例：

```bash
ssh -p 2222 ma-job-...-vj-1-worker-0.ma-job-...
```

Qwen PE 的 64 卡推理就是这样拆开跑的：每个实例执行 `qwen_pe_run_node.sh <node>`，本机起 16 个进程，实例之间没有一个进程组。GEN-only 也可以从 worker-0 用同样的 2222 登进去，在四个实例上各执行同一条训练脚本。只在 worker-0 上执行会停在 HCCL 会合。

## 什么必须放在 `/data`

这组作业的共享盘挂在 `/data`，不在 `/mnt/sfs`。实例自己的根文件系统不共享。代码、权重、语料、图片和 `RUN_DIR` 都放在 `/data` 上，四个实例读到的才是同一份。

| 用途 | 路径 |
|---|---|
| 代码 | `/data/zyd_workspace/bagel-LatentCoT/code/bagel-LatentCoT-anchored` |
| Python | `/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python` |
| 权重 | `/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT` |

脚本已经把 Python 和权重指到上表。作业里解析不了 `github.com`，所以不能在实例上 `git pull`。从能访问 GitHub 的机器更新共享树：

```bash
git archive --format=tar ed073a9 | ssh modelarts-64 'set -euo pipefail
DEST=/data/zyd_workspace/bagel-LatentCoT/code/bagel-LatentCoT-anchored
rm -rf "$DEST"
mkdir -p "$DEST"
tar -xf - -C "$DEST"
printf "%s\n" ed073a9b962d4e25028a0e6c85a69f57c7bbb9fe > "$DEST/REVISION"
chmod 755 "$DEST/scripts/anchored_gen_only_64.sh"'
```

`git archive` 只包含那个提交。工作区里未提交的高噪声课程文件不会进这棵树。

## 数据

`GEN_ONLY_TRAIN` 直接指向现有 annotation index，不用再导出一份 caption-image JSONL。Observe 全量语料是：

```text
/data/zyd_workspace/bagel-LatentCoT/datasets/observe_feedback_corpus
```

入口只打开 `task=generation` 且 `split=train` 的视图。条件文本是记录里和目标图配对的 `goal_prompt`，不是 `target_caption`。图片是解析后的 `target_image`，必须是绝对路径。空 prompt 或空图片会拒绝这一行。验证、测试和 observe 视图不进入这次训练。

采样与 Observe 训练相同：按视图长度的平方根加权，视图内部用互素仿射置换。一次只物化当前 batch。恢复状态保存 fingerprint、视图路径、随机数和游标。`annotation_index_v1` 和 `annotation_index_v2` 都能读。默认 `Corpus` 仍然拒绝 v1，这次训练不需要重建语料。

共享树上的 `ed073a9` 还没有这个读取器。同步本地改动之前，不要把 `GEN_ONLY_TRAIN` 指到语料目录。

原来的 JSONL 仍然可用，它的 `caption` 就是条件文本：

```json
{"caption": "a red dog", "image": "/data/.../0001.png"}
```

空行会跳过。caption 去空白后不能空。同一对 caption 和图片不能重复。四个实例都要能打开这些图片，所以图片也在 `/data`。这条入口没有验证集，也没有出图。

`run.json` 记录 `train_pairs`、`text_field`（语料是 `goal_prompt`，JSONL 是 `caption`）、`source_kind` 和 `corpus_fingerprint`。

`RUN_DIR` 是一次新运行的空目录，放在 `/data`。目录已有文件时，rank 0 会拒绝启动。恢复时目录里可以已有这次运行的产物，并设置 `GEN_ONLY_RESUME` 指向其中的 `step_XXXXXX.pt`。

## 启动命令

先确认这 64 张卡是留给这次运行的。下面这条命令要在 4 个实例上各执行一次。可以写进作业启动命令，也可以在 worker-0 上通过 2222 端口分别登进去执行。

```bash
export GEN_ONLY_TRAIN=/data/zyd_workspace/bagel-LatentCoT/datasets/observe_feedback_corpus
export RUN_DIR=/data/绝对路径/gen_only_run
bash /data/zyd_workspace/bagel-LatentCoT/code/bagel-LatentCoT-anchored/scripts/anchored_gen_only_64.sh
```

平台已经注入 `VC_WORKER_HOSTS` 和 `MA_NUM_HOSTS`。不要改掉它们，也不要设置 `BAGEL_ALLOW_LEGACY_MEMORY_SFT`。

可选变量：

| 变量 | 作用 |
|---|---|
| `GEN_ONLY_RESUME` | 恢复用的 `step_XXXXXX.pt` |
| `GEN_ONLY_LOOP_CONFIG` | 默认是树里的 `configs/anchored_gen_only.json` |
| `GEN_ONLY_TRAIN_CONFIG` | 默认是树里的 `configs/anchored_gen_only_train.json` |
| `MODEL_PATH` / `MODEL_PYTHON` | 覆盖上表的权重和解释器 |
| `MASTER_PORT` | 默认 `29500` |
| `MASTER_TRAIN_INDEX` | 默认 `0`，即 `VC_WORKER_HOSTS` 的第一项 |
| `NPROC_PER_NODE` | 默认 `16`。改成别的数会使世界大小离开 64，脚本会退出 |

默认训练配置是 500 步、全局 batch 64、基础学习率 `1e-5`、weight decay 0、梯度裁剪 1、warmup 20、保存步 100 / 250 / 500、种子 17、分辨率 512、采样 50 步、timestep shift 3.0。全局 batch 必须是 64 的正倍数。标量 `gate.raw` 仍从 0 初始化，单独一组，学习率是 `max(1e-5 × 50, 5e-4) = 5e-4`，这一组的 weight decay 是 0。warmup 乘的是这一组自己的学习率。这不保证 gate 落到某个区间。

循环配置是 `variant=gen_only`、`correction_rounds=1`、最终速度损失。生效窗口是进度 `u ∈ [0.15, 0.65]`，也就是 `t ∈ [0.35, 0.85]`。训练按原生 timestep shift 在这个窗口里连续抽 `t`，每一次都会进入修正。推理仍走 50 步原生网格，窗口外的节点保持原生前向。Body 区间仍是候选 `[20, 28)`。

## 跑起来之后看哪里

rank 0 在 `RUN_DIR` 写 `run.json`、`train.jsonl` 和 `step_XXXXXX.pt`。`run.json` 里的 `time_window` 是训练抽样区间。`scheduled_flow_times` 是落在这个窗口里的原生采样节点，供推理对照。`train.jsonl` 每一行有 loss、flow、`rounds_error_fell`、gate、梯度范数、基础学习率和 `gate_learning_rate`。

64 个进程各自写 `RUN_DIR/logs/rank_00.log` 到 `rank_63.log`。这些文件在共享盘上，从 worker-0 就能看。没有 wandb。

checkpoint 含 gate、adapter、updater、优化器和采样器状态，并带 `variant=gen_only`。原生 BAGEL 不在这份权重里。

## 启动前的失败

脚本在组进程之前就会退出的情况：

- 没有 `VC_WORKER_HOSTS`、`GEN_ONLY_TRAIN` 或 `RUN_DIR`
- `hostname` 对不上 `VC_WORKER_HOSTS` 里任何一项（匹配规则是「主机名加一个点」开头）
- `MA_NUM_HOSTS * NPROC_PER_NODE` 不是 64

进程组起来之后还会拒绝：单进程启动、世界大小不是 64、输出目录非空、JSONL 不合法、语料没有 generation/train、配对 prompt 为空、循环配置不是 `gen_only`。
