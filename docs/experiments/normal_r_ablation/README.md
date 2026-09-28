# Normal-only R 消融：8 hard prompts

当前入口仍是 `scripts/evaluate/bagel_memory_mechanism.py` 与
`scripts/evaluate/run_bagel_memory_mechanism.sh`，但实验 schema 已变为
`bagel-normal-r-ablation-v1`，只运行 `normal_r2/normal_r4/normal_r6/normal_r8`。
不生成 native、static-null、zero-init、shuffled、frozen 组。历史底层 helper 和
回归测试保留，当前评测入口不能选用它们。旧 hard64 产物不会被修改或混入新结果。

## R 的准确含义

| R | strict Read | Write | body 总执行次数 |
| --- | ---: | ---: | ---: |
| 2 | 1 | 1 | 2 |
| 4 | 1 | 3 | 4 |
| 6 | 1 | 5 | 6 |
| 8 | 1 | 7 | 8 |

每个 timestep：prefix 一遍 → Read → R−1 遍 Write → suffix 一遍。
每次 Write 都将所有非 memory hidden 恢复为最初 body 入口的状态，只传递上一轮
更新后的 memory。只用最后一遍 Write 经 suffix 得到的速度做一次 Euler 更新。
每个 timestep 重新初始化 memory；不将上一 timestep 的 memory 带入下一步。
K=8、body=[12,20)、prompt KV 全程可见；CFG 条件/无条件分支独立执行同一个 R。

默认从 `geneval2_hard_128.jsonl` 取前 8 条，正好对应上一轮 hard64 的前 8 条。
保留旧 noise schema 和 seed=42，因此同一 prompt 的初始噪声不随 R 改变。
默认 512×512、50 个时间点=49 次去噪、shift=3、原 CFG 配置不变。

## 指标与输出

同状态 probe 参考轨迹改为 **normal/R2**：每步四个 R 都接收相同的 R2 x_t，
只有 R2 推进参考轨迹。最终 R2 图直接复用该轨迹结果；R4/6/8 各自从原始噪声
独立生成，互不串联。不会为了诊断额外执行 native。

- `index.html`：四列图像，默认共 8×4=32 张/seed。
- `*_trace.jsonl`：`v_r2_norm`、`relative_r4_vs_r2` / `r6_vs_r2` / `r8_vs_r2`，
  相应绝对速度差、body/suffix hidden 差、各 Write 的 memory 入口范数。
- `attention.jsonl`、`memory_checks.jsonl`：带 `round` 字段，Read=0，Write=1…R−1。
- `reference_states/step_*.pt`：R2 参考状态；不再叫 `base_states`。
- `run_manifest.json`：模式、R 列表、配置、源码哈希、noise seed/hash、
  `pixel_mae_vs_r2` 和完成状态。四个 R 的完整性会在合并时检查。

相对指标分母是 `||v_R2||+1e-12`，不能与旧报告的 native 分母指标直接相减。
Pixel MAE 是行为差异，不是质量。更大 R 的计算量更高；运行时长尚未在 GPU 实测。

## H200 命令（更新远端代码后，由用户执行）

先进入 `/private/yida_workspace/bagel-LatentCoT`。启动器保留 CUDA 可见范围，不会
擅自开放其他卡。按 prompt 对分片：默认 8 条最多 4 个 worker；只有 1/2 张可见卡
时自动缩小并行度。可以用 `NUM_SHARDS` 进一步限制，但不能超过可见卡数或 4。

1. 单卡短程 smoke（2 条 prompt、3 个时间点，四个 R 都覆盖）：

```bash
cd /private/yida_workspace/bagel-LatentCoT
PYTHON_BIN=/private/software/conda/envs/lcot/bin/python \
MODEL_PATH=/private/yida_workspace/models/BAGEL-7B-MoT \
MAX_PROMPTS=2 NUM_SHARDS=1 NUM_STEPS=3 OMP_NUM_THREADS=1 \
bash scripts/evaluate/run_bagel_memory_mechanism.sh \
  /private/yida_workspace/outputs/normal_r_smoke_$(date +%Y%m%d_%H%M%S)
```

2. smoke 通过后跑 8 条正式评测（自动选最多 4 张可见卡）：

```bash
cd /private/yida_workspace/bagel-LatentCoT
OUTDIR="/private/yida_workspace/outputs/normal_r_hard8_$(date +%Y%m%d_%H%M%S)"
mkdir -p /private/yida_workspace/outputs
nohup env \
  PYTHON_BIN=/private/software/conda/envs/lcot/bin/python \
  MODEL_PATH=/private/yida_workspace/models/BAGEL-7B-MoT \
  BACKEND=cuda BENCHMARK_DATA=experiments/data/geneval2_hard_128.jsonl \
  MAX_PROMPTS=8 SEEDS=42 NUM_STEPS=50 HEIGHT=512 WIDTH=512 OMP_NUM_THREADS=1 \
  bash scripts/evaluate/run_bagel_memory_mechanism.sh "$OUTDIR" \
  > "${OUTDIR}.launcher.log" 2>&1 < /dev/null &
echo "PID=$! OUTDIR=$OUTDIR"
```

输出目录必须不存在；任一分片失败会停止合并并保留日志。不得拿旧六组输出目录
运行新入口的 `--merge-only`，schema 检查会拒绝覆盖。若手动合并，使用与启动时
完全相同的 benchmark、prompt 数量、seed、时间步、分辨率和分片数。

## 本地验证与当前边界

验证包括 R2/4/6/8 的调用顺序、memory 递推、非 memory reset、suffix 一次、
R2 数值一致性、四个 R 与原同层循环的一致性、CFG 两分支传参、同状态探测、
独立噪声起点、多分片/多 seed 的四列产物链以及旧 schema 拒绝。
2026-09-27 全量本地 CPU 测试：226 项通过；启动器语法、Python 编译和默认
8 prompt / 4 分片 dry-run 通过。尚无真实 R4/6/8 GPU 运行结果。
本轮只修改本地 v1 并做 CPU 测试；未上传、未推送、未启动 H200/NPU 实验。
执行状态见 `CHECKLIST.md`；实验边界见 `PLAN.md`。
