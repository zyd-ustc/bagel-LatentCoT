# 全量静态 / 动态替换：R1、R2、R3

2026-10-06，main。正式评测由用户执行。源码快照：`/private/yida_workspace/bagel-LatentCoT-main-full-memory-20261006`；提交和源码 hash 见该目录的 `SYNC_COMMIT`、`SYNC_SOURCE_SHA256`。

## 本轮定义

全量 Memory 的长度严格等于每个样本的完整 prompt cache 长度。保存全部 token，包括原生特殊 token，保留顺序、原始 RoPE 位置和 packed sample split。没有均匀抽取、pooling、截断或 K=8 上限。全量模式忽略 `memory_slots`，即使它等于 0，也不关闭全量 Memory；R=0 仍走原生 bypass。

`LAYERWISE_FULL_SEED_REPLACE`：每层 Memory KV 原样复制该层完整 P。没有 writer。第 0 轮 GEN body 仍读 P，后续 GEN body 和最终 suffix 读完整 Memory，不叠加 P。每轮 GEN 恢复相同入口，所以静态替换在所有 R 下都应与 Base 完全一致。

`LAYERWISE_FULL_MEMORY_REPLACE`：writer query 是完整 prompt 在 body 入口的 native UND hidden，包含所有 token。writer 按原生层序读取完整 P、上一轮同层 M，以及当前 GEN 的 body 输入 KV，写出同层原生 UND 输入 KV。读回 GEN 前，特殊 token 的 K/V 槽位恢复为原生 P 的 K/V；只替换内容 token 的 K/V。writer 内部 hidden 不额外改写，因此这不是对全部 UND 中间状态施加冻结约束。

第一层 writer 输入尚未观察图像，GEN 始终读取该层静态完整 P 的副本。其他 body 层读取上一轮同层动态 M。最后一次 writer 继续经过 suffix，生成对应层 M；suffix writer 不额外计算 GEN，不读取尚未产生的 suffix GEN KV。最终 GEN suffix、norm、velocity head 各执行一次。

默认 body [0,8)，R 是额外 GEN body 次数。R1/R2/R3 分别执行 2/3/4 次 GEN body、1/2/3 次 UND writer body，均只额外执行一次 UND writer suffix。所有采样步启用；Memory 不跨 denoiser call、样本或 CFG 分支累积。text-removed CFG 走原生路径。窗口、writer 读取结构、CFG 和采样器均延续上一版。

## 与上一版的比较边界

本轮恢复的是完整条件容量、序列位置和特殊 token KV，并保留动态更新能力。原始 prompt cache 和权重不变，没有 adapter、gate、alpha、shuffle 或训练接口。

writer 仍使用新开放的 UND→GEN 反馈通路，仍有固定 P、上一轮 M 和当前 writer token KV 的同位置多份状态，也仍使用 noncausal writer attention。全量替换并不证明这条读取结构已处于原生训练分布。本轮不同时修改它，以免混入额外变量。

静态 / 动态比较在 GEN 读取长度、位置、body 次数和 suffix 次数上匹配；动态版本额外执行 writer，不能当作等耗时比较。动态 K/V 改变或速度改变只证明有作用，不证明语义修复或质量保持。

## 用户运行

默认 hard16 的前 8 个 prompt、seed 0。共 7 路 × 8 张 = 56 张：Base、静态 R1/R2/R3、动态 R1/R2/R3。Base 只生成一次，每个 worker 只加载一次生成模型。8 卡时每个 worker 负责 1 个 prompt 的全部 7 路。

```bash
cd /private/yida_workspace/bagel-LatentCoT-main-full-memory-20261006
export GPUS=0,1,2,3,4,5,6,7
export MODEL_PYTHON=/private/software/conda/envs/lcot/bin/python
export MODEL_PATH=/private/yida_workspace/models/BAGEL-7B-MoT
export RUN=/private/yida_workspace/outputs/full_memory_depth_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
set -o pipefail
LOOP_DEPTHS=1,2,3 MAX_PROMPTS=8 SEEDS=0 START_LAYER=0 END_LAYER=8 \
PROGRESS_START=0 PROGRESS_END=1 \
  bash scripts/evaluate/run_full_memory_depth_8gpu.sh "$RUN" \
  2>&1 | tee "$RUN/driver.log"
```

卡必须是已分配给实验的空卡。`GPUS` 也可指定 4 张卡。将 `MAX_PROMPTS=16` 可运行完整 hard16，共 112 张。若选 8 卡，至少保留 8 个 prompt×seed 工作项，以免出现空 worker。

脚本按顺序执行：

1. 在首张选定卡上运行真实权重 E0。检查 R0、静态 R1/R2/R3 的 conditional 与 CFG velocity parity，以及动态 R1/R2/R3 的 max_abs / relative_l2。输入是两条固定 prompt、不同分辨率、固定高斯张量，测试 timestep 为 0.7 和 0.3；它不是实际采样轨迹，不能用于解释真实采样各阶段的扰动幅度。
2. 静态速度必须完全相等，动态速度必须有限，prompt cache 必须不变，否则停止，不生成图片。动态是否非零另行记录，不设人为幅度阈值。
3. 在所选卡上生成七路配对图片。
4. 检查每个静态 R 的完整 PNG hash 是否与 Base 相同。若不相同或出现无效文件，停止，不继续评分。
5. 评分并合并配对报告。比较每路与 Base、每个动态 R 与同深度静态 R，以及同一路径的较浅 R。
6. 导出内嵌全部图片的 `comparison.html`。

输出：

- `$RUN/full_memory_e0.json`：静态速度 parity 与动态速度差。
- `$RUN/full_static_image_parity.json`：静态图片是否原样复现 Base。
- `$RUN/quality_report/summary.md`：语义、质量代理、Repair/Damage、置信区间和工程耗时。
- `$RUN/comparison.html`：七路配对图片，点击可查看原尺寸，可下载离线打开。

本轮尚无真实权重结果。不要用 CPU 检查通过代替上述 E0 和图片结果。
