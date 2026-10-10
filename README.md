# BAGEL：每层同步 GEN／UND 微循环

本分支 `loop-layer-joint-micro` 从 `loop-layer-NPU@42e7114` 创建。新路径在同一层维护 GEN／Memory 两份连续状态。每个微轮都读取轮初的原生 KV，分别运行 GEN／UND block，再以 `1/K` 更新两份状态并传给下一微轮／下一层。GEN 读取 Memory；UND 读取固定 prompt、轮初 GEN 和 live self KV。窗口后每层执行一次同步更新，最终只读出一次 velocity。

新方案见 [JOINT_MICRO_LOOP.md](docs/JOINT_MICRO_LOOP.md)。没有 adapter、压缩或训练。K 是窗口内每层总计算次数；K1 已经包含 Memory 反馈，不等于 Base。语义编辑、Repair 和质量收益尚未验证。

默认新实验 `configs/joint_micro_pilot.json`：32题、seed0、Base＋旧 Legacy Early20 R2＋Joint Early20 K1/K2/K4，共160图。仍使用 `[0,8)` 与原生 shifted sampling；正式评测由用户启动。

```bash
cd /root/bagel-LatentCoT-loop-layer-joint-micro-20261010
export BACKEND=npu
export NPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CONFIG="$PWD/configs/joint_micro_pilot.json"
export OMP_NUM_THREADS=4
unset CUDA_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES COMPARISON_PROMPTS
export RUN=/root/outputs/joint_micro_$(date +%Y%m%d_%H%M%S)
set -o pipefail
bash scripts/compare_windows_8gpu.sh "$RUN" 2>&1 | tee "${RUN}.log"
```

脚本先执行真实权重数值检查，再生成、打分并导出 `comparison.html`。也可使用 `BACKEND=cuda`／`GPUS` 在 H200 运行。旧实现和旧配置保留如下，不能用其 R 标签解释新路径的 K。

## 起点实现与历史结果

起点 main／loop-layer-NPU 的测试路径为 `full_depth_restart`：每轮从原生第0层 hidden 出发，Memory 连续经过 UND 0–27 层。前一层输出进入下一层；末层 hidden 不回送首层。轮间信息通过 `M1 → GEN1 → GEN1 KV → writer2 → M2` 传递。GEN 每轮重算 `[0,8)`，最后执行一次 GEN suffix。0–7层保留更新后 hidden 的 KV 投影，8–27层保留原生 attention 输入 KV。

保持原生 BAGEL 权重、MoT、全量 prompt、位置、特殊 token、CFG 和采样。不添加 adapter、gate、alpha、压缩或训练。该方案不使用 ViT，也不等价于原生图像理解。

本地 CPU 数值测试40项通过；7项 NPU 测试因无设备跳过。新路径 R1 与原 full_depth 的 R1 数值一致。测试确认第二轮 UND 更新依赖 GEN1 KV；语义和画风改善尚待正式评测。

方案见 [DENOISER_INTERNAL_MEMORY_PLAN.md](docs/DENOISER_INTERNAL_MEMORY_PLAN.md)。默认小规模配置 `configs/repeat_r1_pilot.json`：32题、seed0、Base＋Early10/Early20×R1/2/3/4，共288图。完整配置 `configs/window_comparison.json`：800题，共7200图。两者均使用 full_depth_restart。保留 full_depth 末层回送和 legacy_layerwise 独立层更新作为显式参照。

## H200 用户运行

正式评测由用户启动。下面目录为本次同步的 main 快照；GPUS 是该节点的本地卡号。

```bash
cd /private/yida_workspace/bagel-LatentCoT-main-repeat-r1-20261009
export BACKEND=cuda
export GPUS=0,1,2,3,4,5,6,7
unset NPUS ASCEND_RT_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES COMPARISON_PROMPTS
export OMP_NUM_THREADS=4
export CONFIG="$PWD/configs/repeat_r1_pilot.json"
export RUN=/private/yida_workspace/outputs/repeat_r1_h200_$(date +%Y%m%d_%H%M%S)
set -o pipefail
bash scripts/compare_windows_8gpu.sh "$RUN" 2>&1 | tee "${RUN}.log"
```

脚本先执行真实权重数值检查，通过后生成、打分和导出 comparison.html。需要使用新输出目录。分页 HTML 依赖 gallery/，本地查看须下载整个结果目录。GPU 时间仅为工程记录。NPU 仍支持 BACKEND=npu 和 NPUS，执行说明见方案文档。

## 原生图像观察路径（显式配置保留）

原生观察路径使用 `configs/observation_comparison.json`，不由上述默认配置启动。其计算方式为一次早期图像预测→完整原生 UND 更新→固定 x_t/t 重算 GEN。没有完整文字反馈解码。

首轮默认8题、seed0、5组，共40张图：Base，step9／step19分别各一个STATIC和OBSERVED。每张非Base图只更新一次。两个组保留相同视觉条件、文字token、位置、容量和GEN CFG，区别只在文本Memory编码时是否读取图像。

文本Memory经过全部28层UND连续计算，KV从各层attention输入自然写入。最终GEN使用完整图像VAE＋ViT上下文和文本Memory；原生准备接口重建位置和CFG。重算沿用当前噪声和t，随后原始采样器只推进一次。缓存不跨步。

每次更新保存早期预测图、x_t/t、前后velocity、全部文本Memory KV及四类短问答probe。问答通过缓存副本执行，仅供诊断，不回流到生成。问答正确性待人工标注；memory-only读取明确为非原生诊断。

## 历史证据

此前完整800题、seed0的7200张评测已完成。浅层独立UND循环在R1／R2有小幅正趋势，但净Repair区间跨零；R3／R4明确增加Damage。Early20 R4为412 Repair／1064 Damage。

## Memory 增量定位

NPU诊断使用 `scripts/diagnose_increments_npu.sh` 和 `configs/increment_diagnostics.json`。默认32条真实Early20 R4轨迹，在step0/4/9/19固定x_t/t比较R0–4，记录GEN/Memory逐层变化及条件/CFG velocity。原生采样每步仍只推进一次。执行命令与字段说明见 [INCREMENT_DIAGNOSTICS.md](docs/INCREMENT_DIAGNOSTICS.md)。
