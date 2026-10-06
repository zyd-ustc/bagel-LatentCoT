# 用动态 Memory KV 替换 GEN 的 prompt 读取

目标是文生图 denoiser 内部的语义反馈。当前 append 路径同时给 GEN 完整 prompt P 和由 prompt 子集初始化的 M。替换路径要求反馈启动后 GEN 只能通过 M 获取文本条件，而 writer 保留完整 P 作为语义锚点。输出不同不代表语义改进；本实现尚无真实权重生图结果。

P_l 是原生 prompt 在第 l 层的 KV，只读。M_l^r 是第 r 次 writer 在同一层输入处产生的原生 UND KV。G_l^r 是第 r 次 GEN body 计算在同一层输入处产生的原生 KV。R 是额外 GEN body 次数，body 为 [s,e)，默认 [0,8)。K 是每个样本最多抽取的 content token 数，默认 8。

## 执行顺序

1. 原生 prompt prefill 不变。在 s 层入口保存均匀抽取的 content token hidden，保留原始 RoPE 位置；排除原生特殊 token。各层的对应 prompt KV 同时构成静态 seed bank S_l。
2. 原生 GEN prefix [0,s) 执行一次。GEN body 第 0 轮读取 P，保存每层实际输入 KV G_l^0。启动计算不执行 suffix/readout。
3. writer 从固定的 prompt seed hidden 开始，按原生层顺序走 body。第 l 层读取完整 P_l、上一轮 M_l（若有），以及刚完成的 G_l^r。保存实际 UND 层输入 KV 作为下一轮 M_l。writer 末端 hidden 不返回 s。
4. 下一轮 GEN 重置到固定 body 入口。每一层只读取同层 M_l，移除完整 P_l。s 层 writer 输入尚未看见图像，因此 GEN 在 s 层只读静态 S_s；后续层才读取动态反馈。
5. 重复步骤 3–4，直到完成 R 次额外 body。只保留最新 bank，读取长度不随 R 累积。
6. 最后一次 writer 额外遍历 [e,L) suffix，用完整 P 继续处理已获取图像信息的 UND hidden，保存这些层的 M_l。suffix writer 不额外执行 GEN，不读取尚未计算的 suffix GEN KV。
7. 最终 GEN suffix [e,L) 仅执行一次，每层只读取 M_l，随后执行原生 norm/head。GEN suffix 不重新读取完整 P。

默认 s=0 时，首次 body 启动之后，GEN 的所有层均不再直接读取 P。如果配置 s>0，prefix 仍会读取 P，且它的入口状态会间接携带 P；这是明确保留的原生路径，不应称为“全过程 GEN 完全不接触 prompt”。任何 R 下，首次 body 也明确读取 P。

## 两个新模式和现有对照

| 模式 | 后续 GEN body / suffix 读取 | writer |
|---|---|---|
| BASE | 原生完整 P | 无 |
| LAYERWISE_MEMORY_KV | body: P + 动态 M（s 层仍仅 P）；suffix: P | R 次 body |
| LAYERWISE_SEED_REPLACE | 各层静态 S，不读取完整 P | 无 |
| LAYERWISE_MEMORY_REPLACE | s 层静态 S；其他 body 层和 suffix 读取动态 M，不读取完整 P | R 次 body + 最后一次 suffix |

静态替换用于检查：压缩 prompt 或移除重复 prompt 读取本身是否改变结果。动态替换相对静态替换的语义收益，才是 GEN 反馈的直接证据。静态和动态的 GEN 读取长度、位置、body 次数、suffix 次数相同；动态路径增加 UND writer 计算，因此它们不是等耗时对照。与 append 的比较也同时改变读取拓扑和 writer suffix 深度，不能将结果只归因于某一个变化。

全部模式不创建参数、不修改权重、不覆盖 P，不加 adapter、gate、alpha 或 shuffle。Memory KV 经原生 UND 投影产生，GEN 继续使用原生 GEN attention、MLP、完整 residual。每个 denoiser call 和 CFG branch 单独初始化状态，不跨扩散 timestep 积累 M。

R=0、K=0、非激活 timestep 使用原生 bypass。text-removed CFG 使用原生路径。混合 batch 中没有 content seed 的样本保留完整 P，输出与该样本的原生路径一致。原生 GEN RoPE 和 SOI/EOI query 保留；被移除的是旧 prompt cache 的直接读取，包括其中特殊 token 的 KV。

## 原理和待验证点

这条路径使文本条件经过 Memory 后再进入反馈 GEN，避免同时读取 P 和由 P 派生的 M。但 K=8 是有损条件压缩，不是保证足够的信息容量。均匀 content token 不保证覆盖对象、数量和关系。原生 UND 是否能从 noisy GEN KV 写出可靠语义仍未证明。替换可能减少重复权重，也可能丢失条件并损伤已有正确结构。

原理依据是仓库内原生 BAGEL：`qwen2_navit.py` 的 native cache 写入保存 attention 的层输入 KV；UND / GEN 各自使用原生 MoT 投影；packed context 可以重新构造读取索引，RoPE 位置无需跟 cache 长度同步改变。此实现修改缓存可见性和执行顺序，没有宣称 BAGEL 已训练过这种新拓扑。

## 小规模正式评测：用户在 H200 执行

源码目录：`/private/yida_workspace/bagel-LatentCoT-main-memory-replace-20261006`。默认 hard16、seed 0、R1，四个路径共 64 张图。脚本完成生成、评分、配对报告和内嵌图片 HTML。需要 8 张已分配的空卡；也可以用 GPUS 指定 4 张。

```bash
cd /private/yida_workspace/bagel-LatentCoT-main-memory-replace-20261006
export GPUS=0,1,2,3,4,5,6,7
export MODEL_PYTHON=/private/software/conda/envs/lcot/bin/python
export MODEL_PATH=/private/yida_workspace/models/BAGEL-7B-MoT
export RUN=/private/yida_workspace/outputs/memory_replace_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
set -o pipefail
LOOP_DEPTHS=1 SEEDS=0 START_LAYER=0 END_LAYER=8 MEMORY_SLOTS=8 \
  bash scripts/evaluate/run_memory_replace_8gpu.sh "$RUN" 2>&1 | tee "$RUN/driver.log"
```

重点看动态替换 vs 静态替换的语义 GM、Repair/Damage，动态替换 vs Base 的质量保留，以及与原 append 的图片差异。先确认数量和空间关系的实际图像变化，再解释评分；VLM 质量分数只是代理。当前没有训练接口，CPU 可微性检查也不证明 CUDA 训练可用。替换的 probe 导出暂不支持，旧 append probe 不受影响。
