# 每层同步 GEN／UND 微循环


## 计算规则

`G` 是原生生成 query，包括 VAE latent token 和两个图像边界 token。`M` 保留原 prompt 的全部 token 槽位，hidden 维度采用 BAGEL 原生 hidden size；各样本可具有不同长度。固定的 `P_l` 是原生文本 prefill 在第 l 层写入的 prompt KV。

在活跃去噪步，GEN 前缀 `[0,start_layer)` 执行原生计算。Memory 从原生 prompt 的 `start_layer` 输入 hidden 初始化一次。之后两个状态连续进入各层，既不在每轮重置，也不将末层回送入口层。

窗口 `[start_layer,end_layer)` 内，每层执行 K 次同步微更新，步长 `h=1/K`。定义 `B_l` 为包含 attention 和 MLP 两次原生残差的完整 block，`R_l(Z)=B_l(Z)-Z`。

每个微轮 k：

1. 从轮初的 `G_l^k` 和 `M_l^k`，使用对应原生专家、norm、K/V 投影、K norm 和 RoPE 计算同层输入 KV。
2. GEN block 读取 `M_l^k KV` 和当前 GEN self KV，不再同时读取 P。
3. UND block 读取固定 `P_l`、`G_l^k KV` 和当前 Memory self KV。
4. 两个 block 完成后，同时更新：

   `G_l^(k+1) = G_l^k + h * (B_GEN(G_l^k; M_l^k KV) - G_l^k)`

   `M_l^(k+1) = M_l^k + h * (B_UND(M_l^k; P_l, G_l^k KV) - M_l^k)`

5. 最后两份 hidden 进入下一层。UND 不会在本微轮读取已经更新的 GEN 输出。这是同步更新，不是 GEN→UND 的顺序更新。

`end_layer` 之后，两份状态继续前进，每层各执行一次同步原生 block（步长 1）。因此 GEN suffix 仍只执行一次，但这条路径的 suffix UND 能读取当前同层 GEN KV。它不再使用旧路径“先完整 writer、再单独 GEN suffix”的顺序。

KV 均来自微轮的层输入；不从完整 block 输出再用同层 norm 投影。GEN 分支内的图像边界 token 保留原生 UND 专家处理。Memory 的特殊 token hidden 和 KV 继续固定到同层原生 prompt 参考值；仅这些锚点不参与状态更新。下一层的内容 token 精确接收上一层最终输出。

P、原生 hidden seeds、权重和输入 x_t 均只读。Memory 仅活在本次 denoiser 调用中，不跨去噪步保存。text-removed CFG 和非活跃去噪步直接走原生路径。sampler 每步仍只推进一次，final norm／llm2vae 只读出一次。

## K 与旧 R 的区别

- K 是每层总计算次数。K4 每次使用 0.25 的完整 block 残差。没有先执行一次完整 block，再追加 K 次。
- 旧 R 是额外窗口遍历次数，R2 有 3 次 GEN body。
- K1 在给定上下文下精确返回对应原生 block 输出，但整条 K1 pipeline 已引入 UND→GEN 条件反馈，因此不等于 Base。Base 单独保留。
- 残差步长总和为 1 不保证 K4 与原生单次输出相同。没有新增训练参数、adapter 或可学习 gate。
- 本路径没有 ViT 图像观察，也没有文字解码。更新后 Memory 是否包含可解释的错误信息，以及是否带来 Repair，仍需评测。

## 用户运行的配对测试

`configs/joint_micro_pilot.json`：32 prompts、seed0、512px、原生 50 点 shifted schedule（49 次 denoiser 调用），窗口 `[0,8)`，活跃 step `[0,20)`。

五组：Base、原 `legacy_layerwise` Early20 R2、Joint K1／K2／K4，共 160 张图。旧对照使用完整 prompt、原 KV 读出方式和旧 R 语义。K1／K2／K4 均采用本方案；K1 不是无反馈对照。该配置未加入固定 P 的纯 GEN 微循环，因此本轮不能单独归因 Memory 的质量收益。

`scripts/compare_windows_8gpu.sh` 先运行真实权重数值合同检查；通过后才生成、打分、导出 HTML。正式评测由用户运行。CPU／NPU 小模型数值测试可由开发者运行，不能据此声称生成质量改善。

数值检查覆盖：原生输入 KV 投影一致、UND 实际读取轮初 GEN KV、GEN 实际只读 Memory 条件、1/K 更新公式、层间状态传递、special 固定、cache／seed 不变、观察器与 KV 检查不改变 velocity、null-text CFG 原生绕过、非活跃时间步原生一致。`e0.json` 记录各微轮 GEN／Memory hidden 增量和同层输入 K 的变化。

每个活跃 conditional denoiser 调用的 block 次数：

|组|GEN block|UND block|
|---|---:|---:|
|Base|28|0|
|Legacy R2|44|36|
|Joint K1|28|28|
|Joint K2|36|36|
|Joint K4|52|52|

上述次数不包含文本 prefill、CFG null 分支、额外的原生 KV 投影和 decoder head。它们不是实际耗时比例。Joint K4 的 UND suffix 是 20 次，不是旧 full-depth 路径按 R 重复的完整 writer。

在 NPU 的本分支 checkout 中运行（卡号为当前节点本地逻辑编号）：

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

结果：`e0.json`、`generation/worker_*.log`、`quality_report/summary.md` 和 `comparison.html`。必须使用新的输出目录。开图时下载完整结果目录，包括 gallery 子目录。
