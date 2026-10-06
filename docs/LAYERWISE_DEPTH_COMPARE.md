# 分层 Memory KV：R1 / R2 / R3 对比

日期：2026-10-06。本轮只修改评测接口，不修改 denoiser、Memory writer、原生层计算或权重。旧 R1 结果 `/private/yida_workspace/outputs/layerwise_kv_small_20261006_041201` 保留，不混入新目录统计。新目录重跑 Base/R1 以保证版本、输入和采样配置一致。

## 问题与判断

问题：增加 Memory 读写轮数，是否让目标约束得到修正，还是反复强化错误？

同16个hard prompt、seed0，Base/R1/R2/R3各16张，共64张。固定512px、50时间点、shift3、CFG4、窗口[0,8)、K≤8、全部采样步。Base只生成一次，每个worker只加载一次原生BAGEL。R表示额外GEN body轮数；R2有3次GEN body和2次UND writer，R3有4次GEN body和3次UND writer。suffix仍一次。

报告每个R对Base，以及R2/R3对较浅R的配对GM、quality proxy和Repair/Damage。重点看此前“六个行李箱变三个”等明确计数错误能否恢复，不将所有评分阈值翻转当作人工确认。小样本用于观察深度趋势，不证明总体效果。若加深只增加错误或画面简化，不把更大R作为训练理由；若改善，仍需分清信息反馈与prompt重新加权。

## 用户执行

代码快照：`/private/yida_workspace/bagel-LatentCoT-main-layerwise-depth-20261006`。正式生成、评分由用户启动。

```bash
cd /private/yida_workspace/bagel-LatentCoT-main-layerwise-depth-20261006
export GPUS=0,1,2,3,4,5,6,7
export RUN=/private/yida_workspace/outputs/layerwise_kv_depth_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
set -o pipefail
LOOP_DEPTHS=1,2,3 SEEDS=0 START_LAYER=0 END_LAYER=8 MEMORY_SLOTS=8 \
PROGRESS_START=0 PROGRESS_END=1 \
PROMPTS=/private/yida_workspace/umm-anchored-eval-tools-d126833/data/hard16.jsonl \
  bash scripts/evaluate/run_layerwise_depth_8gpu.sh "$RUN" \
  2>&1 | tee "$RUN/driver.log"
```

结果：`$RUN/quality_report/summary.md` 和 `$RUN/comparison.html`。HTML包含四列配对图片，所有图片内嵌，可离线查看。新 depth driver 不导出 Memory QA，也不训练。不要在旧已完成R1目录中恢复运行。

## Memory 信息增益的候选方案：尚未实现

1. **目标明确的writer query。** 当前均匀选择prompt内容token，未指定slot要检查哪个对象、数量或关系。可使用原生UND短问题作为writer的语义query，例如询问当前行李箱数量和上下关系；仅writer读问题上下文，GEN仍通过同层Memory KV接收反馈。须保持预算并验证读到实际图像内容，而不是复述目标。不能由原生QA能力直接推断它能读noisy GEN。
2. **临时更新选定prompt KV。** 当前GEN读完整P再追加M；M使用所选prompt的原位置，因此可能同时引入反馈与重新加权文本。可在临时GEN缓存中，用对应层M替换所选P的位置，原P不修改，GEN位置与缓存长度不变。这样可减少重复条件加权的影响；也可能覆盖正确文本条件，需要单独验证。只改这一项，不能与深度或writer query一起改。
3. **选择更可信的采样阶段写入。** 原生训练mask阻止其他区段读取noisy image，UND读取高噪声GEN的语义能力未经训练证明。可以保持早期层窗口，另外单独测试在中低噪声阶段启用写入。它与前移/后移层窗口是不同变量，也不加入实时ViT反馈。

BAGEL原生think生成规划文本后，将新文本写入上下文再条件化生成，说明原生UND文本KV是已有的理解→生成接口；这不证明任何新增latent Memory都有效。ViT只用作离线观察或后续监督。来源：[BAGEL inferencer](https://github.com/ByteDance-Seed/Bagel/blob/056b5fd51a88c1eb4547318609e25d40080fcf87/inferencer.py)、[原生attention mask](https://github.com/ByteDance-Seed/Bagel/blob/056b5fd51a88c1eb4547318609e25d40080fcf87/data/data_utils.py)。

## 分层KV与训练

KV是中间激活，并不天然阻断梯度。CPU合成检查验证writer seed→原生UND层KV→冻结GEN输出的反传，在R1/R2上均有有限非零梯度，且没有更新任何权重。这不证明CUDA推理kernel可直接训练。

当前loader冻结模型，generator使用no_grad，runtime明确拒绝training模式；因此现有代码仍是推理入口。训练需独立forward和attention mask、保留Memory写入/读取的计算图、处理多轮激活存储，并检查CUDA backward。不能把native inference cache接口直接当作完整训练接口。

冻结GEN参数不等于把GEN forward放进no_grad：要把梯度传回Memory，仍需经过GEN的读取与后续层，并保存或重算激活。R增加时训练开销会增长；slot少不能保证训练显存很低。

困难主要来自反馈的语义可靠性、深度/模态分布差异和监督滞后，不是字典按层存KV。可借鉴[Looped-DiT deep supervision](https://github.com/OpenSenseNova/Looped-DiT/blob/92a9c1914258361f78e03426588c7c215944b504/looped_dit/diffusion.py)：训练时监督多个退出深度；普通推理仍只执行最终readout。但它的flow target与BAGEL不同，不能照搬训练分布或损失定义。

若未来满足training-free有效这一前提，优先考虑理解侧格式/隐式适配，监督“当前画面实际有什么”，并区分目标与观察。只用prompt正确答案会鼓励文本回声。更新原生UND权重也会影响prompt prefill和GEN边界，冻结GEN专家不等于完整Base不变；需要保留原生分支证据。此轮没有实现或执行训练。
