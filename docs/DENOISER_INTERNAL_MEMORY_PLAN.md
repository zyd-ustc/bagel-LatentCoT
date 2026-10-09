# 分层 UND Memory loop：loop-layer-NPU

## 固定语义

基于旧loop提交a87fa44实现连续完整UND更新。旧实现保留为数值参照，默认评测配置memory_update=full_depth，核心实现full_depth_memory.py。

- 每个活跃去噪步，单份H从原prompt的第0层输入hidden初始化。Memory容量、位置和特殊token固定规则不变，不跨去噪步保存。
- 每轮writer从0到27层连续运行；第l层输出进入第l+1层，最后一层输出进入下一轮第0层。特殊token在各层恢复为该层原生prompt参考值。此次明确不处理深度错配。
- 每层读取固定原prompt P_l和当前Memory self KV；0–7层另读取当前轮GEN_l KV。8–27层没有GEN KV，保持原GEN窗口及suffix计算量，不额外完成草稿或加入ViT。
- 0–7层更新后的hidden用同一层原生UND norm/K/V/RoPE投影，保持旧body readout。8–27层从原生attention自然保存block输入KV，恢复旧suffix读出。UND输出继续进入下一层并回送下一轮；不使用末层投影复制到所有层。
- GEN每轮仍从同一个窗口入口重算0–7层：round0读取P，后续读取M并替换P。最后GEN只经过8–27层一次，读取最后一轮完整writer产生的对应层M。
- R2为3次GEN body、2次完整28层UND writer。原prompt cache不可变，null-text CFG和非活跃时间步保持原生路径，每个去噪步仅推进一次sampler。
- 保留BAGEL原生MoT、专家参数、归一化、投影、RoPE、全量prompt、BF16和原采样，不训练，不新增adapter、gate或alpha。完整UND遍历不是原生ViT图像理解等价路径。
- config.memory_update=legacy_layerwise选择旧路径；full_depth选择上述新路径。两个32题配置除该字段外完全相同，支持配对同seed比较。

## 重复 R1 的连续 writer：full_depth_restart

人工审查显示 full_depth 的 R1 保留画风，R2 发生画风和语义跳变，R3/4 后续变化较小。恢复 8–27 层原生输入 KV 后该现象仍存在。末层 hidden 回送首层是待验证原因，不是已确认原因。

新增配置 `configs/loop_layer_npu_restart_pilot.json`，只将 `memory_update` 改为 `full_depth_restart`。保留既有配置，便于复现。

- 每轮都从原生第0层入口 hidden 出发，连续经过 UND 0–27 层。各层输出进入下一层，不恢复旧的独立层状态。
- 不将第27层输出回送第0层。每轮 writer 的计算规则与当前 full_depth 的 R1 完全相同。
- 轮间反馈为 `M1 → GEN1 → GEN1 KV → writer2 → M2`。R1 更新通过 Memory 改变下一轮 GEN 的隐藏状态和 KV，再被下一轮 UND writer 读取。不显式累加 `H1−H0`，不新增残差、adapter、gate、alpha或压缩。
- 0–7层读取当前 GEN KV；8–27层没有 GEN KV。body 读出更新后 hidden 的 KV，suffix 读出原生 attention 输入 KV。原 prompt 锚点、special 固定、GEN 入口重置与最终 GEN suffix 一次执行均不变。
- R1 与原 full_depth 的 R1 应数值一致。R2 的第二轮 UND 入口应等于原生入口，而第二轮 writer 的结果应依赖 GEN1 KV。这些是数值合同，不代表语义或质量提升。
- R2 仍是3次 GEN body和2次完整 UND writer。计算量不因重置入口而减少。状态不跨去噪步保存。

该路径验证“重复原生入口的 R1 writer 能否保留画风，同时通过 GEN 反馈产生后续编辑”。它不保证直接累积 R1 的 hidden 增量，也不保证反馈不会衰减。正式评测由用户启动。

## ModelArts 执行

SSH 为 modelarts-job。代码目录 `/root/bagel-LatentCoT-loop-layer-NPU`，分支 loop-layer-NPU。继承已有 Ascend SDPA、packed 变长 attention 和 BF16 适配，不承诺跨 CUDA/NPU 逐位一致。

正式评测由用户启动。下面选择重复R1路径，32题、9组、288张图，使用16个逻辑芯片。完整800题将 CONFIG 改为 configs/window_comparison.json。旧路径参照使用 configs/loop_layer_npu_legacy_pilot.json。

```bash
cd /root/bagel-LatentCoT-loop-layer-NPU
npu-smi info
export BACKEND=npu
export NPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export OMP_NUM_THREADS=4
export CONFIG="$PWD/configs/loop_layer_npu_restart_pilot.json"
unset COMPARISON_PROMPTS
mkdir -p /root/outputs
export RUN=/root/outputs/repeat_r1_$(date +%Y%m%d_%H%M%S)
set -o pipefail
bash scripts/compare_windows_8gpu.sh "$RUN" 2>&1 | tee "${RUN}.log"
```

脚本绑定源码、权重、数据与配置；真实权重检查通过后生成、评分，再输出 comparison.html 和 quality_report。质量评分是 VLM 代理；GM 上升不能直接认定质量或净 Repair 改善。正式结果尚需本分支命令执行。
