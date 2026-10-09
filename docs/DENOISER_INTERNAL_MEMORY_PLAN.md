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

## ModelArts 执行

SSH 为 modelarts-job。代码目录 `/root/bagel-LatentCoT-loop-layer-NPU`，分支 loop-layer-NPU。继承已有 Ascend SDPA、packed 变长 attention 和 BF16 适配，不承诺跨 CUDA/NPU 逐位一致。

正式评测由用户启动。下面默认32题、9组、288张图，使用16个逻辑芯片。完整800题将 CONFIG 改为 configs/window_comparison.json。旧路径参照使用 configs/loop_layer_npu_legacy_pilot.json。

```bash
cd /root/bagel-LatentCoT-loop-layer-NPU
npu-smi info
export BACKEND=npu
export NPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export OMP_NUM_THREADS=4
export CONFIG="$PWD/configs/loop_layer_npu_pilot.json"
unset COMPARISON_PROMPTS
mkdir -p /root/outputs
export RUN=/root/outputs/loop_layer_npu_$(date +%Y%m%d_%H%M%S)
set -o pipefail
bash scripts/compare_windows_8gpu.sh "$RUN" 2>&1 | tee "${RUN}.log"
```

脚本绑定源码、权重、数据与配置；真实权重检查通过后生成、评分，再输出 comparison.html 和 quality_report。质量评分是 VLM 代理；GM 上升不能直接认定质量或净 Repair 改善。正式结果尚需本分支命令执行。
