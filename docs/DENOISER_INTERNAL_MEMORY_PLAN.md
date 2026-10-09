# 分层 UND Memory loop：loop-layer-NPU

## 固定语义

从 NPU 提交 13622854dcc9fb0a44532a337c7dce036060b1e9 提取旧路径。核心代码是 qwen_latent_cot/bagel/und_state_loop.py。本次分支不修改其计算。

- BAGEL 保持原生 28 层 MoT、UND/GEN 专家、归一化、投影、RoPE 和 flow sampler，权重冻结。
- 每个活跃去噪步，从原 prompt 的各层输入 hidden 初始化独立 Memory；容量为完整 prompt，不压缩。Memory 不跨去噪步保存。
- 窗口 `[0,8)` 内，每层独立更新：H_l^(r+1) = UND_l(H_l^r; P_l, GEN_l^r KV, live self KV)。原 P_l 固定；特殊 token hidden/KV 固定为原生值。
- Round0 GEN 读取原 prompt KV；后续 GEN 读取 Memory KV，替换原 prompt KV。每轮 GEN 从同一窗口入口 hidden 重算，不递推上轮 GEN 输出；各轮使用相同 x_t/t。
- 更新后的 H_l 经过同一层原生 UND norm/K/V/RoPE 投影，供下一轮同层 GEN 读取。保存的 GEN KV 只用于 UND 更新，不直接供下一轮 GEN 读取。
- 最后一次 writer 从窗口末层状态继续经过 UND 8–27 层一次，写入各层原生 attention 输入 KV。这一段不读取 GEN KV，也不回送前8层。
- 最终 GEN 经过8–27层一次，读取更新后的 Memory KV。null-text CFG 走原生路径。没有 adapter、gate、输出 alpha、视觉理解回灌或文本解码。
- R2 表示前8层执行3轮 GEN、2次 Memory 更新；每个活跃去噪步仅推进一次原生采样。

## ModelArts 执行

SSH 为 modelarts-job。代码目录 `/root/bagel-LatentCoT-loop-layer-NPU`，分支 loop-layer-NPU。继承已有 Ascend SDPA、packed 变长 attention 和 BF16 适配，不承诺跨 CUDA/NPU 逐位一致。

正式评测由用户启动。下面默认32题、9组、288张图，使用16个逻辑芯片。完整800题将 CONFIG 改为 configs/window_comparison.json。

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
