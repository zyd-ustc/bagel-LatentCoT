# BAGEL：原生观察驱动的 denoiser 内部 Memory

**本页对应 NPU 分支。** 运行后端与检查状态见 [NPU_RUNTIME.md](docs/NPU_RUNTIME.md)。

当前默认方案是 **一次早期图像预测→完整原生 UND 更新→固定 x_t/t 重算 GEN**。没有完整文字反馈解码、adapter、gate、压缩或训练。正式质量效果尚未验证。

原main的22项CPU数值测试保留。NPU分支增加设备、attention和RNG隔离检查，真实权重数值检查与正式效果的状态分别记录在NPU_RUNTIME.md。

完整实现方案见 [DENOISER_INTERNAL_MEMORY_PLAN.md](docs/DENOISER_INTERNAL_MEMORY_PLAN.md)。这份文档对应当前代码；10月5日旧同层1/N方案不再代表实现。

首轮默认8题、seed0、5组，共40张图：Base，step9／step19分别各一个STATIC和OBSERVED。每张非Base图只更新一次。两个组保留相同视觉条件、文字token、位置、容量和GEN CFG，区别只在文本Memory编码时是否读取图像。

文本Memory经过全部28层UND连续计算，KV从各层attention输入自然写入。最终GEN使用完整图像VAE＋ViT上下文和文本Memory；原生准备接口重建位置和CFG。重算沿用当前噪声和t，随后原始采样器只推进一次。缓存不跨步。

每次更新保存早期预测图、x_t/t、前后velocity、全部文本Memory KV及四类短问答probe。问答通过缓存副本执行，仅供诊断，不回流到生成。问答正确性待人工标注；memory-only读取明确为非原生诊断。

## 用户运行

正式生成与评分由用户启动。脚本先绑定源码／权重／数据，再执行真实权重数值检查，通过后生成40张图、评分并导出HTML。默认配置为configs/observation_comparison.json。

```bash
cd /root/bagel-LatentCoT-NPU
export BACKEND=npu
export NPUS=0,2,4,6,8,10,12,14
export OMP_NUM_THREADS=4
export CONFIG="$PWD/configs/observation_comparison.json"
unset COMPARISON_PROMPTS
mkdir -p /root/outputs
export RUN=/root/outputs/und_native_observation_npu_$(date +%Y%m%d_%H%M%S)
set -o pipefail
bash scripts/compare_windows_8gpu.sh "$RUN" 2>&1 | tee "${RUN}.log"
```

在modelarts-job上执行。当前8张物理卡各有2个设备；上述命令每张卡选择一个Phy-ID。NPUS控制ASCEND_RT_VISIBLE_DEVICES，每个进程内部使用npu:0。运行前用npu-smi info确认分配和占用。仅保留上述两个比较脚本，不新增独立启动入口。结果在comparison.html、quality_report/和generation/worker_*/traces/；HTML内嵌早期图和probe。generation_seconds包含诊断，probe_seconds单独记录，不能作严格预算比较。

支持原配置／源码／权重／分片数量不变时RESUME=1续跑。通用PROMPTS环境变量不会影响数据；自定义数据须显式设置COMPARISON_PROMPTS。

## 历史路径与证据

此前完整800题、seed0的7200张评测已完成。浅层独立UND循环在R1／R2有小幅正趋势，但净Repair区间跨零；R3／R4明确增加Damage。Early20 R4为412 Repair／1064 Damage。历史结果不是新完整UND观察方案的验证。

旧runner和两份旧配置保留用于复现与数值参照，须显式指定window_comparison.json或feedback_comparison.json。当前默认路径不调用它们。旧冻结远端目录与结果保持原样。

本次修改前的完整tracked源码和Git bundle已备份到工作区older/before-native-observation-memory-20261009_151938/。
