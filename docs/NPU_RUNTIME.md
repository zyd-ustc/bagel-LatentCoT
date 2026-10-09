# NPU分支：ModelArts运行适配

基于main提交79cb39b。只在NPU分支修改，main未改动。完整UND更新、单次固定x_t/t重算、STATIC／OBSERVED、问答生命周期、40图默认实验与评分定义保持原方案。

## 适配内容与来源

- 参考远端旧代码的qwen_latent_cot/bagel/accelerator.py，以及qwen2_navit.py的_sdpa_varlen_inference和siglip_navit.py的SDPA分支。
- 新accelerator.py统一显式设备、BF16 autocast、同步和峰值显存接口。auto选择可用后端；明确指定cuda或npu时，不静默切换。
- CUDA继续调用FlashAttention。NPU使用旧适配中的按样本拆分SDPA，保留packed长度、GQA、样本隔离和右下对齐causal mask。CPU使用FP32参考。日志明确区分三个后端，不把NPU写成FlashAttention。
- BAGEL权重严格加载。原生模型层、投影、RoPE、MoT分工、VAE采样和Memory容量均未替换。没有导入transfer_to_npu或全局改写torch.cuda。
- 原生准备接口生成的初始噪声仍被丢弃，GEN继续沿用当前x_t/t。没有生成新的采样轨迹。
- compare_windows_8gpu.sh同时支持BACKEND=npu及cuda。NPUS优先于GPUS；NPU使用ASCEND_RT_VISIBLE_DEVICES，单进程内设备为npu:0。保留原有两个脚本入口。
- Qwen3-VL评分器明确使用NPU SDPA；原GenEval2函数使用qwen_model.device，无需更改评分函数或阈值。

## 真实权重检查发现的配对问题

BAGEL的DiagonalGaussian默认执行mean + std * randn_like(mean)。相同RGB图像的两次VAE编码会生成不同latent，继而得到不同视觉KV。此前CPU测试中的图像缓存是固定夹具，未覆盖真实VAE后验。

现在在局部RNG上下文中执行条件准备。种子由更新前x_t的hash导出，因此同step STATIC／OBSERVED使用相同后验样本；离开上下文后恢复CPU与当前加速器RNG。记录visual_posterior_seed并在配对检查中校验。保持原生随机后验，不改为均值编码，不改当前x_t。

原生参考路径也在相同RNG状态下执行，才能检查逐元素parity。此修复不意味着CUDA／NPU跨设备逐位一致，后端浮点差异仍存在。

## 主机与依赖

- SSH：modelarts-job；root@dev.modelarts.cnszaismartcity01.api-ai.smartcitysz.com:32692。
- 主机：ma-job-42eb50fb-8150-4c7b-b670-7cf82ae18bcc-vj-0-worker-0，aarch64。
- 代码：/root/bagel-LatentCoT-NPU，Git分支NPU。原/root/bagel-LatentCoT和/cache目录未覆盖。
- Python：/root/venvs/bagel-NPU/bin/python。独立venv继承系统PyTorch/NPU插件，另安装transformers4.57.1、tokenizers0.22.2；原系统环境不变。
- PyTorch2.7.1+cpu＋torch-npu2.7.1.post1；这里cpu是wheel版本标识，NPU由torch_npu注册，已能实际执行算子。
- BAGEL：/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT。
- Judge：/data/model/Qwen3-VL-8B-Instruct。
- 官方评分源码：/root/npu-eval-tools/GenEval2/evaluation.py。与H200版本相同，SHA256为90791e35321a1a2bf65366517dd60f39037f90435490eb122c0027b52d42cda9。
- 检查产物：/root/npu-eval-tools/validation/；不混入正式实验目录。

## 检查状态

2026-10-09完成以下检查：

- 29项CPU／NPU测试通过，包括原22项、cached causal对齐、变长GQA和样本隔离、4层完整MoT decoder CPU／NPU近似数值对齐、重复运行、缓存不变、RNG回放与恢复。跨设备attention采用atol/rtol=0.02，decoder采用0.03；不是跨设备逐位parity。
- CPU模拟的5组分片生成、续跑、篡改拒绝、评分合并及HTML导出通过。NPU／CUDA多卡脚本的物理设备选择、单进程逻辑index0与CPU准备／报告阶段检查也通过，没有启动正式生成。
- BAGEL真实权重11项检查全部通过，结果在validation/e0_fixed.json：与原生编辑参考的上下文、flow布局、velocity逐元素相同；STATIC／OBSERVED视觉前缀相同；当前x_t/t、canonical cache和权重不变；velocity有限且可重复。输入为人工灰色RGB和seeded noise，不是正式样本评测。
- 原生VAE在NPU从随机latent解码512×512有限RGB通过，峰值PyTorch分配约1.20GiB，仅包含这个VAE检查进程，不是整套pipeline峰值。结果在validation/vae_smoke.json。
- Qwen3-VL-8B在NPU完成合成灰图的单次接口检查，结果在validation/scorer_smoke.json。仅验证接口，不验证打分准确性。

正式40图生成和评分由用户启动，尚未执行；不根据数值parity声称语义收益或质量保持。

NPU的BF16算子和CUDA FlashAttention并不保证同seed图片逐位一致。必须在同一NPU环境内比较Base、STATIC与OBSERVED，并重新绑定源码、权重、数据和后端。不能混用H200生成清单或旧plan续跑。

## 用户命令

在modelarts-job执行，先确认所选设备属于本任务且可用。Phy-ID0、2、4、6、8、10、12、14分别选择每张物理卡的一个设备；当前PyTorch可见全部设备时device_count为16。

```bash
npu-smi info
cd /root/bagel-LatentCoT-NPU
export BACKEND=npu
export NPUS=0,2,4,6,8,10,12,14
export OMP_NUM_THREADS=4
unset COMPARISON_PROMPTS
export CONFIG="$PWD/configs/observation_comparison.json"
mkdir -p /root/outputs
export RUN=/root/outputs/und_native_observation_npu_$(date +%Y%m%d_%H%M%S)
set -o pipefail
bash scripts/compare_windows_8gpu.sh "$RUN" 2>&1 | tee "${RUN}.log"
```

脚本先执行真实权重数值检查。通过后才生成40张图，再评分并导出comparison.html。续跑须保持源码、配置、模型、数据和分片数量完全一致，并显式设置RESUME=1。4设备可将NPUS改为0,2,4,6。
