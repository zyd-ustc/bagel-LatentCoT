# NPU：Memory 增量定位

该诊断针对 `full_depth_restart`。每轮从原生 UND 入口出发、连续经过全部28层；GEN只循环0–7层。它检查轮间增量在哪个计算阶段减弱，不判断语义或质量提升。

## 输入与比较范围

默认32个 prompts、seed0、512分辨率，沿真实 Early20 R4 的50点shift3采样轨迹运行。每条轨迹只推进49次原生sampler。取step0/4/9/19的真实 x_t/t；每个采样点独立计算Base/R1/R2/R3/R4，所有候选共享相同x_t、t、prompt与cache。完整suffix和readout在每个候选各执行一次。候选结果不推进sampler。最终用关闭诊断的R4调用推进该步，并要求与诊断开启时R4 velocity完全一致。

本诊断仅使用Early20 R4作为采样轨迹。固定状态R1不是独立Early20 R1生成的相同步号，不应将两者混为同一比较。每个候选的x0预览由x_t−t·v推导，是早期预测，不是完成的图像。只保存一张实际轨迹最终图。

## 记录内容

- `within_deepest`：一次R4调用内GEN round0–4、Memory writer round1–4。
- `final_depth`：独立Base/R1–4完整调用的最终GEN各层输入/输出hidden，包含0–7层body和8–27层suffix。另记录实际送入llm2vae的归一化后GEN hidden，帮助区分suffix、final norm和velocity投影。
- GEN：层输入hidden、原生attention输入K/V、层输出hidden、实际读取的Memory K/V。图像token与边界token分别记录。
- UND：层输入/输出hidden、live self K/V、实际供GEN读取的Memory K/V。内容与固定特殊token分别记录。
- 每项记录相邻轮变化、首轮参考变化；Memory K/V另记录相对原prompt KV的变化。UND block内部输入→输出变化独立标为`within_block`，不能当作轮间增量。
- 输出相对L2、最大绝对差、余弦、exact equality、发生变化的元素比例和token delta RMS的中位数/P90/最大值。计算统计时将快照转到CPU float32，不改变实际BF16计算。
- velocity记录真实conditional、null-text CFG branch、BAGEL返回的CFG后结果。额外的CFG前float32重建只用于诊断。记录相邻轮/相对Base变化以及dt乘velocity差造成的Euler更新差。
- 检查诊断前后velocity一致、x_t/t不变、prompt KV和原生hidden seeds不变、null分支和完整步/层覆盖。任何只读合同或完整性检查失败时停止。

GEN KV是该层attention输入的投影，不是该层更新后的hidden投影。Memory body KV保持同层更新后hidden投影；suffix KV保持原生输入读出。诊断开启时仅在临时cache保存实际KV，原prompt不被写入。默认不保存巨大完整hidden/KV张量；`tensor_layers`显式指定的层会保存所有观察到的快照，可能显著增加磁盘开销。

## 用户运行

代码位于modelarts-job的 `/root/bagel-LatentCoT-loop-layer-NPU`。用分配给本任务的逻辑NPU编号；独立prompt分片，不使用torchrun/DDP。正式诊断由用户启动。

```bash
cd /root/bagel-LatentCoT-loop-layer-NPU
export NPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export OMP_NUM_THREADS=4
export CONFIG="$PWD/configs/increment_diagnostics.json"
unset DIAGNOSTIC_PROMPTS
export RUN=/root/outputs/increment_trace_$(date +%Y%m%d_%H%M%S)
set -o pipefail
bash scripts/diagnose_increments_npu.sh "$RUN" 2>&1 | tee "${RUN}.log"
```

脚本绑定源码、权重和输入，只加载BAGEL，不加载评分模型、不训练。默认32条轨迹、128个固定状态探针，每个探针计算R0–4。每个worker日志在workers/worker_N.log。失败日志和部分结果保留，但不导出假完整报告；需新目录重跑。

需要更小规模时，将max_prompts改为8并保存成新配置，CONFIG指向该文件。steps也可缩小，但必须是0–19之间递增且不重复的步号。诊断开销包含重复suffix、CPU快照和VAE预览，不能用作正式速度预算。

## 输出与判断

- `diagnostics.html`：逐层增量表、固定状态各R的x0预览、实际轨迹最终图。
- `summary.md/json`：完整性合同与聚合统计。
- `layers_summary.csv`、`velocity_summary.csv`：按step/层/轮次分别统计中位数与exact equality比例，不混合时间步。
- `localisation.jsonl`：逐prompt、逐step、逐相邻轮列出完全固定的Memory/GEN KV层，body/suffix出口hidden变化和velocity/Euler变化。
- 每个prompt目录保存step_NN/layers.jsonl、velocity.jsonl、case.json、state.pt，以及trajectory.json。state.pt保存真实x_t/t，可复查。可选selected_layer_tensors.pt由tensor_layers控制。

若浅层Memory/KV逐轮固定，支持结构性逐层锁定。如果Memory变化、GEN输出不变，优先检查读取路径。如果body有变化、suffix/final velocity更小，优先检查后半网络。如果conditional增量保留但CFG后增量变小，检查guidance与归一化。以上是定位线索；数值变化没有直接给出语义Repair，不能把原始hidden更新大等同于有效信息增益。

HTML使用相对图片路径；本地查看需下载整个RUN目录。
