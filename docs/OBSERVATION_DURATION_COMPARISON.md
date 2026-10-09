# 固定观察 Memory 的作用时长对比

2026-10-09，NPU 分支。33 项 CPU/NPU 数值测试通过，32 prompt 输入覆盖检查通过。正式评测由用户启动。

问题：原先只在一个去噪调用中使用更新后的条件。现在仅延长同一缓存的使用时间，检查持续作用是否增加语义 Repair，同时保留旧 Early20 R2 原样控制。

## 固定项与对照

- BAGEL 权重冻结，512 图像、50 点 shift3 调度、seed0。
- 32 个既有 `data/prompts32.jsonl` prompt。Base、旧控制、4 个时长各自 STATIC/OBSERVED，共 320 张图。
- 新路径在 step9 由原生速度构造早期预测图，只执行一次完整 VAE＋ViT 图像编码和 28 层 UND 文本写入。完整缓存随后固定，不重复 writer、不压缩、不生成反馈文字。
- STATIC 和 OBSERVED 都有完整视觉缓存；区别仍是文本写入时是否读取图像。每个时长使用相同编辑 CFG text3/image1.5。Base 和旧控制保持 text4/image1。
- 新窗口分别为 `[9,10)`、`[9,14)`、`[9,19)`、`[9,29)`，持续 1/5/10/20 个调用。
- 旧 `LEGACY_EARLY_20_R2` 保持 `[0,20)`、模型层 `[0,8)`、R2，直接复用现有实现，没有编辑视觉缓存。

新 20 步窗口的总 delta t 约 0.2561，旧控制约 0.1869。两者调用步数相同，但开始时间和具体 t 区间不同；不能解释为仅 writer 架构不同的严格对照。

## 每一步执行

窗口内始终使用该步当前的 x_t/t。原生 prepare 接口生成的随机噪声不进入轨迹。固定缓存使用原生编辑位置和索引；该调用只执行一次 Euler 更新。窗口结束后释放缓存，并恢复原始文生图条件。

为了诊断，窗口内每一步仍计算当前实验轨迹状态下的原生参考速度，再计算固定缓存条件速度。参考不是独立 Base 轨迹的对应速度，不参与 Euler 更新。它增加计算量，工程计时不能当作部署预算结果。

只在首次写入时保存早期预测图、完整 Memory state 和问答 probe。后续记录标记 `updates=0`，保存 t、dt、速度相对差、速度 RMS 差和 `dt*||delta_v||/||x_t||`。最后一次读取验证文本 Memory fingerprint 不变。源图像和各分支缓存随窗口延续；probe 不进入生成缓存。

分片合并检查完整窗口覆盖、一次 writer、STATIC/OBSERVED 首次观察一致、同种新路径在各时长下使用相同初始 Memory、文件 hash，以及旧控制的窗口/R/模型层设置。

## 报告

输出所有组相对 Base 的语义 GM、质量代理、invalid、Repair/Damage；另外比较：

1. 同时长 OBSERVED 对 STATIC。
2. OBSERVED 的相邻时长。
3. 每个 OBSERVED 时长对旧 Early20 R2。

以旧控制为 Repair 保留参照。质量代理和 Repair/Damage 仍按原有协议解释，变化幅度不等于语义改善。

32 个图像组的离线 HTML 分为两个页面。下载时须同时下载 `comparison.html` 和 `gallery/`。

## ModelArts，16 个 NPU 芯片

节点有 8 张双芯片卡，Phy-ID 0–15 对应 16 个可见设备。每个进程只看到选定芯片，内部使用 npu:0。运行前检查占用。

```bash
cd /root/bagel-LatentCoT-NPU
export BACKEND=npu
export NPUS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export OMP_NUM_THREADS=4
export CONFIG="$PWD/configs/observation_duration_comparison.json"
unset COMPARISON_PROMPTS
mkdir -p /root/outputs
export RUN=/root/outputs/und_memory_duration_npu_$(date +%Y%m%d_%H%M%S)
set -o pipefail
bash scripts/compare_windows_8gpu.sh "$RUN" 2>&1 | tee "${RUN}.log"
```

脚本先绑定配置和源码，执行观察路径原生 parity、旧 Early20 R2 数值检查，随后生成、评分、合并报告。不跳过失败的配对。
