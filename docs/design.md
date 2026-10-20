# 全层双状态循环

## 状态与执行

固定外层状态 x_t 和 t，一次原生全层 GEN 产生 baseline hidden 和完整 GEN KV。
每次反馈执行：冻结 C2 Read → 可训练 Write K/V → 图像 hidden 回灌 → 全层 GEN。
默认两次反馈，合计三次全层遍历。外层只使用最后一次速度做一次原生 CFG/Euler 更新。

循环仅覆盖采样前半程更新编号，不按 t>=0.5 判断。50 节点、49 次更新时，前 25 次更新启用循环。
训练从同一前半程节点均匀采样。每个外层步骤重新初始化内循环。

## 条件与回灌

每层 attention 拼接 [Prompt K/V, Memory K/V, Image K/V]，16 slot 不受小残差比例上限限制。
Memory 开启时 GEN RoPE 相对原始位置固定 +16，不随反馈次数累计。CFG 参考分支不注入 Memory 或回灌 hidden。

图像入口：H0 = vae2llm(x_t) + time_embedder(t) + latent_pos_embed(position)。
回灌：H_in = H0 + beta * GlobalRMSMatch(P(H_previous), H0)。
取上一轮末层、最终输出 norm 之前的图像 hidden。文本和边界 token 保持原始入口。
P 初始为单位矩阵，beta 初始 0.15。删除逐 token RMSNorm；投影后仅用全图标量 RMS 匹配 H0，保留 patch 之间的相对空间能量。

C2 UND block 始终使用冻结参数。Write LoRA 只在每层读完后的 K/V 投影启用，不进入 Reader attention。
中性 Writer seed 每次重置；GEN hidden 和新的逐层 Memory KV 构成两条跨轮状态路径。
跨反馈轮不 detach，梯度可经过冻结 Reader 的输入传播；第一遍 baseline 无梯度。

## 训练与保存

Write K/V rank 256，GEN Q/V/O rank 128，比例 alpha/rank 固定为 1。
共享回灌桥采用完整 hidden-width 投影。冻结其余 BAGEL 参数。
两次反馈的损失权重为 0.5、1.0。每轮损失包含 Hard-Patch 加权 flow MSE 和低误差区域速度保真项。
令 e_i 为 baseline 的逐 patch flow MSE，r_i = e_i / max(mean(e), 1e-8)。
Hard-Patch 权重为 (1 + alpha*r_i) / mean(1 + alpha*r)，alpha 默认 1；保真权重为 1/(1+r_i)。
每轮损失为 mean(w_hard * flow_error) + 0.02 * mean(w_keep * (v_round-v_baseline)^2)。
所有权重、target 和 baseline 均 detach；不另加空间平滑项，以免直接压低真实纹理。
误差权重不识别语义结构或平坦背景。日志分别保留原始 round_flow、round_weighted_flow 和未乘系数的 round_keep。
删除 bridge.norm 并增加 consumer V LoRA 后，checkpoint 升级为 v3；v2 不可直接载入或续训。
新格式保存 adapters、bridge、optimizer、sampler 和逐 rank RNG。拒绝旧 LoRA/RL 格式。

删除旧浅层循环、固定 early20、多种重置入口模式和 Flow-GRPO 路径。
C2 读出准备、数据准备及 BAGEL 原生模块继续作为基础设施保留。

## 验收

测试覆盖全层访问、hidden 回灌、跨轮梯度、冻结 Reader、条件分支隔离、位置偏移不累计、
checkpoint 重算等价、前半程边界、训练/采样同核、保存重载和优化器续训。
上一版（逐 token RMS、beta=1）通过 81 项 CPU 回归测试，并在 modelarts-job 上完成 16 卡、64 组原生/新循环采样对比。
首组 512²、49 次更新、seed 17 的实测：原生 13.64 秒、峰值 29.75 GiB；
新循环 25.89 秒、峰值 30.73 GiB。该组新循环成图出现明显块状噪声、画质退化。
这里的时间和显存是首组测量，不是全量均值。

本轮全图 RMS、beta=0.15、consumer Q/V/O 与新损失通过 83 项 CPU 回归测试。
测试覆盖空间能量比例、Hard-Patch/keep 权重的数值与梯度隔离、各层 V LoRA 梯度、一次优化更新和重载。
16 卡重采样完成相同 64 组对比；所有原生对照与上一轮逐像素一致。
新循环平均 26.22 秒/图，最大峰值 30.79 GiB。已查看草莓碗、自行车、向日葵、书灯四例，块状噪点明显减轻。
书灯样例仍生成四本书，未满足三本书的 Prompt；本轮不据此声称语义结构已纠正。
Training-free 对比不执行新训练损失；该采样变化来自回灌修改。


C2 准备阶段仅训练和验证语义读出，不再调用已删除的旧生成循环或生成验证模式。
生成训练、逐轮 flow 验证及终图采样统一由新的 interaction 路径负责。

参考 [Looped-DiT 官方实现](https://github.com/OpenSenseNova/Looped-DiT) 的共享计算与中间输出监督；
保留 BAGEL 的 RMSNorm、GEN 专家、token 输入及 KV 条件注入接口，不搬入 MMDiT block。
