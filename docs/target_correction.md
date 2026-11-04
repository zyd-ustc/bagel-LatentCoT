# 目标驱动纠偏：当前实现

2026-11-04。kind=`u15_target_correction_v1`。当前训练和推理入口。只支持本方案的检查点。

## 定稿落地

- 训练／部署均 `rho=0`，从真实历史态回退；不是非编辑区域保持的数学保证。
- Tier-B 强编辑源为同种子原生完稿 `x_fail`，学生读取真实 preview。外部编辑指令与部署条件分开：训练 condition 只从轨迹 record 读取，不读取外部裁判的完稿诊断。
- 图像优先 UND prefill；全部原生权重冻结，GEN 前 36/42 层 rank64 LoRA，零初始化 start_embedding 和 proposal_proj。
- 当前原生 clean 外推经 GEN 图像嵌入后注入适配支；仅提案条件 detach，不截断实际状态链。额外图像编码有成本。
- Stage 1：不退场的残差速度监督，线性桥指向编辑目标；该阶段不允许作为部署检查点。
- Stage 2：窗口与尾段前向均为部署 Heun 网格（终点 Euler），反向分别为 5 步与 3 步 Euler 代理雅可比。代理梯度有偏。
- 结构监督：冻结 UND 多层图像 token 特征 + 4×下采样 Charbonnier；非目标特征／像素 L1 保持；沿粗窗口实际状态计算低权重残差保持；padding 单独约束。
- UND Yes/No 语义反传默认移除，外部 VLM 只做离线验收。

上游 NEOVisionModel 实际仅返回浅层嵌入，hidden_states=None。因此结构损失改用冻结 UND 解码器第约1/3、2/3及最后层的图像 token。它增加 UND 前反向成本；不能用旧 Stage A 显存或“26次调用”推算总预算。

## 源码

`qwen_latent_cot/u15_target_correction/`：model.py（条件与适配器）、field.py（精确前向/代理反向）、loss.py（结构和保持）、data.py（两层语料与统一RNG采样器）、train.py（两阶段/EMA/恢复）、sampling.py（文生图闭环与原生轨迹）、judge.py（外部API）、cli.py（入口）。

配置：`configs/u15_proposal_correction/target_stage1.json`、`target_stage2.json`。

Stage1：6000步，batch256，lr1e-4；A/B/zero=80/15/5%。Stage2：4000步，batch128，lr3e-5；20/70/10%。Stage2 的每个 tier 都展开到成图：前向沿部署网格，反向为 5 步窗口加 3 步冻结尾段。编辑对的锚点仍是线性桥，不是真实失败历史。保存周期250步。每个样本仍逐个前反向、累计梯度，全局求和，底座与LoRA复制式并行。所有抽样随机状态归入 sampler checkpoint。

## 数据接口

语料是JSON manifest：`{"records":[...]}`。每行必须有 uid、family、split（train/val/test）、approved=true、tier。

Tier A：
```json
{"uid":"a1","family":"source1","split":"train","approved":true,"tier":"A","source_image":"src.png","target_image":"edit.png","raw_prompt":"完整目标场景描述","instruction":"结构编辑指令"}
```
源和目标使用同一适配矩形，不裁切；黑色padding单独监督。此阶段用源图作为参考上下文，不声称与真实预览同分布。

Tier B：
```json
{"uid":"b1","family":"prompt1","split":"train","approved":true,"tier":"B","trace":"trace/00000.pt","arm":"deep","target_image":"edited_native_final.png","raw_prompt":"原始生成提示词","target_edit_instruction":"仅用于离线生产目标的强编辑要求"}
```
trace须由新入口生成，绑定底座/CFG/网格/读取点/回退协议。target_image必须与原生成图画布大小一致，不能缩放真实状态。preview、历史态、x_fail、deployment condition均读取trace；不从编辑目标反演。

zero行格式同B，tier改为zero，不需要target_image，目标自动取trace.final。只将经外部确认原生完稿正确的样本批准为zero。训练代码不使用完稿诊断重写在线条件。

评测family不可进入训练split；不要把现有GenEval2评测图迁入训练。

## 命令（由用户执行，本次未执行）

在仓库根目录，使用现有环境解释器，设置 MODEL/SOURCE/OUT/CORPUS/PROMPTS：

```bash
python -m qwen_latent_cot.u15_target_correction.cli trace \
 --model "$MODEL" --source "$SOURCE" \
 --config configs/u15_proposal_correction/target_stage1.json \
 --prompts "$PROMPTS" --output "$TRACE_OUT" --device npu:0

python -m qwen_latent_cot.u15_target_correction.cli train \
 --model "$MODEL" --source "$SOURCE" \
 --config configs/u15_proposal_correction/target_stage1.json \
 --corpus "$CORPUS" --output "$STAGE1_OUT" --device npu:0

python -m qwen_latent_cot.u15_target_correction.cli train \
 --model "$MODEL" --source "$SOURCE" \
 --config configs/u15_proposal_correction/target_stage2.json \
 --init-checkpoint "$STAGE1_CHECKPOINT" \
 --corpus "$CORPUS" --output "$STAGE2_OUT" --device npu:0

python -m qwen_latent_cot.u15_target_correction.cli sample \
 --model "$MODEL" --source "$SOURCE" \
 --config configs/u15_proposal_correction/target_stage2.json \
 --checkpoint "$STAGE2_CHECKPOINT" --ema \
 --prompts "$PROMPTS" --output "$OUT" --device npu:0 --compare-native
```

torchrun多卡追加`--distributed`，沿用现有4节点16进程启动环境。`--resume`是同stage/config/world/corpus的精确恢复；`--init-checkpoint`仅移入EMA模型权重，优化器与采样器重置。改变stage不会放宽底座、适配器或轨迹协议匹配。

## 外部 GLM 裁判

```bash
python -m qwen_latent_cot.u15_target_correction.cli judge \
 --requests "$REQUESTS" --output "$JUDGE_OUT" \
 --base-url "$GLM_BASE_URL" --model "$GLM_MODEL" --key-env GLM_API_KEY
```

请求JSONL：index（整数、用于文件名）、uid、instruction、images（每项role与path）。路径相对请求文件；可同时提供原生图与编辑目标。instruction写明原提示词和需要验收的要求。结果保存原请求、模型、JSON回答、用量，已有结果不重复调用。独立运行目录绑定同一批请求；不要在旧结果目录修改请求。API错误直接暴露，不暗中换模型。密钥从环境变量读取，不写入请求或日志。

Qwen-Image-2.1生成接口尚未提供，因此不新增猜测的远端调用：消费既有编辑生产流水线输出的目标图和外部验收结果。GLM同样需要实际base URL、模型ID与环境变量；本次未调用API。

## 运行成本

精确前向需要完整窗口和尾段求值。粗网格代理反向减少保留激活的链长，不消除前向成本。
