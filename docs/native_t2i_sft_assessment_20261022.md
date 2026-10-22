# 用新图文对提升原生 BAGEL：最小 T2I 后训练

2026-10-22，只读核对及方案。没有修改训练代码、停止当前 guidance-safe 作业或启动新训练。

## 结论

采用原生 GEN-only Rectified Flow SFT，直接更新 BAGEL 的生成专家和图像输入/时间/输出模块。保留原生网络与速度参数化；不带 UND 纠错器、不做 guidance-safe 目标、不做 semantic/contrastive loss。冻结文本 UND、SigLIP/connector、VAE 是本次聚焦生成任务的建议，不是论文混合任务 SFT 的完整复现。首选 GEN 全参数微调；分片和激活重计算由实际 NPU 显存确定，不能照搬53M纠错 adapter 的优化器配置。

## 论文和现有源码

来源：本地 `/Users/zyd/Documents/LCoT-codex/bagel.pdf`，§2.2–2.3/p5、§4/Table3/p10。已渲染Table3核对。

- 冻结 FLUX VAE；图像转16通道latent、2×2 patch；保留2D位置与时间编码。
- 文本在前、目标带噪图在后；图像token内部双向，图像可看前置文本，文本不可偷看目标图。纯单图T2I不需把目标图另作SigLIP或干净VAE条件输入。
- 论文SFT使用高质量图文对和交错数据，并混合理解任务；纯T2I仅取其中生成子任务，不需文字CE。
- SFT原表：lr2.5e-5、constant、warmup500、AdamW beta(0.9,0.95)/eps1e-15、weight_decay0、clip1、EMA0.995、shift4、最短边512/最长边1024；15K是原论文72.7B-token混合训练的预算，不直接套用此数据集。§2.3文本条件dropout0.1支持CFG。
- [原生训练forward](../qwen_latent_cot/bagel/modeling/bagel/bagel.py#L190)已实现sigmoid时间再shift、加噪和target=noise-clean的MSE。[原生入口适配核对](native_t2i_training_path.md)说明当前缺少可用NPU原生GEN训练脚本。

令z0=VAE(image)，epsilon~N(0,I)，u~N(0,1)，s=sigmoid(u)，t=4s/(1+3s)，zt=(1-t)z0+t epsilon。原生GEN预测v_theta(zt,t,P)，唯一图像损失为mean((v_theta-(epsilon-z0))²)。使用整个时间区间，不能继承纠错器前25步时间窗口。训练每对只采样一个时间，不运行50步生图。训练不先加CFG，也不投影目标；推理保持原生CFG和renorm。

## 数据快照

远端 `modelarts-64`，`/data/250010019/unimoe_r3/qwen_image_vllm_20261019/pairs/pairs-20261022.jsonl`：本次顺序扫描606433行、2517077158bytes，JSON坏行0、重复uid0。这里不是完整图片文件有效性或内容去重审计。

全部行有uid/image/receipt/prompt/benchmark/wh_ratio/source_file/source_line。prompt字符数min796、max20193、mean3538.09。按每10000行取一条、共61条，用当前BAGEL tokenizer检查：374–1332 tokens，mean707.77；53条>512、6条>1024。不能默认截断512；实际接入应测完整token长度并显式处理超长样本。

主要比例：1:1=150452、2:3=122319、3:2=324664，另有少量极端比例。按原比例分桶和等比例缩放，不直接拉成512正方形；极端比例另设处理规则，不能同时宣称所有样本均满足论文短边/长边范围。

14个benchmark来源计数：dpg32362、t2i_corebench31747、wise_verified30361、t2i_compbench_pp22515、dsg31771、genai_bench48444、spatial_geneval37982、unigenbench73310、prism_bench42034、oneig_bench74435、tiif_bench151474、geneval224382、geneval_pp2927、geneval2689。

前两份receipt标注teacher为Qwen/Qwen-Image-2.1、50steps、seed42、true_cfg_scale1.0，尺寸1024²与832×1248。仅代表已读receipt；不据此宣称全体配置一致。首条source记录含raw_prompt、thinking、answer.rewritten_prompt；pair.prompt是扩写提示，不是天然人工caption。训练只用prompt，不把thinking/receipt当标签。

已下载并实际检查前两张studio/library样图：材质、光照和细节有较高可用性，适合进一步考虑生成质量迁移；不能由两张推断606433张都具有精确计数/文字/空间关系。样图在 `artifacts/native_t2i_assessment_20261022`。

训练前的必要数据处理：检查图像可解码与尺寸、提示/图像对齐、来源分组去重和切分。数据明确来自DPG/GenEval等benchmark提示及扩写；相同原始prompt及其扩写/同源图片必须同组，训练用过的题目不作为独立泛化评测。保留raw_prompt映射，分别检查短提示和扩写长提示的表现。

## 推荐第一轮

从原生ema.safetensors起步，GEN全参数+原图像输入/输出/时间模块，冻结其余模块。复用原生训练合同并适配NPU可微注意力；先做单步前反向/显存与保存恢复检查，再确定分片训练配置。建议起点采用论文SFT lr2.5e-5、shift4、warmup500、weight_decay0、clip1、EMA0.995、文本drop0.1；这是待验证起点，不是已运行配置。

预算先按约1epoch而非15K/5000照搬。若全局batch256，本次未过滤快照一轮ceil(606433/256)=2369steps，5000steps约2.11epoch；最终以过滤后的train量为准。用未见源prompt、同seed、相同CFG/分辨率/步数对比原生和SFT，分别评估画风/结构/文字与语义；MSE只作训练健康信号。先得到更好的原生生成基线，再研究与corrector联合训练。
