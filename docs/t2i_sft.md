# 原生 BAGEL 的纯 T2I SFT

用高质量图文对直接更新原生 GEN 专家。没有 UND corrector、CFG 训练目标、guidance-safe 投影或附加语义损失。文本 UND、VAE、SigLIP/connector 冻结；GEN 注意力/MLP/归一化、`vae2llm`、`time_embedder`、`llm2vae` 全参数训练。输出是原生模块权重，不是 LoRA 或新纠错头。

## Table 3 配方与纯 T2I 的适配

依据 `/Users/zyd/Documents/LCoT-codex/bagel.pdf` 第10页 Table3，及第5页条件丢弃说明。配置在 `configs/t2i_sft.json`：

| 项目 | 值 |
|---|---|
| 更新步数 | 15000 |
| 学习率 | 2.5e-5，500步线性warmup后constant |
| AdamW | beta=(0.9,0.95)，eps=1e-15 |
| Weight decay / gradient clip | 0 / 1.0 |
| EMA | 0.995 |
| Diffusion timestep shift | 4.0 |
| 图像尺寸 | 短边至少512、长边最多1024，16像素对齐 |
| 文本条件丢弃 | 0.1 |

本版本只取生成任务。论文的CE:MSE=0.25:1、理解/视频/交错数据采样比不应用到纯T2I；文字没有CE目标。丢弃条件时移除整个文本段（包括文本边界），与原生CFG的空文本缓存一致。工程上采用global batch256、每rank逐图累积、单图内部双向注意力，而非论文每rank 40K–45K token packing。序列上限40960，只用于拒绝超长样本，不是每rank实际token量。不能将本配置宣称为完整Table3混合训练、72.7B tokens或其benchmark复现。

令z0为冻结VAE编码，epsilon为独立标准高斯噪声。对每图采样u~N(0,1)，s=sigmoid(u)，t=4s/(1+3s)，zt=(1-t)z0+t epsilon。原生GEN预测速度，唯一损失为每图mean((v_theta-(epsilon-z0))²)，再对全局batch图像平均。整个时间区间参与训练，每图每次只做一个时间点，不运行50步生图。文本在前且因果，图像split内部双向、可看文本；目标图不作为额外SigLIP或干净VAE条件。

## 1. 在数据所在主机准备独立索引

在部署代码根目录执行。准备过程只使用CPU，原始JSONL、图片和源文件只读。输出目录必须不存在。下面路径为当前ModelArts环境：

```bash
export PYTHONPATH="$PWD"
PY=/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python
MODEL=/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT
CORPUS=/data/zyd_workspace/bagel-LatentCoT/datasets/native_t2i_pairs_20261022
"$PY" -m qwen_latent_cot.t2i.cli prepare \
  --pairs /data/250010019/unimoe_r3/qwen_image_vllm_20261019/pairs/pairs-20261022.jsonl \
  --model "$MODEL" --config configs/t2i_sft.json --output "$CORPUS"
```

准备过程解码图片，按原比例缩放并对齐尺寸，拒绝尺寸不足或宽高比过大、重复uid/图片路径、超长序列和无效图文对。它不裁剪计数场景、不静默截断文本，也不把解码成功等同于语义高质量。少于512像素短边或比例超过2:1的样本被排除；具体数量写入manifest。

通过 `source_file/source_line` 回查源记录的 `raw_prompt`，验证uid，再按规范化原始提示全局分组。相同原题的扩写跨benchmark也同组；按固定种子分成约98%/1%/1% train/val/test。源文件必须可读，否则准备失败，不退化成逐行随机切分。`manifest.json` 只在准备完成后写入；中断目录保留用于诊断，重试使用新输出目录。

数据来自benchmark原题与扩写，留出部分不代表整个benchmark仍独立。训练用过的题目及其派生提示不得进入独立泛化评分。现有32条纠错图册不能直接当本次SFT的未见题评测。

## 2. 在空闲的NPU上先做硬件试跑，再正式训练

先验证真实GEN的前反向、有限梯度、峰值显存和分片恢复。当前运行中的guidance-safe作业不能被此入口自动停止。下面正式命令要求四节点及其64卡已明确分配给本SFT；各节点共享同一代码、模型、corpus和输出路径。

在四个节点分别设置NODE_RANK=0/1/2/3，MASTER_ADDR为节点零的集群地址：

```bash
export NNODES=4 NPROC_PER_NODE=16
export NODE_RANK=0                 # 其他三个节点分别为1、2、3
export MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
export MASTER_PORT=29673
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
bash scripts/train_t2i_npu.sh \
  --model /data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT \
  --corpus /data/zyd_workspace/bagel-LatentCoT/datasets/native_t2i_pairs_20261022 \
  --output /data/zyd_workspace/bagel-LatentCoT/outputs/native_t2i_sft_table3_20261022
```

该脚本不停止其他进程，也不自动占用另一作业的卡。global batch256在64rank时每rank累积4张图。FSDP FULL_SHARD按原生MoT层分片；FP32 master/optimizer/EMA，BF16计算。原生层中冻结UND与GEN混合，`use_orig_params=True`支持此设置，但临时梯度显存可能包含冻结参数对应区域，必须实测峰值，不能套用纠错器的显存结果。[PyTorch 2.7 FSDP说明](https://docs.pytorch.org/docs/2.7/fsdp.html)

首步及每1000步保存rank-local checkpoint；每500步及最终步用EMA做固定噪声的held-out flow MSE验证。最终为step015000。MSE不是画质或语义评分；应另外用未见提示、相同seed/CFG/尺寸/步数比较原生与EMA生图。

## 3. 精确续训

保持相同world size、原生模型路径、prepared corpus和优化配方，在全部节点原命令中加：

```bash
--resume /data/zyd_workspace/bagel-LatentCoT/outputs/native_t2i_sft_table3_20261022/step_001000
```

每rank保存本地GEN分片、FP32 EMA、AdamW状态、数据排列与游标、Python/CPU/设备RNG。目录有完整`manifest.json`才可续训。不能把当前UND纠错checkpoint交给此入口。完成检查应确认最终manifest、全部64份rank文件、step15000日志及EMA验证记录。

## 4. 导出原生GEN权重并生图

导出不需要NPU。保留原生BAGEL目录不变，导出目录只含GEN完整权重覆盖层及来源信息：

```bash
"$PY" -m qwen_latent_cot.t2i.cli export \
  --checkpoint /data/zyd_workspace/bagel-LatentCoT/outputs/native_t2i_sft_table3_20261022/step_015000 \
  --output /data/zyd_workspace/bagel-LatentCoT/outputs/native_t2i_sft_table3_20261022_ema
"$PY" -m qwen_latent_cot.t2i.cli sample \
  --model "$MODEL" \
  --weights /data/zyd_workspace/bagel-LatentCoT/outputs/native_t2i_sft_table3_20261022_ema \
  --prompt 'a photo of three books on a wooden table' \
  --seed 17 --width 1024 --height 1024 --steps 50 --cfg 4 \
  --output /data/zyd_workspace/bagel-LatentCoT/outputs/native_t2i_books.png --device npu:0
```

不传`--weights`就是原生对照；`export --online`导出在线权重而非默认EMA。采样复用原生CFG/renorm和Euler，不接纠错器。导出需要容纳完整GEN权重的CPU内存和磁盘空间；rank-local checkpoint不是单卡可直接加载的导出格式。

## 当前验证状态

- 27项CPU测试通过，包括原生训练forward对照、GEN梯度/UND冻结、激活重计算、分组索引、精确续训、EMA导出及原生采样。
- 本地及ModelArts目标Python环境的两进程CPU/Gloo FSDP FULL_SHARD试跑通过，包含逐rank精确续训、EMA和重组导出。
- 代码已独立部署在 `modelarts-64:/data/zyd_workspace/bagel-LatentCoT/code/bagel-native-t2i-sft-20261022`，不替换当前纠错作业代码。ModelArts CPU测试日志为 `artifacts/t2i_sft_20261022/modelarts_cpu_tests.log`，分片日志为 `modelarts_fsdp_two_rank.log`。
- 尚未准备606433条全量数据的最终corpus；尚未在真实原生7B GEN上执行NPU/HCCL显存及前反向验收；尚未启动正式SFT。现有64卡纠错训练未改动。

日志在本地`artifacts/t2i_sft_20261022/cpu_tests.log`与`fsdp_two_rank.log`。可复跑分片合同测试：

```bash
TMP=$(mktemp -d)
PYTHONPATH="$PWD/tests:$PWD" OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc-per-node=2 tests/fsdp_t2i_worker.py "$TMP"
```
