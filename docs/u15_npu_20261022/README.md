# U1.5 NPU Pipeline：使用与验证

分支：`codex/sensenova-u15-npu`。

已实现纯T2I SFT、逐层GEN-hidden caption reader、单一guidance-safe RGB纠错、checkpoint续跑/导出、64rank启动入口及六榜适配。外部输入只有prompt，不使用ViT或外部图像，不在GEN各层注入残差。

## 1. 源码与环境

在项目根目录建立带显式补丁的上游副本，保留`refs/SenseNova-U1`：

```bash
python3 scripts/prepare_u15_source.py \
  --source /Users/zyd/Documents/LCoT-codex/refs/SenseNova-U1 \
  --output /Users/zyd/Documents/LCoT-codex/refs/SenseNova-U1-NPU
```

补丁包含两个改动：causal mask常量在输入设备分配；UND注意力允许显式reader LoRA增量。原生调用不传`und_lora`，行为不变。不要直接安装上游的Torch2.8 CUDA依赖覆盖NPU环境。

本次远端验证使用：

- 主机：`modelarts-64`，空闲逻辑设备15；HCCL/FSDP测试使用14、15。
- Python：`/data/zyd_workspace/bagel-LatentCoT/venvs/u15-NPU/bin/python`。
- 代码副本：`/data/zyd_workspace/bagel-LatentCoT/code/u15-npu-20261022`。
- 上游副本：`/data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b`，补丁v2。
- 环境基于既有Torch2.7.1/torch_npu和transformers4.57.1的只读依赖，在新venv运行；未升级BAGEL环境。

准备完整的`sensenova/SenseNova-U1.5-8B-MoT`权重到本地模型目录，固定下载revision，并保留tokenizer/config/safetensors。首次不合并8步LoRA。执行前核对目标设备空闲；`npu:0`指可见设备映射后的索引。现有评测占用卡0/1时不要使用它们。下列命令在远端项目根目录执行，设置：

```bash
B=/data/zyd_workspace/bagel-LatentCoT
PYTHON=$B/venvs/u15-NPU/bin/python
SOURCE=$B/code/sensenova-u15-c35c83b
MODEL=$B/models/SenseNova-U1.5-8B-MoT
CONFIG=configs/u15/correction.json
```

`MODEL`是待放置真实权重的路径，不表示本次已经下载。Runtime只从本地加载，不静默联网取其他模型。

## 2. 原生生成与数据准备

```bash
"$PYTHON" -m qwen_latent_cot.u15.cli native \
  --source "$SOURCE" --model "$MODEL" --config "$CONFIG" \
  --prompt 'A red ball to the left of a blue cube.' --seed 17 \
  --output outputs/u15_native/example.png

"$PYTHON" -m qwen_latent_cot.u15.cli prepare \
  --source "$SOURCE" --model "$MODEL" --config "$CONFIG" \
  --pairs /data/250010019/unimoe_r3/qwen_image_vllm_20261019/pairs/pairs-20261022.jsonl \
  --evaluation-manifests /data/250010019/unimoe_r3/outputs/0_qwen38max_fullbench_20261002/manifests \
  --output "$B/datasets/u15_pairs_512"
```

数据准备复用既有图文对索引、原始prompt family划分和图像检查，并排除六榜prompt family和明确的benchmark来源标签。该排除是对已知来源的检查，不等于检测任意同义改写。首版固定512方图，只保留满足原有无裁剪尺寸规则且geometry匹配的样本，移除数量写入manifest。

## 3. 三阶段训练与导出

统一配置起点为512px、50次Euler更新、前25次纠错、CFG4、shift3、cfg_norm=none。它是本分支固定工程配方，不宣称复现U1.5全部SFT/RL/MOPD。GEN SFT只使用原生flow loss，不添加猜测的LPIPS权重。

单卡caption训练：

```bash
MODEL_PYTHON="$PYTHON" NPROC_PER_NODE=1 bash scripts/train_u15_npu.sh caption \
  --source "$SOURCE" --model "$MODEL" --config "$CONFIG" \
  --corpus "$B/datasets/u15_pairs_512" --output "$B/outputs/u15_caption"

"$PYTHON" -m qwen_latent_cot.u15.cli export \
  --checkpoint "$B/outputs/u15_caption/step_001000" \
  --output "$B/outputs/u15_reader.pt"
```

继承reader后训练corrector：

```bash
MODEL_PYTHON="$PYTHON" NPROC_PER_NODE=1 bash scripts/train_u15_npu.sh correct \
  --source "$SOURCE" --model "$MODEL" --config "$CONFIG" \
  --reader "$B/outputs/u15_reader.pt" \
  --corpus "$B/datasets/u15_pairs_512" --output "$B/outputs/u15_correct"

"$PYTHON" -m qwen_latent_cot.u15.cli export \
  --checkpoint "$B/outputs/u15_correct/step_001000" \
  --output "$B/outputs/u15_corrector.pt"
```

`export`默认导出online权重；`--ema`显式选择EMA。精确续跑在原训练命令增加`--resume .../step_XXXXXX`，保持world size和不可变config一致。部署必须使用portable导出，不把rank-local分片当成完整权重。

完整8B的GEN-only SFT使用下节多卡启动器的`--stage t2i`，不传reader；不要将上述单卡caption命令直接用于完整GEN SFT。多卡使用FSDP FULL_SHARD、FP32 master与BF16计算。SFT导出为`gen.pt`后，caption、correct、native/sample均通过`--gen-weights gen.pt`选择该底座。随后reader、corrector和global校准均须重新训练，不加载旧底座的corrector。

## 4. 64卡启动

在四节点ModelArts任务的节点零，确认64个逻辑NPU均可用于本任务后：

```bash
"$PYTHON" scripts/launch_u15_64.py \
  --python "$PYTHON" --code "$PWD" --source "$SOURCE" --model "$MODEL" \
  --corpus "$B/datasets/u15_pairs_512" --config "$PWD/$CONFIG" \
  --stage caption --output "$B/outputs/u15_caption_64"
```

启动器先核对四节点路径和空闲NPU，有其他任务就拒绝启动，不停止它们。它在`OUTPUT_launch`保存审计、node日志和PID。correct阶段增加`--reader`；SFT后的底座增加`--gen-weights`。既有27B裁判占卡时不能直接抢占这64卡。

## 5. 来源诊断与三路生成

```bash
"$PYTHON" -m qwen_latent_cot.u15.cli describe \
  --source "$SOURCE" --model "$MODEL" --config "$CONFIG" \
  --checkpoint "$B/outputs/u15_reader.pt" --prompt 'Two red balls.' \
  --node 12 --seed 17 --output outputs/u15_caption_correct.json
```

该命令先真实rollout到node12，再读取逐层hidden。增加`--source-off`屏蔽来源，或`--swap-seed 18`保持prompt、节点不变而交换生成状态。它们是诊断，不进入训练loss。

```bash
for ARM in native global corrected; do
  "$PYTHON" -m qwen_latent_cot.u15.cli sample \
    --source "$SOURCE" --model "$MODEL" --config "$CONFIG" \
    --checkpoint "$B/outputs/u15_corrector.pt" --arm "$ARM" \
    --prompts configs/semantic32.jsonl --seed 17 --output "outputs/u15_32_${ARM}"
done
```

每张图保存原prompt、seed、底座身份和trace。正式训练只有一个纠错MSE；不加caption CE、cosine、shuffled或raw辅助loss。GEN冻结。日志分别记录同目标zero_correction_mse、物理速度误差、reader/head梯度及applied_negative_fraction。

## 6. 六榜与HTML

`benchmark_u15_sixbench.py`复用原manifest和独立27B评分器，新建单arm run：

```bash
"$PYTHON" scripts/benchmark_u15_sixbench.py prepare \
  --run "$B/outputs/u15_native_sixbench" --source "$SOURCE" --model "$MODEL" \
  --config "$CONFIG" --arm native --label u15_native

"$PYTHON" scripts/benchmark_u15_sixbench.py generate \
  --run "$B/outputs/u15_native_sixbench" --device npu:0
```

64卡生图时在四节点分别使用torchrun的`--nnodes=4 --nproc-per-node=16 --node-rank=... --master-addr=...`调用该generate命令；worker按RANK/WORLD_SIZE分片，不做分布式模型通信。corrected/global各准备独立run，增加`--checkpoint`并使用不同label。

完整生图后按顺序运行`materialize`、`score`、`status`。评分使用每个新run的`scoring_qwen27b_v1/scoring_config.json`，必须启动对应namespace的独立27B代理，不能借用正在运行的BAGEL评分缓存。评分器不在generate过程中自动启动。六榜6378条/arm，完整三路为19134张。

下载三个run的manifest、PNG、receipt与scores后，在本地：

```bash
python3 scripts/render_u15_comparison.py \
  --runs native=/local/u15_native global=/local/u15_global corrected=/local/u15_corrected \
  --output /local/u15_comparison --limit 32
```

HTML给出固定顺序三路图片和各榜有效分母、原始summary；没有完整有效分母就保持未完成，不把失败计0，不用六榜异质分数直接求平均。

## 7. 已取得的验证与实际缺口

测试使用上游真实NEOChat/Qwen3双专家结构、原生PixelShuffle head的随机小配置，不是mock网络。证据见本目录`evidence/`。

- CPU：5项测试通过，包括严格native权重加载roundtrip、逐图投影、数据排除、原生/零纠错一致性、未来caption答案不回流。
- 真实NPU：5项测试通过，包括caption/纠错前后向、native结果一致、冻结基础参数、reader继承、global校准、导出/恢复和不中断对照一致。
- 双卡HCCL/FSDP：真实两步GEN-only SFT、验证和分片导出通过，日志标记`HCCL_FSDP_GEN_EXPORT_PASSED`。混合冻结/可训练FSDP层仍有额外梯度内存，64卡microbatch须实测。
- 完整8B权重生成质量、512px峰值显存、64rank实跑和自然数据语义改善尚未验收；当前已检查的模型目录没有U1.5 release权重。
- 没有启动长训练，也没有重新占用原评测的卡0/1。

下一项直接依赖是完整U1.5权重。权重就绪后先跑原生32条与真实caption阶段，再决定64卡预算；小模型通过不作为语义或画质改善结论。
