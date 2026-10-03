# Step-5000 reader 固定注入诊断

比较三组：`native`、`untrained_reader`、`step5000_reader`。每个 prompt
使用同一初始噪声、模型和采样配置。两组 reader 使用相同的固定注入系数。
默认 `GATE_SCALE=1.0`，可修改。native 不注入。

`untrained_reader` 将 translation 的 B 置零。由于残差为 B(A(x))，此时
输出严格等于初始化时的 reader 输出，与 A 的取值无关。它不是零 memory。
`step5000_reader` 加载训练得到的 A/B。所有模型参数均冻结。

每个去噪步都重新计算 frozen Read bank。GEN 使用自己的当前状态，经原生
query 读取 memory，在 body `[12,20)` 的 attention output 上加固定系数的
reader 输出。两组 reader 的条件和实现相同，仅 translation 是否训练不同。
prompt KV 保持可见；不做 Write、mask、额外 R 轮数或跨 timestep memory 累积。

模型路径、K=8、body、rank/alpha、512×512、CFG=1、50个时间点和 shift=3
继承 checkpoint 同目录的 `resolved_config.json`，即49次 Euler 更新。
不允许用另一模型路径覆盖原训练模型。默认 seed=42，prompt seed=42+全局序号。
每个 prompt 先验证 gate=0 的 reader 路径与 native velocity 差值不超过1e-6。

这次手动固定 gate 会改变 Warm-up 的无注入生图路径。它不是 gate-trained OPD。
生图变化和较低的重建误差都不能证明图像质量更好。后续仍需语义评分。

## 在20474上运行

使用包含本次脚本的最新代码目录。进入 tmux 后执行以下命令。
不要修改正在训练使用的旧部署。模型和原训练数据必须仍在原路径。

```bash
cd /private/yida_workspace/bagel-LatentCoT-reader-eval
export PYTHON_BIN=/private/software/conda/envs/lcot/bin/python
export READER_CHECKPOINT=/private/yida_workspace/outputs/phase1a0_reader8_20474_20260930_115000_resume750/main/reader_warmup_step_0005000.safetensors
export GATE_SCALE=1.0
export NUM_SHARDS=8
export MAX_PROMPTS=16
export MIN_FREE_GIB=45
# 使用分配给自己的8张卡；已有 CUDA_VISIBLE_DEVICES 时，保留原分配。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTDIR=/private/yida_workspace/outputs/reader_injection_step5000_g${GATE_SCALE}_hard16_$(date +%Y%m%d_%H%M%S)
set -o pipefail
bash scripts/evaluate/run_bagel_reader_injection.sh "$OUTDIR" 2>&1 | tee "${OUTDIR}.launcher.log"
```

launcher 先做 CPU 配置、数据哈希和 benchmark 重叠检查，再检查可见卡数量和
每卡空闲显存。45 GiB 是可改的保守预检阈值，不保证不会 OOM。显存不足时
直接退出，不会轮询或抢占其他任务。等待显存足够后重新执行即可。
输出目录必须全新。若任一 shard 失败，不会发布完成的 HTML。

从已有 hard64 独立清单中选16条：atomicity 7/8/9/10各4条。
8个进程各加载一份模型，各处理2条 prompt，每条依次生成三组图。
成功后得到48张 PNG、`index.html`、原始 VQA 清单、三组评分映射、
checkpoint/config/代码哈希，以及每条 prompt 的噪声和时间表哈希。

只做 CPU 预检，不加载 GPU 模型：

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD" "$PYTHON_BIN" scripts/evaluate/bagel_reader_injection.py \
  --checkpoint "$READER_CHECKPOINT" --output-dir "$OUTDIR" \
  --gate-scale "$GATE_SCALE" --num-shards 8 --max-prompts 16 --dry-run
```

## 生图完成后评分

前提：已有可用的 SoftTIFA 评分服务。以下命令不会启动该服务。
将 `SOFTTIFA_URL` 改为实际地址，`OUTDIR` 指向已完成的生图目录。

```bash
export SOFTTIFA_URL=http://127.0.0.1:18086
mkdir -p "$OUTDIR/scores"
for arm in native untrained_reader step5000_reader; do
  PYTHONPATH="$PWD" "$PYTHON_BIN" scripts/evaluate/score_geneval2_server.py \
    --benchmark-data "$OUTDIR/benchmark.jsonl" \
    --image-paths "$OUTDIR/${arm}_image_map.json" \
    --server-url "$SOFTTIFA_URL" --output "$OUTDIR/scores/${arm}.json"
done
PYTHONPATH="$PWD" "$PYTHON_BIN" scripts/evaluate/geneval2_report.py \
  --benchmark-data "$OUTDIR/benchmark.jsonl" \
  --run "native=$OUTDIR/scores/native.json" \
  --run "untrained_reader=$OUTDIR/scores/untrained_reader.json" \
  --run "step5000_reader=$OUTDIR/scores/step5000_reader.json" \
  --baseline-run native --output-dir "$OUTDIR/report"
```

先比较 trained 与 untrained，判断 translation 的作用。再分别比较 native，
判断增加 reader 路径后的总效果。保留失败和负结果。16条是小样本诊断，
不能替代正式阶段的 teacher baseline 门槛或更大样本的语义验证。

计划和执行状态分别见 [PLAN.md](PLAN.md) 与 [CHECKLIST.md](CHECKLIST.md)。
