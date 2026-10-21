# Base / 1 / 2 / 3 / 4 次反馈：GenEval2 人工比较

> 历史 Stage A v4 记录：以下配置、检查点、运行状态和测试结果属于重构前版本。当前实现请使用 [Stage A v2](stage_a_v2_canvas_reentry.md)，旧检查点不能加载。


## 固定的条件

使用 Stage A `step_004000.pt`，全部组输入同一条 GenEval2 原始完整 Prompt，初始噪声 seed 17。
Base 是原生 BAGEL：恢复原版 UND 权重，关闭全部 Loop 和 adapter。
1–4 指每个循环时间点在一次原生 GEN 之后执行的反馈次数，即分别执行 2 / 3 / 4 / 5 遍全层条件 GEN。
只有前 25 次 Euler 更新启用 Loop，后半程为原生采样；全部 28 个 block 都参与每次反馈。

checkpoint 训练时有两次反馈、两个速度门控。推理第一次反馈用第一个门控，第二次用第二个，
额外第三、四次继续复用第二个门控。Writer、consumer 和 ReentryBridge 共享原有权重。
没有额外放大 gate、没有新增参数、没有改动 checkpoint 或正在运行的 64 卡训练。
可用 `feedback_updates` 覆盖推理轮数；训练默认值及 checkpoint 恢复条件保持不变。

## 高复杂度 Prompt

GenEval2 的原始 800 条中，`atom_count` 分布为 3–10，每档 100 条。
从最大档 `atom_count=10` 的 100 条中固定 selection seed 17 抽取 32 条，
保留原始文本、ID、全部 VQA 问题、答案和 skills，不根据本模型的成绩选题。
这里的“高难”指原始标注的结构复杂度，不是官方 hard split，也不是已测出的失败率。

共 32 条 × 五组 = 160 张图。页面展示五列，可切换 Base 与指定反馈轮数两列比较，
点击图片查看原尺寸，展开原始问题逐项检查。没有自动 GenEval2 打分。

## modelarts-job 运行

```bash
cd /data/zyd_workspace/bagel-LatentCoT/code/bagel-feedback-sweep-20261021
PYTHON=/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python
RUN=/data/zyd_workspace/bagel-LatentCoT/outputs/geneval2_hard_feedback_sweep_step4000_20261021
PYTHONPATH=. "$PYTHON" scripts/sample_feedback_sweep.py prepare --output "$RUN" \
  --checkpoint /data/zyd_workspace/bagel-LatentCoT/outputs/stage_a_full_v1_64npu_20261020_20000/step_004000.pt
bash scripts/feedback_sweep_16.sh "$RUN"
```

需要新输出目录和空闲设备。脚本固定 checkpoint 路径，16 个独立 worker 分片采样。
每张图附带门控索引、时间、显存和逐步执行记录；汇总会核对实际遍历次数、图片完整性与输入身份。
重新生成页面可用 `sample_feedback_sweep.py render --output ...`，不会重新生图。

## 推理 API

```python
# engine 已从训练 checkpoint 加载。省略 feedback_updates 时仍用训练配置的两次反馈。
base, _ = engine.generate(prompt, seed=17, native=True)
image, _ = engine.generate(prompt, seed=17, feedback_updates=4)
```

覆盖参数只用于推理，不通过修改 `student.config` 或重建 `vel_scale` 改轮数。
已检查默认两轮数值一致、1–4 轮的真实全层执行次数、最后一轮 CFG、最后门控复用，
以及 GEN RoPE 偏移不随反馈次数累加。训练路径仍接受原有轮数和监督权重约束。

## 本次运行结果

2026-10-21，modelarts-job 的 16 个 worker 全部完成：32 条 Prompt、160 张 PNG。
逐步执行记录通过核验；全部图片通过解码检查。单卡峰值显存最大 31.00 GiB。
本地页面：`artifacts/geneval2_hard_feedback_sweep_step4000_20261021/index.html`。

验证：完整测试首次运行 96 通过、1 失败；失败是新测试对 tuple 位置编码使用了标量加法。
修正测试断言后，反馈专项 4 项全部通过；运行时代码未因此修改。
