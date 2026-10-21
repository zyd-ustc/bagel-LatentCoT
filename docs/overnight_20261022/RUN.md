# Post-CFG photo correction：64卡、5000步

## 当前运行

2026-10-22 01:55（北京时间）在 modelarts-64 启动，4节点×16NPU。训练代码已提交并推送：`da7a915cffccd6a39d458fbfbbf785b6f3010802`，分支 `loop-layer-NPU`。本文件会记录后续验收快照；未完成5000步及最终验证前，不记为训练完成。

- 固定代码：`/data/zyd_workspace/bagel-LatentCoT/code/bagel-postcfg-photo-20261022`
- 输出：`/data/zyd_workspace/bagel-LatentCoT/outputs/und_postcfg_photos_64npu_20261022_5000`
- 四节点日志、PID、启动清单、配置：输出路径加 `_launch`
- 照片索引：`/data/zyd_workspace/bagel-LatentCoT/datasets/correction_photos_v2_20261022`
- 原生权重：`/data/zyd_workspace/bagel-LatentCoT/models/Bagel-7B-MoT`
- Python：`/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python`
- 全局 batch256，每rank顺序累计4张，512px，LoRA64，lr1e-4，warmup100，clip1。全新初始化，不续训旧1000步模型。

## 三项修改

1. 原生GEN的CFG/renorm先完成。训练target=`epsilon-x0-v_guided`，proposal=`x_t-t*v_guided`；推理在同一guided速度之后只加一次FP32纠错，不再将纠错乘CFG4。纯MSE不变，原生零纠错路径保持一致。
2. 使用COCO/Flickr30k人工描述照片，按有效图文对数采样。检查150,068张独立路径的完整解码；移除211个重复图文对、1条编辑式caption、135条小图片记录、5,795条跨集合冲突记录。有效train743,062/val734/test720。27个独立路径低于最小尺寸；没有解码损坏记录。原始语料未改动。保留test>val>train的原有split身份，移除同图训练别名；validation再按图片去重。`selection.json`保留完整来源与筛选记录。来源限制和结构检查不等于逐条核对所有语义。
3. 在同一批训练残差上估计每个t的64维全局常量场，作为精确样本均值最小二乘基线。不增加loss、参数或优化器耦合。checkpoint包含其累计和与计数。固定验证同时比较native/global/corrected。

## 验收证据

- 16项CPU测试通过。包括真实tiny MoT训练、原生CFG一致性、纠错只加一次、全局基线训练统计、保存恢复、精确续训、数据过滤、同图跨split冲突，以及三路评估。
- 完整模型NPU pilot4步通过，峰值约47.94GiB。验证零初始化、head先收到梯度、第二步UND收到梯度、冻结原生参数、保存恢复和优化器/采样器/RNG续训。三路图片已下载并查看。
- pilot路径：`/data/zyd_workspace/bagel-LatentCoT/outputs/und_postcfg_photo_pilot_20261022`。
- pilot的全局基线为了覆盖25个t，使用一个训练pair补齐空节点；该小样本均值图仍明显插画化。这不是正式全局基线的质量证据。正式运行的全局统计只来自实际全batch训练数据。
- 本地证据：`/Users/zyd/Documents/LCoT-codex/bagel-loop-layer-NPU/artifacts/overnight_20261022`。

## 自动输出

保存Step10/100/500/1000/2000/3000/4000/5000。

- `validation_######/records.json`：8条不重图验证样本，4个t上的native/global/full残差MSE；以及24张三路图片。
- `semantic_000100`、`semantic_001000`、`semantic_003000`、`semantic_005000`：固定32条语义prompt，包含计数、位置、属性及复杂组合。保存global/corrected图与原始benchmark标注。native图片在同一个run的`semantic_native`复用；seed均为17。
- 语义结果需实际检查图像或运行相应benchmark评测；残差MSE不是语义成功率。

## 健康检查与恢复

在本地查看健康快照：

```bash
ssh modelarts-64 '/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python /data/zyd_workspace/bagel-LatentCoT/outputs/und_postcfg_photos_64npu_20261022_5000_launch/status.py'
```

完成条件：Step5000日志、`step_005000.pt`、`validation_005000/records.json`和`semantic_005000/records.json`全部存在且数值有限。训练进程与SSH断开独立；不要以本地连接状态判断训练停止。

本线程heartbeat `64-5000`每30分钟检查。正常未变化时保持安静，完成/失败时报告；新验证结果下载后检查并记录。完成后关闭该heartbeat。

发生故障时先读四节点日志并确认原因，保留checkpoint。训练CLI支持在同一代码、配置、语料及64-rank布局下加`--resume /absolute/path/step_######.pt`恢复到5000总步数。恢复需重新启动同一4节点的torchrun，保持NODE_RANK=0..3、MASTER_ADDR为原node0、NPROC_PER_NODE=16、NNODES=4；使用空闲端口和新的恢复日志。现有fresh-run启动器会拒绝覆盖旧输出，不能直接再次调用它当作恢复入口。不得停止无关作业或将旧1000步混合数据checkpoint拿来续训本轮。

## 启动验收快照

64个rank均已载入，每节点16个。Step10检查点已实际读取验证：包含64份rank RNG、25个时间节点的全局基线统计，累计2560个训练样本；所有时间节点均有样本。代码revision和post_cfg配置正确。

本地下载日志时已到Step32/5000：loss0.562028，同批原生guided MSE0.699720；最近20步均值2.617秒/步，剩余纯训练约3.61小时，另加checkpoint与固定生图。预计北京时间今早6点前后完成。启动验收时四节点均无错误，所有loss/梯度有限，UND梯度非零，峰值约48.16GiB。以上为启动状态，不是最终质量结果。

启动/健康/数据/pilot证据均保存在本地artifacts/overnight_20261022；heartbeat `64-5000`已创建并处于ACTIVE。
