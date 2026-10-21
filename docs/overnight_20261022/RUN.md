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

## 02:26 heartbeat：Step636，已检查Step100语义与Step500照片对照

四节点各16个rank均正常，未发现异常；Step636 loss0.413147、同批base MSE0.657873、UND梯度0.137190，最近20步均值2.675秒。Step10/100/500 checkpoint均存在，全部日志数值有限。训练继续到已授权的5000步，不自动调参或改数据/目标。

已下载semantic_native、semantic_000100、validation_000100、validation_000500、run.json和训练日志；查看了全部32条Step100三路contact，以及8条Step500照片三路contact。本地图册：artifacts/overnight_20261022/und_postcfg_photos_64npu_20261022_5000/semantic_000100/index.html；8条Step500照片对照：同run下validation_000500/contact.jpg。可复用render_semantic.py生成后续图册。

**质量警报：移除CFG四倍放大和改用照片数据后，当前仍有明显的构图/风格迁移。** Step100“三本书”在global和full中都出现3本，不能归因于UND独有能力；马仍在西兰花左边。full把简洁厨房改成描边/做旧质感，把女孩改成强烈纹理插画，新闻标题仍有伪文字。高难组合中对象数量和属性仍有错误。

Step500照片对照更清楚：狗探出车窗的近景被full改成整条公路及两辆车，猫在窗边的场景变成户外，两名工人的场景/姿势被重排；多数global图更接近native。说明当前full的变化不止此前识别的全局常量分量，单靠本次两项修正还不能宣布画风问题解决。

同时，固定8条验证上的残差MSE已经明显超过global基线：Step500在t=1时base1.665219/global1.419890/full0.619243；t≈0.938931时0.652931/0.626486/0.422545；t≈0.860870时0.450791/0.442051/0.364801；t≈0.757576时0.401827/0.401433/0.379461。这表明full学到了常量场之外、可降低该验证目标的修正，但生图证明该指标不能替代语义与画风质量。完整数值保存在validation_summary_000500.json。后续按计划检查Step1000/3000/5000，不把本次早期负面结果直接推广到最终checkpoint。

## 02:56 heartbeat：Step1287，已检查Step1000三路对照

64个rank继续正常，四节点无错误；Step1287 loss0.405109，同批base MSE0.710391，UND梯度0.172197，最近20步2.620秒，全部数值有限。Step1000 checkpoint、semantic_001000和validation_001000均完成。未改变训练目标、数据、步数或参数；继续5000步。

已下载Step1000的64张global/corrected语义图及24张照片验证图、records和最新训练日志；复用已下载的32张native语义图，生成semantic_001000/index.html、4张contact和validation_001000/contact.jpg，并查看全部32条语义和8条照片三路对照。

此前质量警报仍成立，当前未观察到恢复：相较Step100，部分样本在Step1000出现更强的霓虹纹理和结构损失。案例009完整纠错图里微波炉不再清楚可辨，案例010三本书又变成更少的书，案例019人物面部消失，案例027/029人体结构和姿态严重改变；高难案例仍有计数/对象错误。照片验证的狗探车窗仍变成无狗近景的公路，猫在窗边仍变成户外场景。global图大多维持原生布局，但也有个别强饱和/卡通化。

固定8条验证在四个t上的base/global/full MSE分别为：t=1：1.665219/1.419999/0.545079；t≈0.938931：0.652931/0.626501/0.400563；t≈0.860870：0.450791/0.442011/0.354710；t≈0.757576：0.401827/0.401424/0.374958。full的残差目标继续改善，但实际图像没有随之改善。完整数值见artifacts/overnight_20261022/validation_summary_001000.json。

这是同一质量问题的延续，不重复叫醒用户；等待计划中的Step3000和5000结果，完成后做整体判断。

## 03:26 heartbeat：Step1976，运行正常

四节点各16个rank均无错误；Step1976 loss0.408655，同批base MSE0.697466，UND梯度0.084583，最近20步2.559秒，日志数值有限。尚无新的计划内语义验证（下一次Step3000）；Step2000 checkpoint尚未到达。继续按原配置训练，不重复下载或检查已有Step100/1000结果。

## 03:56 heartbeat：Step2647，运行正常

四节点各16个rank继续无错误；Step2647 loss0.404400，同批base MSE0.701315，UND梯度0.109796，最近20步2.584秒，日志数值有限。Step2000 checkpoint已保存，下一次计划内语义验证为Step3000，当前尚未到达。继续原配置；已报告的质量问题没有新增已检查证据，不重复通知。

## 04:26 heartbeat：Step3309，已检查Step3000三路对照

四节点各16个rank均正常，全部日志数值有限。Step3309 loss0.408255、同批base MSE0.698210、UND梯度0.046777，最近20步2.606秒；Step3000 checkpoint及semantic_003000/validation_003000均完成。训练继续到5000步，不改配置。

已下载Step2000照片验证、Step3000三路验证与64张语义global/corrected图、records和训练日志。已生成并查看semantic_003000/index.html及4张contact、validation_003000/contact.jpg，覆盖32条语义与8条照片。

质量未恢复。Step3000完整纠错把“吹风机”和“键盘+微波炉”都变成与目标对象无关的夜景/道路；“女人白裙”的脸仍消失；人体姿态样本继续严重失真；简单物体和高难组合常出现相似紫色/青色背景纹理。狗探车窗依然变成公路，教堂外景变成失真的黑白内景。global大多仍接近原生，但个别样本有饱和/图形化。当前观测已不能支持“移除CFG四倍放大+照片子集足以解决画风和语义问题”。这是此前质量警报的延续，不重复通知。

Step3000固定8条、四个t的base/global/full MSE：t=1为1.665219/1.420014/0.526780；t≈0.938931为0.652931/0.626481/0.395366；t≈0.860870为0.450791/0.442006/0.346180；t≈0.757576为0.401827/0.401422/0.363588。完整数值保存在validation_summary_003000.json。残差目标继续改善，但图像证据显示语义和画风验收仍未通过；按用户指定预算完成最后2000步，再统一报告最终结果。

## 04:56 heartbeat：Step3998，运行正常

四节点各16个rank无错误，数值有限；Step3998 loss0.409607，同批base MSE0.770477，UND梯度0.095637，最近20步2.561秒。Step4000 checkpoint尚未到达，下一次计划内语义验证为Step5000。继续原配置；没有新的语义图可检查，不重复已有检查或质量通知。

## 05:26 heartbeat：Step4673，接近预算终点

四节点各16个rank继续无错误，数值有限。Step4673 loss0.411604，同批base MSE0.728704，UND梯度0.066328，最近20步2.566秒。Step4000 checkpoint已保存；Step5000与最终验证尚未完成。按当前速度剩余纯训练约14分钟，再加最终checkpoint和生图。继续原配置，等待最终文件齐全后验收，不提前宣布完成。

## 05:56 heartbeat：5000步与最终验证完成

已确认Step5000日志、step_005000.pt、validation_005000/records.json和semantic_005000/records.json齐全。完整5000条日志均有限。最终checkpoint已实际读取：642,215,833字节、64份rank RNG、25个时间节点、1,280,000次训练统计。四节点均已无本轮训练进程。

最终语义64张global/corrected图及24张照片图、全部records/轨迹和四节点日志已下载；复用32张native语义图，已生成并查看全部三路对照。四个计划语义检查点均验证了32条记录、三路图片和每张49步轨迹的完整性。最终图册已实际浏览器渲染检查：http://127.0.0.1:8770/index.html。

最后100步平均loss0.400167，对照base MSE0.700796。完整UND的固定验证MSE继续下降，但最终仍有对象消失、面部缺失、关系错误和重复彩色背景；画风及语义验收未通过。不追加训练、不提升本checkpoint为可用生成模型。完整结论见RESULTS.md，32项逐条观察与数值系列见本地artifacts。完成后删除heartbeat。

最终验收后已删除自动检查任务`64-5000`。没有启动后续训练。
