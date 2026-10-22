# Step2000 benchmark：改用用户提供的评测部署包

用户授权：暂停 guidance-safe 训练，使用其Step2000，在原64个NPU上跑GenEval、GenEval2、DSG。之后用户要求先停下原评测路径搭建，参考另一项目的现有部署包，只替换生图器。

## 已完成的状态变化

- Step2000检查点已成功加载核对：`step=2000`、64份rank RNG、`correction_objective=conditional_no_anti_guidance`。
- 检查点：`modelarts-64:/data/zyd_workspace/bagel-LatentCoT/outputs/und_guidance_safe_64npu_20261022_5000/step_002000.pt`。
- 仅停止该run的四个独立进程组，逐节点核对输出路径、launcher PID和PGID后发送SIGTERM；四节点该run剩余worker均为0。
- 停止证据：该run的`_launch/user_stop_for_benchmarks.json`。旧训练heartbeat `guidance-safe-64-5000`已删除，不再以5000步为待完成目标，也不会自动续训。
- benchmark生图、评分、27B服务均未启动；没有启动新的T2I SFT。
- 已中断旧评分路径调查agent及自己的旧目录搜索；不停止其他项目进程。

## 现有包核对

参考附件：`/Users/zyd/.codex/attachments/c68de6e4-4041-4ae7-ac8c-d6e9e81f9f4c/Pasted text.txt`。

实际可读包：`modelarts-64:/data/250010019/unimoe_r3/outputs/0_eval62326_fourarm_handoff_20261018`。已读取`SCORING_NOTES.md`和`code/prepare_sixbench_scoring.py`。

关键差异：该包绑定四个UniMoE arm，六榜为GenEval、GenEval++、T2I-CompBench++、DPG、CoRe、WISE。没有GenEval2或DSG；GenEval也是本地27B自定义VLM yes/no指标，不是官方检测器指标。包生成契约固定512px/28步/CFG4.5，不可覆盖我们Step2000的512px/50节点/shift3/CFG4及guidance-safe采样语义。不要改历史包、环境、manifest或结果，不借用训练奖励池。

## 已确认执行口径

用户选择文档六榜和27B内部评分；仅一个`ours_step2000` arm，共6378张。详见PLAN.md。GenEval2/DSG不再属于本次名单。训练保持停止，不自动恢复。

原部署27B环境在当前主机不可用。用户已授权新建隔离环境，不能覆盖现有环境。judge环境负责人在新目录搭建原27B软件栈；详见JUDGES.md。生图不依赖此环境。

生图适配器`scripts/benchmark_sixbench.py`复用原包normalize和原始manifest，只替换生成模型。模型代码固定于原训练部署8d8004d，调用原`correction.sampling.sample`。每张保存原输入、seed、完整采样trace、checkpoint绑定和图像完整性信息。真实NPU smoke和64卡启动待执行。
