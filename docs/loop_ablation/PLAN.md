# Step4000 画风退化的小规模消融

问题：画风变化首先来自 parent Prompt KV、Memory、入口回灌，还是 CFG 放大？
父运行：geneval2_hard_feedback_sweep_step4000_20261021。固定 step4000、seed17。
GenEval2 atom_count=10 抽8条；GenEval六类覆盖共8条；DPG、DSG各8条。
固定 selection seed17，保留原文和标注，32条×8组=256图。

## 对照及解释
- base4 vs parent4：只替换 UND parent，不开启 Loop。
- parent4 vs loop1：首次反馈的全部影响。
- loop1 vs masked1：屏蔽 Memory；保留位置偏移、consumer、回灌。
- loop1 vs noentry1：关闭入口回灌；保留 Memory、consumer。
- parent4→loop1 vs parent1→loop1cfg1：区分 CFG 本身与其对 Loop 的放大。
- loop1 vs loop2：已训练范围内增加反馈次数。

每图记录逐时间点实际遍历次数，条件速度差异RMS比与余弦值。先小模型检查控制路径，
再 modelarts-job 16卡采样。独立代码快照，不改64卡训练。无付费API、不启动自动裁判。
页面沿用户要求供人工查看，不把速度幅度或人工抽查当作官方benchmark分数。

工具限制：技能提到的 bash_exec、memory、artifact 工具在本会话不可用，已发现并确认；
使用现有执行工具、计划及运行JSON留存证据。
