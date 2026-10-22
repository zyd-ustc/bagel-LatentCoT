# 已确认：Step2000 六榜与独立27B内部评分

用户选择文档的六榜，不再跑先前请求的GenEval2/DSG。保留六榜原prompt/seed/rubric与评分逻辑：GenEval553、GenEval++280、T2I-CompBench++2400、DPG1065、T2I-CoRe1080、WISE-Verified1000，共6378张，单个ours_step2000 arm。不运行文档的四个UniMoE模型或编辑/反思流程。

生图使用当前guidance-safe检查点Step2000的完整corrected模型，保留512px、50个采样节点、shift3、CFG4、global renorm和其guidance-safe实际修正，不照搬文档UniMoE的28步/CFG4.5。64个原训练逻辑NPU四节点分片，原始seed不变。训练已停止，不自动恢复，不启动新T2I SFT。

复用文档的独立本地27B、原逐榜评分源码/参数、依赖规则、解析与聚合；使用新run、新namespace和空缓存，不读历史票。路径前缀/data是本机同SFS的实际挂载，历史/mnt/sfs路径必须显式映射而不是修改共享源码/环境。评分只在生图释放后的自有卡启动，不碰训练奖励池/他人进程；不使用外部API或key，不升级现有环境。

结果注明27B内部协议与完整valid/expected覆盖，不称官方榜单分数，不和文档其他模型或不同采样配置的历史数值直接归因比较。先做真实NPU生图smoke，再启动64卡6378张；评分失败保留pending，不记0分，不反复重评取高。
