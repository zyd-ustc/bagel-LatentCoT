# 早期 Memory probe：代码检查

日期：2026-10-05。H200 代码目录：`/private/yida_workspace/bagel-LatentCoT-main-early-memory-20261005`。

执行的是小模型接口与合成数据测试，未加载正式 BAGEL/Qwen 权重，未启动正式生成、评分、Memory 问答、图像标注或预算评测。

- H200 `lcot` Python、`CUDA_VISIBLE_DEVICES=""`：32 passed，3 skipped，6.56 秒。
- 3 项跳过检查依赖 CUDA；本次未执行。
- Python 编译、shell 语法和自有代码的 Git 空白检查通过。ImageTransform 保留原生源文件的空白格式。
- Native ViT ImageTransform 在现有 torchvision 0.20.1+cu124 环境成功导入。
- BAGEL QA 与 Qwen3-VL 标签入口的 CLI 参数可解析；研究协议 YAML 可解析。

测试覆盖：原生完整 UND 层顺序、每层不同 KV 长度、因果 attention、答案候选分批与单独计算的一致性、KV 不被问答写回、Memory 快照不改变 GEN 运算、x0 捕获不改变采样轨迹、快照/图像哈希绑定、unknown 排除规则、要求/观察不一致时的 prompt 复述、按 prompt 聚类统计、评分 shard 完整覆盖与 provenance 拒绝。

这些结果证明代码合同，不证明 Memory 可读出语义信息或最终 T2I 质量。完整模型与正式评测的结果由用户运行后取得。
