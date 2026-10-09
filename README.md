# BAGEL：分层 UND Memory loop（NPU）

本页对应 `loop-layer-NPU` 分支，从 NPU 提取已存在的旧循环。运行入口默认选择旧 `loop_grid`，不启用图像观察、文字反馈或固定 observation cache。旧循环实现和 Ascend 运行后端没有改动。

架构与执行命令见 [DENOISER_INTERNAL_MEMORY_PLAN.md](docs/DENOISER_INTERNAL_MEMORY_PLAN.md)。默认正式配置为 `configs/window_comparison.json`：800 prompts、seed0、Base＋Early10/Early20×R1/2/3/4，共7200张图。小规模配置为 `configs/loop_layer_npu_pilot.json`：32 prompts，同样9组，共288张图。两者均为模型层窗口 `[0,8)`、512px、50点原生采样。

正式生成和评分由用户执行。脚本先检查真实权重数值，再生成、评分并导出 comparison.html。旧评测目录不可用新分支源码续跑。

NPU 环境为 ModelArts 主机 `/root/bagel-LatentCoT-loop-layer-NPU`，Python 为 `/root/venvs/bagel-NPU/bin/python`。main 与 NPU 分支保持各自实现。本分支保留来源中的其他模块和数值测试作为参考，但默认入口只选择旧循环。
