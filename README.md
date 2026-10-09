# BAGEL：连续完整 UND Memory loop（NPU）

本页对应 `loop-layer-NPU` 分支。Memory 每轮从0层连续经过全部28层 UND，末层输出回送下一轮0层。GEN仍在 `[0,8)` 重算，suffix只执行一次。原生权重、全量prompt容量、特殊token固定、同层UND投影、CFG和采样不变。不处理出口回送的深度错配，不添加adapter/gate/alpha。

架构与命令见 [DENOISER_INTERNAL_MEMORY_PLAN.md](docs/DENOISER_INTERNAL_MEMORY_PLAN.md)。默认正式配置 `configs/window_comparison.json`：800题、seed0、Base＋Early10/Early20×R1/2/3/4，共7200图。小规模配置 `configs/loop_layer_npu_pilot.json`：32题，同样9组，共288图。全部设置 `memory_update=full_depth`。历史各层独立更新实现保留在und_state_loop.py，配置memory_update=legacy_layerwise可明确选择旧路径。

代码目录 `/root/bagel-LatentCoT-loop-layer-NPU`，Python `/root/venvs/bagel-NPU/bin/python`。正式评测由用户执行，脚本先检查真实权重数值，再生成、评分并导出comparison.html。必须使用新目录，不能续跑旧分支源码绑定的plan。数值检查不证明语义增益或质量保持。
