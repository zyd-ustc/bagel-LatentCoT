# 独立27B评分运行环境核对

2026-10-22，只读调查。范围是用户已确认的六榜、统一27B内部评分。没有启动模型或评分、安装/升级包、改环境、占用NPU、读取API密钥或停止进程。

## 结论：客户端可执行，原27B后端环境尚不具备

在本次已使用的两台主机 `modelarts-64` 与 `modelarts-job` 上，参考包的原27B后端都不能直接运行。缺的不是权重，而是原容器的 Python3.12 / 系统安装包 / CANN9 组合。不要把该环境替换成旧8B judge，也不要把未评分项记0分。6378张生图可独立继续。

### 可用部分

- 两台主机都能读取相同路径和尺寸的 `step_002000.pt`、参考包代码与27B配置，表明本任务所需的 `/data/zyd_workspace`、`/data/250010019` 和 `/data/ltj` 对两台都可见。
- 权重目录：`/data/ltj/modelscope/Qwen3.8-27B`。配置为 `Qwen3_5ForConditionalGeneration` / `model_type=qwen3_5`，`language_model_only=false`，有图像token配置；模型目录名“3.8”不能解释成已有旧Qwen3-VL-8B可代替。
- 评分客户端实际可执行：`/data/250010019/unimoe_vg_16npu/runtime/conda_training/bin/python`，实测 Python3.11.15、requests2.32.5、Pillow12.1.0。此检查只导入CPU客户端库，没有导入27B模型或初始化NPU。
- 原后端/代理命令来源：`/data/250010019/unimoe_r3/outputs/0_eval62326_fourarm_handoff_20261018/code/judge27b_command.py`。
- 原代理源码：`/data/250010019/unimoe_r3/outputs/0_qwen38max_fullbench_20261002/code/local_judge_proxy.py`（其路径由上述launcher指定；本任务父agent负责独立run、namespace与评分协议接入）。

### 确认的环境缺口

| 检查项 | modelarts-64 | modelarts-job |
|---|---|---|
| 主机标识 | 当前原64卡训练组节点零 | `ma-job-42eb50fb-8150-4c7b-b670-7cf82ae18bcc-vj-0-worker-0` |
| 原基础解释器 `/usr/local/python3.12.13/bin/python3` | 不存在 | 不存在 |
| 共享27B venv的 `bin/python3` | 指向上述不存在的基础解释器 | 同样悬空 |
| `/usr/local/Ascend/cann-9.0.0/set_env.sh` | 不存在 | 不存在 |
| 已装CANN | `ascend-toolkit/latest/version.cfg` 报8.3.RC1 | 同为8.3.RC1 |
| 历史 `/mnt/sfs` 前缀 | 不存在，实际挂载为 `/data` | 可见该前缀与相应共享目录 |

共享venv：`/data/250010019/envs/vllm_qwen38_npu_py312`。`pyvenv.cfg` 明确写：

```text
home = /usr/local/python3.12.13/bin
include-system-site-packages = true
version = 3.12.13
executable = /usr/local/python3.12.13/bin/python3.12
```

其自身site-packages里可见transformers5.5.4，但本次目录元数据检查没有发现torch/torch_npu/vllm/vllm_ascend对应安装记录；它依赖原基础Python的系统包。因此**单补symlink或换一个Python3.12二进制并不能恢复原已验证部署**。

只读搜索了当前主机的 `/usr/local`、`/opt`、`/usr/bin`、已有 `/root/venvs` 及共享的 `/data/250010019/envs`、`unimoe_vg_16npu/runtime`、`/data/ltj/{envs,venvs,runtime}`。没有找到可用的实际Python3.12二进制。其他看似可用的py312目录也都是同一悬空基础路径：

- `/data/ltj/venvs/gitcode-upload-py312/bin/python` → `/usr/local/python3.12.13/bin/python`
- `/data/ltj/envs/lmms-eval-official/bin/python3` → `/usr/local/python3.12.13/bin/python3`

已有geneval2-soft-tifa环境是Python3.11.10 / PyTorch2.7.1基础环境，不是原27B的Python3.12 / torch2.10 / vLLM0.22.1组合，不能据其可执行就宣称27B兼容。

## 评分后端恢复所需条件

请提供或协调一台**允许本任务使用、非文档禁止训练机器**的910C主机/容器，具备：

1. 可执行的CPython3.12.13及原匹配的torch2.10、torch-npu2.10、vLLM0.22.1、vllm-ascend和transformers5.5.4栈；保留参考包规定的版本和27B视觉支持。
2. 原要求的CANN9环境（含所需ATB）；不能把当前CANN8.3路径改名当9.0。
3. 同一SFS上的27B权重、原评分源码、图像与本轮空缓存目录可读。允许显式 `/data` 路径映射，不应修改历史共享源码或旧结果。
4. 两个已预留且无进程的逻辑NPU；先做模型/processor与一条真实评分验收，再进入本轮六榜评分。

目前不能给出两台已检查主机上可直接启动原27B后端的正确命令。补齐主机/容器条件后，复用原TP2后端参数与本轮独立代理，不改rubric、解析或聚合。只使用本任务生图释放后的自有卡或负责人明确协调的资源；不连接8298/8417–8424训练奖励池，不接触7984、8727、4c19/30112、a889/32238等禁止训练机器。

SSH配置另有 `modelarts-job-30072` / `modelarts-job-new`，但本调查未将“配置里存在”视作本任务访问授权，未连接该主机。若负责人确认可只读核对，可再检查它是否恰为原部署容器；不能据未检查的主机名推断其具备环境。
