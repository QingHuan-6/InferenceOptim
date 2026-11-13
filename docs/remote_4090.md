# 单张 4090 首次运行

本说明从真实普通解码开始：先用 Qwen2.5-1.5B-Instruct 核对输出和测量，再扩展模型。
不训练模型，不开启量化、投机解码、多卡或并发服务。

## 上传与环境

日常同步推荐 GitHub 私有仓库，部署与更新命令见 [Git 工作流规范](git_workflow.md)。
以下 ZIP 方式适用于首次临时上传或 GitHub 网络不可用的情况。

本地执行 `.venv/Scripts/python.exe scripts/package_remote.py` 生成独立上传包。
将 ZIP 上传到自己的服务器工作目录并解压，进入 AInferenceOptim 目录。
包内不包含本地虚拟环境、模型、实验结果或 AutoDL 抢卡工具与登录资料。

服务器需要 Linux、Python 3.11–3.13、可用的 NVIDIA 驱动，以及模型与依赖下载网络。
推荐新建 Python 3.11 环境；安装脚本在 `.venv-4090` 中隔离依赖。

```bash
nvidia-smi
python3 --version
bash scripts/setup_4090.sh
```

脚本安装 PyTorch 2.9.0 的 CUDA 12.8 wheel 和 Transformers 4.57.1。
这些是本项目固定的初始组合，不要求使用最新版；驱动必须支持该 wheel。
脚本只检查并安装，不下载预训练权重、不启动计费任务、不自动关机。
若 Python 入口不同，可用 `PYTHON_BIN=/path/to/python bash scripts/setup_4090.sh`。

如果平台镜像自带 PyTorch，不要直接覆盖其环境。若安装失败，先保存报错和
`nvidia-smi` 输出再定位版本问题，不要无序升级所有依赖。

## 模型缓存与版本

模型下载到指定目录、数据集来源、HumanEval 转换和本机配置生成的完整步骤见
[README](../README.md)。推荐按 README 生成 configs/local 下的配置并运行。
以下为原始预设使用 Hub 缓存的替代方案，不自动复用自定义 MODEL_DIR。

模型缓存应放到自己确认可保留的数据盘目录，下面的路径必须替换：

```bash
export HF_HOME=/your/persistent/disk/huggingface
mkdir -p "$HF_HOME"
```

每次新终端都要设置相同 HF_HOME；可写入自己的终端配置。
默认 run/verify 在缺少缓存时下载模型。也可以先单独预下载，观察网络是否正常：

```bash
.venv-4090/bin/hf download Qwen/Qwen2.5-1.5B-Instruct
```

配置中的 revision=main 会先解析为 commit，权重和分词器随后使用同一个 commit；
实际版本写入后端 metadata。正式实验前把 smoke 与 baseline 两份配置的 revision
统一改为已记录的 commit。若使用本地权重，model 必须写绝对路径或以 ./、../ 开头，
并自行保存来源 commit；本地文件目前不计算完整权重哈希。
缓存齐全后可设置 local_files_only=true，离线运行；不应上传账号令牌或凭据。

## 先核对普通解码输出

```bash
.venv-4090/bin/python -m ainference_optim verify --config configs/4090_smoke.toml --samples 3
```

命令逐 token 比较手写 KV Cache 循环与库内 generate 的输出，写入独立目录的
verification.json。退出码 0 表示所选样本全部一致，2 表示存在不一致，1 表示执行错误。
失败时停止性能实验，先定位原因。验证只覆盖所选贪心请求，不证明采样分布等价。
比较采用独立 GenerationConfig，关闭采样、惩罚和强制 token 等默认生成策略。

### 短输出通过而长输出失败

smoke 最多检查 32 token，baseline 检查 128 token；后续位置发生分叉时，两者可以分别通过和失败。
不要把晚期不一致自动判成可忽略的 BF16 误差，更不能直接开始正式性能对照。
查看 verification.json 中每条请求的 actual/reference、长度和 first_mismatch_index（从 0 起），
并保留模型来源、torch/transformers 版本、精度和 attention 设置。
TF32 旧接口警告与 token 不一致是不同问题，去掉警告本身不证明正确性已恢复。

当前修订对齐了手写与参考路径的输入准备、DynamicCache 和末位 logits 投影，并替换 TF32 API。
CPU 随机小模型的 128-token 检查不能代替目标机器验收；需要用原 BF16/SDPA 配置重新 verify。
若仍不一致，再用独立配置作 eager attention 或 FP32 的定位对照，保留所有报告；
不得修改原报告、删除失败样本或用不同精度结果宣称原 BF16 配置通过。

## 冒烟与初步基线

```bash
.venv-4090/bin/python -m ainference_optim run --config configs/4090_smoke.toml
.venv-4090/bin/python -m ainference_optim verify --config configs/4090_baseline.toml --samples 3
.venv-4090/bin/python -m ainference_optim run --config configs/4090_baseline.toml
.venv-4090/bin/python -m ainference_optim summarize outputs/实际目录名
```

先逐条确认命令成功，再执行下一条。smoke 最多生成 32 token，遇 EOS 停止；
baseline 忽略 EOS，固定生成 128 token，仅用于性能测量，末尾文本可能不自然。
两者都应用模型聊天模板，token 计数包含模板输入和实际生成的特殊 token。

目前两份配置都使用三个短管线样本，不代表 128/512/2048 token 的完整研究数据集，
也不能用其 P99 证明尾时延结论。正式实验前需准备长度受控且可公开的数据集。

## 计时和结果解释

指标使用 host 单调时钟：请求前同步，起点包含模板渲染、分词和输入传输；
每个 argmax token 经 item() 到达 CPU 后记录时间，包含其隐式设备同步开销。
末尾文本解码和最终同步计入 elapsed_s，不计入 token 间隔。模型加载不计时。
这是一条可观测的研究基线，不能把它当成高并发部署引擎的极限性能。

max_memory_allocated 为 PyTorch 分配器的请求期峰值，包含常驻权重，
不含所有驱动/非 PyTorch 占用，也不等同于 nvidia-smi 读数。
每次请求重置峰值统计，不在请求之间 empty_cache，避免改变缓存分配行为。

manifest 的 synthetic=false 表示确实执行模型计算，不表示已经完成 4090 验收；
必须同时保留目标机器上的 verification.json 和实验结果。metadata 记录设备、
驱动、运行时、精度、attention、模型 commit 与生成策略，未知驱动信息明确标为 unavailable。

本地 CPU 随机小模型测试能验证接口与 KV 循环，但不能验证 BF16 CUDA 数值、
显存或 4090 加速效果。目标设备的首次输出核对和实测仍是必需步骤。

## 测试、保存与关机

```bash
.venv-4090/bin/python -m unittest discover -s tests -v
.venv-4090/bin/python -m pip freeze > outputs/environment.txt
nvidia-smi > outputs/nvidia-smi.txt
```

将整个输出目录、环境信息与使用的配置下载保存，再到平台确认关机状态。
关闭 SSH、终端或抢卡工具不会结束云端计费；GPU 关机后磁盘是否继续收费要另外确认。

## 实现参考

- [Transformers 4.57.1 KV Cache 文档](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)
- [Transformers 4.57.1 GenerationConfig 文档](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation)
- [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)
- [Qwen2.5-1.5B-Instruct 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
