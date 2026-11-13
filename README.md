# AInferenceOptim

毕设《推理引擎中的解码加速优化研究》的实验基础设施。

当前完成硬件无关的串行实验管线：严格配置校验、JSONL 数据加载、后端协议、
请求级指标、原始结果落盘、实验汇总与失败记录。
现有 mock 和 Transformers 普通贪心解码后端；后者支持单设备 CUDA 或显式 CPU 功能测试。
当前不包含昇腾、量化、投机解码或并发调度。4090 性能须在租用设备上实际验证。

租到机器前可先打包：`python scripts/package_remote.py`。
服务器安装、真实模型输出核对及实验步骤见 [4090 运行说明](docs/remote_4090.md)。
已完成冒烟测试后，按 [第一轮原生生成基线](docs/baseline_round1.md) 准备固定输入并运行正式基线。
run 默认显示预热和正式请求进度、轮次及预计剩余时间；使用 --quiet 可关闭。
两组完成后先分析现有结果；timing-overhead 是按需诊断，不是必跑流程。
计时与后续算法开发须先阅读 [源码审阅与实验成本约定](docs/source_review.md)，
不以未证实的开销猜测要求重复付费测试。
日常通过 GitHub 私有仓库同步；分支、AI 提交边界与服务器更新流程见
[GitHub 与 AI Git 规范](docs/git_workflow.md)。

首次部署按以下顺序操作：
[环境配置](#服务器环境配置) → [模型下载](#服务器安装和模型下载) →
[数据集准备](#数据集下载与输入格式) → [运行实验](#运行真实模型)。

## 服务器环境配置

下面是 Linux 服务器上的真实推理环境；Windows 本地 mock 开发环境见后文。
**先安装环境，再下载模型和数据。安装脚本不包含模型或数据集下载。**

| 项目 | 本项目初始配置 |
| --- | --- |
| 系统 | Linux x86_64，初始建议使用 Ubuntu 22.04/24.04 镜像 |
| GPU | 单张 NVIDIA RTX 4090，标准 24GB 显存；检查实际设备与可用显存 |
| Python | 推荐 3.11；固定推理组合限定 3.11–3.13 |
| PyTorch | 固定 2.9.0，服务器安装 CUDA 12.8 wheel（cu128） |
| Transformers | 固定 4.57.1 |
| 精度与设备 | BF16、cuda:0，batch=1；不可用时直接报错，不自动回退 CPU |
| 推理环境目录 | 项目根目录下 .venv-4090，与平台自带环境隔离 |
| 开发工具 | inference、dev 两个可选依赖组，具体声明见 pyproject.toml |

这是准备使用的 GPU 环境组合，尚未在你的租用机器验收。
本地 CPU 功能测试使用 torch 2.9.0+cpu，不能直接复制该环境到 GPU 服务器。
安装依据见 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/)。

### 1. 检查机器和 Python

在服务器项目根目录执行：

```bash
nvidia-smi
python3 --version
```

确认设备型号、显存、驱动及 Python 版本。nvidia-smi 显示的 CUDA Version 是驱动支持信息，
不等于当前 Python 中 PyTorch wheel 使用的 CUDA runtime 版本；安装完成后查看 torch.version.cuda。
驱动必须兼容 cu128；加载失败时先检查驱动与 wheel，不通过安装 CPU 版 PyTorch绕过。
这条路线使用预编译 wheel，当前无需编译自定义 CUDA 内核。

如果 Python 已处于 3.11–3.13，直接进行下一步。若平台已提供 Conda，但默认 Python 不合适，
可以先准备一个 Python 3.11 入口，再用它创建项目的独立 venv：

```bash
conda create -n ainfer-py311 python=3.11 -y
conda activate ainfer-py311
PYTHON_BIN="$(command -v python)" bash scripts/setup_4090.sh
```

这组命令仅适用于已有且能激活的 Conda；没有 Conda 时，选择带 Python 3.11 的平台镜像，
或让 PYTHON_BIN 指向已经安装的兼容 Python。不要将上面的 Conda 命令与手动安装重复执行。

### 2. 一键安装环境

默认使用 python3 建立环境；Python 已兼容时执行：

```bash
bash scripts/setup_4090.sh
```

指定其他兼容 Python 入口时执行：

```bash
PYTHON_BIN=/你的Python绝对路径/python bash scripts/setup_4090.sh
```

脚本会检查机器、创建 .venv-4090、安装 GPU PyTorch 与项目、核对 CUDA/BF16，最后执行 pip check。
失败后先排查报错，安装成功前不要继续下载模型和运行实验。

### 3. 手动安装方式

下面与脚本是两种安装方式，选择一种即可。先确认 python3 为兼容版本：

```bash
python3 -m venv .venv-4090
.venv-4090/bin/python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
.venv-4090/bin/python -m pip install -e '.[inference,dev]'
```

顺序不能颠倒：先显式装 GPU wheel，再安装项目可选依赖。
使用 -e 是为了让后续 git pull 的源码修改立即生效；依赖变更时仍需重新安装依赖。
不要上传 Windows 的 .venv，也不要直接覆盖平台镜像自带的 PyTorch。

### 4. 验证环境

安装成功后执行，不需要下载模型：

```bash
.venv-4090/bin/python - <<'PY'
import sys
import torch
import transformers

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA 不可用，请检查驱动及安装的 wheel"
with torch.cuda.device(0):
    print("GPU:", torch.cuda.get_device_name(0))
    print("Total VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
    print("BF16:", torch.cuda.is_bf16_supported())
    assert torch.cuda.is_bf16_supported(), "当前设备不支持配置的 BF16"
PY
.venv-4090/bin/python -m pip check
.venv-4090/bin/python -m ainference_optim --help
```

应能看到 4090、PyTorch 的 CUDA 12.8 runtime、BF16=True，pip check 无依赖冲突，
CLI 帮助中有 run、verify、summarize。单纯 import 成功或显示 CPU 版版本号不代表 GPU 环境可用。
确认后再按下文设置 MODEL_DIR、DATA_DIR 并下载。

### 5. 新终端和常见问题

本文始终使用 .venv-4090/bin/python 等完整相对路径，因此无需 activate。
如希望直接使用 python、pip、hf，可在每个新终端的项目根目录先执行：

```bash
source .venv-4090/bin/activate
```

模型路径等 export 变量仍需在新终端重新设置；activate 不会自动恢复它们。

| 现象 | 处理方向 |
| --- | --- |
| Python 版本不符合范围 | 更换兼容 Python 入口或平台镜像，再创建环境 |
| venv / ensurepip 缺失 | 使用支持 venv 的 Python；Ubuntu 系统 Python 可由管理员补齐对应 python3-venv 包 |
| torch.cuda.is_available() 为 False | 检查是否装成 CPU wheel、nvidia-smi 是否正常、驱动是否兼容；不要用 CPU 回退冒充验收 |
| 下载 wheel 或模型失败 | 分别检查 download.pytorch.org、包索引或 huggingface.co 的连通性，保存完整报错 |
| pip / python 指向平台环境 | 使用本文 .venv-4090/bin/ 下的命令，避免混用环境 |
| GPU 显存不足 | 用 1.5B、batch=1、短输入先测，检查其他进程占用；不自动改变精度或模型 |

## 已确定的评价指标

第一阶段主结果为 **解码耗时与解码加速比**，同时报告 TPOT、端到端延迟和加速比、TTFT、
实际单请求解码速率、峰值显存、贪心序列匹配率及任务质量。投机阶段增加接受率、每轮推进 token 数
和开销分解；并发阶段再评价双 SLO 达标率、服务容量和尾时延。
定义、聚合方式、任务书目标和实现状态以 [评价指标约定 V1.1](docs/evaluation_metrics.md) 为准。
正确性检查、输出一致性、任务质量和性能分别报告；严格验证失败仍保留失败状态，
可继续采集待验证基线，但不能将其描述为已经通过无损验收。
P99 请求平均 TPOT 与 P99 逐 token TBT 分开报告，不能互换。

## 模型和数据集保存位置

| 内容 | 来源 | 位置与状态 |
| --- | --- | --- |
| 默认真实模型 | [Qwen2.5-1.5B-Instruct 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 不随 Git 下载，按下文下载到指定目录 |
| 分词器与权重 | 与模型同一版本的仓库文件 | 下载模型时一并获取 |
| 3 条流程样本 | 本项目编写，对话/代码/JSON 各 1 条 | data/smoke.jsonl，随 Git 下载，无需额外下载 |
| 正式/外部数据集 | 按论文依据选择 | 当前未自动下载；建议放 data/private/ 或独立数据盘 |
| 结果 | 实验运行生成 | outputs/，不入 Git |

现有两套 4090 配置都读取 3 条流程样本，不能当成完整研究基准。
正式负载的论文依据、数据集选择及指标区别见 [评测文献记录](docs/benchmark_references.md)。
其中 vLLM（SOSP 2023）使用 ShareGPT/Alpaca，DistServe（OSDI 2024）使用
ShareGPT/HumanEval/LongBench，SpecInfer（ASPLOS 2024）使用 5 组真实提示。

## 服务器安装和模型下载

在确认可保留的数据盘工作目录中 clone 项目，配置私有仓库读取权限后执行：

```bash
git clone git@github.com:QingHuan-6/InferenceOptim.git AInferenceOptim
cd AInferenceOptim
bash scripts/setup_4090.sh
```

安装脚本创建 .venv-4090 并安装固定版本推理依赖，**不下载模型或数据集**。
以下命令均在项目根目录的同一个 Bash 终端中执行。若项目不在可保留的数据盘，
将 AINFER_STORAGE 改为自己的持久化绝对路径：

```bash
export AINFER_STORAGE="$PWD"
export HF_HOME="$AINFER_STORAGE/.cache/huggingface"
export MODEL_DIR="$AINFER_STORAGE/models/Qwen2.5-1.5B-Instruct"
export DATA_DIR="$AINFER_STORAGE/data/private"
mkdir -p "$HF_HOME" "$MODEL_DIR" "$DATA_DIR"
export MODEL_REVISION="$(.venv-4090/bin/python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("Qwen/Qwen2.5-1.5B-Instruct").sha)')"
test -n "$MODEL_REVISION" && \
  .venv-4090/bin/hf download Qwen/Qwen2.5-1.5B-Instruct \
    --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR" && \
  printf '%s\n' "$MODEL_REVISION" > "$MODEL_DIR/UPSTREAM_REVISION.txt"
```

确认下载成功后，MODEL_DIR 中应有 config.json、分词器文件和 .safetensors 权重；
UPSTREAM_REVISION.txt 记录本次下载的 commit。模型、缓存与 data/private 已被 Git 忽略。
新终端需要重新设置路径变量，下载失败时使用同一版本和目录重试。

### 配置使用本地模型

预设中的 model 是远程模型 ID，不会自动读取自定义 MODEL_DIR。
下面生成本机配置，使用下载好的绝对路径和离线读取，并修正配置搬入 configs/local 后的路径：

```bash
.venv-4090/bin/python - <<'PY'
import json
import os
from pathlib import Path

root = Path.cwd()
model = Path(os.environ["MODEL_DIR"]).resolve()
assert (model / "config.json").is_file(), "模型尚未下载完成"
revision = (model / "UPSTREAM_REVISION.txt").read_text().strip()
target = root / "configs/local"
target.mkdir(parents=True, exist_ok=True)
for name in ("4090_smoke", "4090_baseline"):
    text = (root / f"configs/{name}.toml").read_text(encoding="utf-8")
    replacements = {
        'model = "Qwen/Qwen2.5-1.5B-Instruct"': f"model = {json.dumps(str(model))}",
        'revision = "main"': f"revision = {json.dumps(revision)}",
        'dataset = "../data/smoke.jsonl"': f"dataset = {json.dumps(str(root / 'data/smoke.jsonl'))}",
        'output_dir = "../outputs"': f"output_dir = {json.dumps(str(root / 'outputs'))}",
        'local_files_only = false': 'local_files_only = true',
    }
    for before, after in replacements.items():
        assert before in text, f"预设已变化，请人工检查：{before}"
        text = text.replace(before, after)
    with (target / f"{name}.toml").open("x", encoding="utf-8") as handle:
        handle.write(text)
PY
```

configs/local 不入 Git，已存在的文件拒绝覆盖；需改路径时编辑现有文件。
模型目录是本地文件时，来源版本由配置和 UPSTREAM_REVISION.txt 保存；目前不计算完整权重哈希。

若直接运行原始预设，模型会使用 Hub 缓存，缺少缓存时自动下载，常见位置为
$HF_HOME/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/COMMIT/；未设置 HF_HOME 时
通常位于 ~/.cache/huggingface/hub/。HF_HUB_CACHE 等已有设置可以改变缓存位置，检查方法：

```bash
.venv-4090/bin/python -c 'from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)'
```

## 数据集下载与输入格式

先用 Git 自带的 data/smoke.jsonl 跑通流程，不需要额外下载数据。
正式负载建议先准备指令/对话和代码输入，再扩展长上下文；不要默认运行 32K。
数据来源和使用依据：

| 数据集 | 获取入口 | 用途与当前状态 |
| --- | --- | --- |
| Alpaca | [Stanford 官方仓库](https://github.com/tatsu-lab/stanford_alpaca) | 指令输入；需将 instruction/input 合并，转换器尚未实现 |
| ShareGPT | [清洗版数据卡](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) | 对话输入；须固定文件版本及轮次抽取规则，转换器尚未实现 |
| HumanEval | [OpenAI 官方仓库](https://github.com/openai/human-eval) | 代码输入；下文提供原始下载和 prompt 转换示例 |
| LongBench | [THUDM 官方仓库](https://github.com/THUDM/LongBench) | 长上下文扩展；须选定 v1 摘要子集与模板，尚未接入 |

### HumanEval 可选下载示例

需要 curl，以及前面设置的 DATA_DIR。只下载数据，不安装或执行其评测程序。
上游仓库已注明不再维护；这里用于固定已有代码提示。

```bash
curl -fL --retry 3 \
  https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz \
  -o "$DATA_DIR/HumanEval.jsonl.gz" && \
  sha256sum "$DATA_DIR/HumanEval.jsonl.gz" > "$DATA_DIR/HumanEval.jsonl.gz.sha256"
```

确认下载成功，再转换为本项目格式；原始压缩包不能直接传给 dataset：

```bash
.venv-4090/bin/python - <<'PY'
import gzip
import json
import os
from pathlib import Path
from ainference_optim.data import parse_prompts

directory = Path(os.environ["DATA_DIR"])
source = directory / "HumanEval.jsonl.gz"
output = directory / "humaneval-prompts.jsonl"
with gzip.open(source, "rt", encoding="utf-8") as handle:
    original = [json.loads(line) for line in handle if line.strip()]
rows = [{"id": item["task_id"], "scenario": "code", "prompt": item["prompt"]}
        for item in original]
content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
parse_prompts(content, str(source))
with output.open("x", encoding="utf-8") as handle:
    handle.write(content)
print(f"已生成 {len(rows)} 条输入：{output}")
PY
```

压缩包、SHA256 和转换结果都在 DATA_DIR，不入 Git；转换文件存在时拒绝覆盖。
下载地址跟随上游 master，正式实验需保留原始文件、URL 和 SHA256，固定该份输入。
编辑 configs/local/4090_baseline.toml 中现有的 dataset 行，改为转换文件的绝对路径：

```toml
dataset = "/你的绝对数据目录/humaneval-prompts.jsonl"
```

TOML 不展开 $DATA_DIR、$MODEL_DIR 或 ~。提取 prompt 只用于性能负载，
**不等于标准 HumanEval pass@k 评测**；当前不执行答案测试，聊天模板也会改变任务形式。
JSON/工具调用的正式数据集仍待独立选择，自带 JSON 提示只验证管线。

### 自己的数据格式

UTF-8 JSONL，每行仅有 id、scenario、prompt 三个非空字符串字段，id 全文件唯一：

```json
{"id":"chat-001","scenario":"chat","prompt":"请解释 KV Cache。"}
{"id":"code-001","scenario":"code","prompt":"用 Python 实现整数相加。"}
```

多行代码需在字符串中编码为 \n，不要将一个 JSON 对象拆成多行。
额外的答案/测试字段要在转换时移除；scenario 只是分组标签，不启用评分或 JSON 约束。
实际输入 token 数包含聊天模板，超出 max_input_tokens 会报错而非自动截断。

## 运行真实模型

在同一项目根目录逐条运行，确认成功再继续；更换模型、输入或生成设置后重新 verify：

```bash
.venv-4090/bin/python -m ainference_optim verify --config configs/local/4090_smoke.toml --samples 3
.venv-4090/bin/python -m ainference_optim run --config configs/local/4090_smoke.toml
.venv-4090/bin/python -m ainference_optim verify --config configs/local/4090_baseline.toml --samples 3
.venv-4090/bin/python -m ainference_optim run --config configs/local/4090_baseline.toml
.venv-4090/bin/python -m ainference_optim summarize outputs/实际生成的目录名
```

smoke 最多生成 32 token、遇 EOS 停止；baseline 预热 5 次、正式重复 3 次，
忽略 EOS 并固定输出 128 token，仅用于性能测量。
verify 只核对所选贪心请求的 token 序列，不证明任务准确率或采样分布等价。

### 重复运行不稳定时的确定性诊断

若同一 seed 下手写或参考路径各自重复也不一致，先在全新 Python 进程运行：

```bash
.venv-4090/bin/python -m ainference_optim verify \
  --config configs/4090_deterministic.toml --samples 3 --repeats 3
```

该预设直接读取项目 models/Qwen2.5-1.5B-Instruct，无需编辑已有服务器配置；
若模型另存别处，复制后修改 model，并修正相对数据/输出路径。
它保持 BF16，显式设置 deterministic=true 和 sdpa_backend="math"，
在 CUDA 初始化前设置 cuBLAS workspace，并以严格模式拒绝已知非确定性算子。
每条提示使用同一个 seed 重复，必须同时满足两条路径互相一致、各自重复一致。
此处 --repeats 属于 verify，与实验配置中性能重复次数独立，默认值为 1。

默认配置仍为 deterministic=false、sdpa_backend="auto"，不静默改变已有实验。
math 内核和严格确定性可能改变速度、内存及输出；诊断通过不证明默认内核已稳定，
两种配置不得混为同一个性能对照。不同软件/设备上的位级一致也不受此开关保证。
依据：[PyTorch 可复现性](https://docs.pytorch.org/docs/2.9/notes/randomness.html)、
[SDPA 内核选择](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)。

## 确定性验证仍失败时

若 `verify --repeats 3` 显示同一路径重复运行也不一致，保留失败报告，
用固定共同前缀检查首个跨路径分叉处的原始 logits：

```bash
.venv-4090/bin/python -m ainference_optim diagnose \
  --config configs/4090_deterministic.toml \
  --report outputs/<失败验证目录>/verification.json --repeats 3
```

命令从报告旁的 `dataset.jsonl` 读取原输入并核对哈希，不修改原报告。
只选择首条跨路径不一致记录，分别重复完整上下文无缓存计算和逐 token KV cache 重放。
重放始终喂入同一组已保存 token，避免自由生成分叉改变后续输入。
终端展示 `finite`、`greedy_token`、`margin`、`equal_to_first`、`max_abs_difference`，
完整 top-5 与哈希保存在新的 `outputs/*-diagnose-*/diagnostics.json`。

若需精度对照，在同一命令末尾加 `--dtype float32`，会新建另一份报告。
FP32 仅用于定位问题，其结果不能代替 BF16 验收或 BF16 性能基线。
两种计算路径之间不要求 logits 逐位一致；先查看各路径内部的重复稳定性。
判读边界见 [实验协议](docs/experiment_protocol.md#固定前缀-logits-诊断)。

## 本地无 GPU 快速开始

需要 Python 3.11 或以上。mock 核心运行不依赖 PyTorch 或任何第三方包。
固定推理依赖建议使用 Python 3.11–3.13，安装方式详见 4090 运行说明。
在项目根目录的 PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ainference_optim run --config configs/mock.toml
.\.venv\Scripts\python.exe -m ainference_optim summarize outputs/<实际生成的目录名>
```

也可以不安装包，直接执行源码：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m ainference_optim run --config configs/mock.toml
python -m unittest discover -s tests -v
```

mock 返回确定性的合成时间线，不测量本机性能；其 input_tokens 是字符数占位。
示例数据仅用于管线检查，不是正式基准，也没有输出质量评分。

## 目录与职责

```text
configs/                         TOML 实验配置
data/                            可公开的 JSONL 样本
docs/engineering.md              代码、类型、注释及扩展规范
docs/experiment_protocol.md      测量定义与复现约定
docs/evaluation_metrics.md       已确定的指标、统计口径和验收目标
src/ainference_optim/
    domain.py                    数据模型与后端协议
    config.py                    配置解析和校验
    data.py                      数据集读取和校验
    backends/                    设备、模型与具体生成实现
    metrics.py                   纯函数指标与聚合
    results.py                   文件序列化
    runner.py                    实验编排与资源生命周期
    verification.py              与参考生成逐 token 核对，单独保存报告
    diagnostics.py               从失败报告组织固定前缀 logits 诊断
    cli.py                       命令行装配入口
tests/                           无 GPU 测试与可选真实库 CPU 集成测试
outputs/                         自动生成，不纳入版本管理
```

每次实验生成独立目录，包含 `manifest.json`（配置、环境、数据哈希、状态）、
`dataset.jsonl`（输入快照）、`requests.jsonl`（逐请求输出、时间线和指标）、
`summary.json`（整体与分场景汇总）。失败的实验保留已有请求与错误信息，
不生成成功汇总。输出包含输入和模型生成文本，使用私有数据时应自行控制访问。

## 开发检查

Windows 下安装开发依赖后可运行 `./scripts/check.ps1` 一次完成以下检查。

```powershell
python -m unittest discover -s tests -v
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

开发前阅读 [工程规范](docs/engineering.md) 和 [实验协议](docs/experiment_protocol.md)。
普通自回归后端通过目标设备验证后，再添加固定长度投机解码作为对照。
框架导入和 mock 运行不会加载 torch/transformers；推理依赖位于独立可选组。
