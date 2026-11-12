# AInferenceOptim

毕设《推理引擎中的解码加速优化研究》的实验基础设施。

当前完成硬件无关的串行实验管线：严格配置校验、JSONL 数据加载、后端协议、
请求级指标、原始结果落盘、实验汇总与失败记录。
现有 mock 和 Transformers 普通贪心解码后端；后者支持单设备 CUDA 或显式 CPU 功能测试。
当前不包含昇腾、量化、投机解码或并发调度。4090 性能须在租用设备上实际验证。

租到机器前可先打包：`python scripts/package_remote.py`。
服务器安装、真实模型输出核对及实验步骤见 [4090 运行说明](docs/remote_4090.md)。
日常通过 GitHub 私有仓库同步；分支、AI 提交边界与服务器更新流程见
[GitHub 与 AI Git 规范](docs/git_workflow.md)。

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
src/ainference_optim/
    domain.py                    数据模型与后端协议
    config.py                    配置解析和校验
    data.py                      数据集读取和校验
    backends/                    设备、模型与具体生成实现
    metrics.py                   纯函数指标与聚合
    results.py                   文件序列化
    runner.py                    实验编排与资源生命周期
    verification.py              与参考生成逐 token 核对，单独保存报告
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
