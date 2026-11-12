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

## 快速开始

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
