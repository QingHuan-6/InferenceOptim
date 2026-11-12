#!/usr/bin/env bash
# 安装到独立环境，不覆盖平台镜像自带的 PyTorch。
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 13), "Use Python 3.11-3.13"'
command -v nvidia-smi >/dev/null
nvidia-smi
"$PYTHON_BIN" -m venv .venv-4090
.venv-4090/bin/python -m pip install 'torch==2.9.0' --index-url https://download.pytorch.org/whl/cu128
.venv-4090/bin/python -m pip install -e '.[inference,dev]'
.venv-4090/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check driver/wheel"; assert torch.cuda.is_bf16_supported(), "BF16 unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
.venv-4090/bin/python -m pip check
printf '%s\n' '环境检查通过；按 docs/remote_4090.md 设置 HF_HOME 后执行 verify 和 run。'
