"""命令行入口，只负责参数解析、装配和结果展示。"""

import argparse
import json
import sys
from pathlib import Path

from ainference_optim.backends import create_backend
from ainference_optim.config import load_config
from ainference_optim.data import load_prompts
from ainference_optim.runner import run_experiment
from ainference_optim.verification import verify_backend


def main(argv: list[str] | None = None) -> int:
    """执行命令；配置和文件错误以非零退出码返回。"""
    parser = argparse.ArgumentParser(description="解码优化实验基础设施")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="运行串行实验")
    run.add_argument("--config", type=Path, required=True)
    verify = commands.add_parser("verify", help="核对普通解码与参考 generate 的输出 token")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--samples", type=int, default=3)
    report = commands.add_parser("summarize", help="查看已完成实验的汇总")
    report.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in {"run", "verify"}:
            if args.command == "verify" and args.samples < 1:
                raise ValueError("--samples 必须大于零")
            config = load_config(args.config)
            # 在加载权重或占用 GPU 之前发现数据集格式错误。
            load_prompts(config.dataset)
            backend = create_backend(config.backend, config.transformers)
            if args.command == "verify":
                output, passed = verify_backend(config, backend, args.samples)
                print(f"验证报告：{output}；token 一致性：{'通过' if passed else '失败'}")
                return 0 if passed else 2
            else:
                print(f"结果目录：{run_experiment(config, backend)}")
        else:
            manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
            if manifest["status"] != "completed":
                raise ValueError("实验尚未成功完成，请检查 manifest.json")
            summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
            if summary["synthetic"]:
                print("注意：以下全部是合成数据，仅用于验证流程。")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0
