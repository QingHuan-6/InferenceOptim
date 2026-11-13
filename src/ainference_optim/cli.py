"""命令行入口，只负责参数解析、装配和结果展示。"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from ainference_optim.backends import create_backend
from ainference_optim.comparison import compare_runs
from ainference_optim.config import load_config
from ainference_optim.data import load_prompts
from ainference_optim.diagnostics import diagnose_backend
from ainference_optim.preparation import prepare_baseline_data
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
    verify.add_argument("--repeats", type=int, default=1, help="同输入同种子的稳定性重复次数")
    diagnose = commands.add_parser("diagnose", help="从失败报告的共同前缀重复探查原始 logits")
    diagnose.add_argument("--config", type=Path, required=True)
    diagnose.add_argument("--report", type=Path, required=True)
    diagnose.add_argument("--repeats", type=int, default=3)
    diagnose.add_argument("--dtype", choices=("bfloat16", "float32"))
    report = commands.add_parser("summarize", help="查看已完成实验的汇总")
    report.add_argument("run_dir", type=Path)
    prepare = commands.add_parser(
        "prepare-baseline", help="从原始 Alpaca/HumanEval 固定抽样性能输入"
    )
    prepare.add_argument("--alpaca", type=Path, required=True)
    prepare.add_argument("--humaneval", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=Path("data/baseline-v1"))
    prepare.add_argument("--seed", type=int, default=42)
    compare = commands.add_parser("compare", help="校验并配对比较两次固定长度实验")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-baseline":
            prepared = prepare_baseline_data(
                args.alpaca, args.humaneval, args.output_dir, args.seed
            )
            print(f"输入目录：{prepared}")
            return 0
        if args.command == "compare":
            output = compare_runs(args.baseline, args.candidate, args.output_dir)
            print(f"对比报告：{output}；仅为计时对比，不代表通过正确性验收")
            summary = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            if summary["synthetic"]:
                print("注意：这是合成计时对比，不代表真实性能。")
            print(json.dumps(summary["by_scenario"], ensure_ascii=False, indent=2))
            return 0
        if args.command in {"run", "verify", "diagnose"}:
            if args.command == "verify" and (args.samples < 1 or args.repeats < 1):
                raise ValueError("--samples 和 --repeats 必须大于零")
            config = load_config(args.config)
            if args.command == "diagnose":
                if args.repeats < 2 or config.transformers is None:
                    raise ValueError("diagnose 要求真实模型配置和 --repeats≥2")
                if args.dtype is not None:
                    if config.transformers.device == "cpu" and args.dtype != "float32":
                        raise ValueError("CPU 诊断只支持 float32")
                    config = replace(
                        config, transformers=replace(config.transformers, dtype=args.dtype)
                    )
            # 在加载权重或占用 GPU 之前发现数据集格式错误。
            load_prompts(config.dataset)
            backend = create_backend(config.backend, config.transformers)
            if args.command == "diagnose":
                output = diagnose_backend(config, backend, args.report, args.repeats)
                print(f"诊断报告：{output}；探针完成不等于验证通过")
                diagnostic = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
                for item in diagnostic["observations"]:
                    print(
                        f"{item['path']} repeat={item['repeat']} finite={item['finite']} "
                        f"greedy_token={item['greedy_token_id']} margin={item['top_margin']} "
                        f"equal_to_first={item['equal_to_first']} "
                        f"max_abs_difference={item['max_abs_difference']}"
                    )
            elif args.command == "verify":
                output, passed = verify_backend(config, backend, args.samples, args.repeats)
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
