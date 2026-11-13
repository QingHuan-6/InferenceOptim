"""编排串行实验，不承载模型计算或设备逻辑。"""

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from ainference_optim import __version__
from ainference_optim.config import ExperimentConfig
from ainference_optim.data import parse_prompts
from ainference_optim.domain import Backend, GenerationRequest, RunProgress
from ainference_optim.metrics import RequestMetrics, measure, summarize
from ainference_optim.results import SCHEMA_VERSION, append_record, write_json


def _code_identity() -> dict[str, object]:
    try:
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip()), "status": status}
    except (OSError, subprocess.SubprocessError) as error:
        return {"commit": None, "dirty": None, "unavailable_reason": type(error).__name__}


def run_experiment(
    config: ExperimentConfig,
    backend: Backend,
    progress: Callable[[RunProgress], None] | None = None,
) -> Path:
    """运行预热与正式测量，将成功或失败状态保存到独立目录。

    Args:
        config: 已验证的实验配置。
        backend: 已初始化的后端；本函数取得其生命周期管理权。
        progress: 可选请求间回调；发生在后端计时之外，默认不输出进度。

    Returns:
        本次实验输出目录的绝对路径。

    Raises:
        Exception: 原样传播加载、生成与写入错误；已开始的实验记录失败状态。
    """
    try:
        return _run(config, backend, progress)
    finally:
        backend.close()


def _run(
    config: ExperimentConfig, backend: Backend, progress: Callable[[RunProgress], None] | None
) -> Path:
    # 同一份字节快照同时用于哈希与执行，避免数据文件修改导致证据不一致。
    dataset_bytes = config.dataset.read_bytes()
    prompts = parse_prompts(dataset_bytes.decode("utf-8-sig"), str(config.dataset))
    provenance: dict[str, str] = {}
    for name in ("preparation.json", "download_sources.json"):
        source = config.dataset.parent / name
        if source.exists():
            provenance[name] = source.read_text(encoding="utf-8")
    if "preparation.json" in provenance:
        preparation = json.loads(provenance["preparation.json"])
        if (
            preparation["selection"][config.dataset.stem]["sha256"]
            != hashlib.sha256(dataset_bytes).hexdigest()
        ):
            raise ValueError("输入快照与 preparation.json 的固定抽样哈希不符")
    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%dT%H%M%S}-{uuid4().hex[:12]}"
    output = config.output_dir / f"{config.name}-{run_id}"
    output.mkdir(parents=True, exist_ok=False)
    snapshot = asdict(config)
    snapshot["dataset"] = str(config.dataset)
    snapshot["output_dir"] = str(config.output_dir)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": started.isoformat(),
        "package_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "backend": asdict(backend.info),
        "config": snapshot,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "measurement_mode": "serial_single_request",
        "code": _code_identity(),
    }
    write_json(output / "manifest.json", manifest)
    (output / "dataset.jsonl").write_bytes(dataset_bytes)
    for name, content in provenance.items():
        (output / name).write_text(str(content), encoding="utf-8")
    collected: list[RequestMetrics] = []
    grouped: dict[str, list[RequestMetrics]] = {}
    call_index = 0
    try:
        phase_started = perf_counter()
        if progress is not None and config.warmup:
            progress(RunProgress("warmup", 0, config.warmup, None, None, config.repeats, 0.0))
        for index in range(config.warmup):
            request = GenerationRequest(
                prompts[index % len(prompts)], config.max_new_tokens, config.seed + call_index
            )
            warmup_result = backend.generate(request)
            if measure(warmup_result).output_tokens > config.max_new_tokens:
                raise ValueError("预热返回的 token 数量超过 max_new_tokens")
            call_index += 1
            if progress is not None:
                progress(
                    RunProgress(
                        "warmup",
                        index + 1,
                        config.warmup,
                        request.prompt.id,
                        None,
                        config.repeats,
                        perf_counter() - phase_started,
                    )
                )
        phase_started = perf_counter()
        total = len(prompts) * config.repeats
        if progress is not None:
            progress(RunProgress("measurement", 0, total, None, None, config.repeats, 0.0))
        for repeat in range(config.repeats):
            for prompt in prompts:
                request = GenerationRequest(prompt, config.max_new_tokens, config.seed + call_index)
                result = backend.generate(request)
                metrics = measure(result)
                if metrics.output_tokens > config.max_new_tokens:
                    raise ValueError("后端返回的 token 数量超过 max_new_tokens")
                append_record(
                    output / "requests.jsonl",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "prompt_id": prompt.id,
                        "scenario": prompt.scenario,
                        "repeat": repeat,
                        "seed": request.seed,
                        "synthetic": backend.info.synthetic,
                        "result": asdict(result),
                        "metrics": asdict(metrics),
                    },
                )
                collected.append(metrics)
                grouped.setdefault(prompt.scenario, []).append(metrics)
                call_index += 1
                # 后端已完成计时与设备同步，且记录已落盘，再通知展示层。
                if progress is not None:
                    progress(
                        RunProgress(
                            "measurement",
                            len(collected),
                            total,
                            prompt.id,
                            repeat + 1,
                            config.repeats,
                            perf_counter() - phase_started,
                        )
                    )
        write_json(
            output / "summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "synthetic": backend.info.synthetic,
                "aggregation": "request-level; linear-interpolated percentiles",
                "overall": summarize(collected),
                "by_scenario": {name: summarize(rows) for name, rows in grouped.items()},
            },
        )
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
        manifest["completed_requests"] = len(collected)
        write_json(output / "manifest.json", manifest)
    return output
