"""编排串行实验，不承载模型计算或设备逻辑。"""

import hashlib
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ainference_optim import __version__
from ainference_optim.config import ExperimentConfig
from ainference_optim.data import parse_prompts
from ainference_optim.domain import Backend, GenerationRequest
from ainference_optim.metrics import RequestMetrics, measure, summarize
from ainference_optim.results import SCHEMA_VERSION, append_record, write_json


def run_experiment(config: ExperimentConfig, backend: Backend) -> Path:
    """运行预热与正式测量，将成功或失败状态保存到独立目录。

    Args:
        config: 已验证的实验配置。
        backend: 已初始化的后端；本函数取得其生命周期管理权。

    Returns:
        本次实验输出目录的绝对路径。

    Raises:
        Exception: 原样传播加载、生成与写入错误；已开始的实验记录失败状态。
    """
    try:
        return _run(config, backend)
    finally:
        backend.close()


def _run(config: ExperimentConfig, backend: Backend) -> Path:
    # 同一份字节快照同时用于哈希与执行，避免数据文件修改导致证据不一致。
    dataset_bytes = config.dataset.read_bytes()
    prompts = parse_prompts(dataset_bytes.decode("utf-8-sig"), str(config.dataset))
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
    }
    write_json(output / "manifest.json", manifest)
    (output / "dataset.jsonl").write_bytes(dataset_bytes)
    collected: list[RequestMetrics] = []
    grouped: dict[str, list[RequestMetrics]] = {}
    call_index = 0
    try:
        for index in range(config.warmup):
            request = GenerationRequest(
                prompts[index % len(prompts)], config.max_new_tokens, config.seed + call_index
            )
            warmup_result = backend.generate(request)
            if measure(warmup_result).output_tokens > config.max_new_tokens:
                raise ValueError("预热返回的 token 数量超过 max_new_tokens")
            call_index += 1
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
