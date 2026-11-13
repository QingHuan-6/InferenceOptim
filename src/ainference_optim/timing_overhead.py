"""交错测量相同原生生成实现的逐 token 观测开销，不替代性能或正确性验收。"""

import hashlib
import math
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import uuid4

from ainference_optim.config import ExperimentConfig
from ainference_optim.data import parse_prompts
from ainference_optim.domain import (
    Backend,
    GenerationRequest,
    NativeTiming,
    NativeTimingBackend,
    RunProgress,
)
from ainference_optim.results import SCHEMA_VERSION, append_record, write_json


def _validate(value: NativeTiming, expected_tokens: int) -> None:
    if (
        not math.isfinite(value.generation_elapsed_s)
        or value.generation_elapsed_s <= 0
        or not math.isfinite(value.request_elapsed_s)
        or value.request_elapsed_s < value.generation_elapsed_s
    ):
        raise ValueError("计时探针返回非法整体耗时")
    if value.input_tokens < 1 or len(value.token_ids) != expected_tokens:
        raise ValueError("计时探针未返回预期固定长度 token")
    if any(type(token) is not int or token < 0 for token in value.token_ids):
        raise ValueError("计时探针包含非法 token ID")


def measure_timing_overhead(
    config: ExperimentConfig,
    backend: Backend,
    samples: int = 10,
    repeats: int = 3,
    warmup: int = 2,
    progress: Callable[[RunProgress], None] | None = None,
) -> Path:
    """同模型交错执行有/无回调的原生生成，并保存原始配对记录。

    Args:
        config: 固定长度原生 HF 配置，保持模型精度及 attention 设置。
        backend: 支持整体计时的后端，函数退出时总会关闭。
        samples: 输入快照中选用的前缀样本数，至少1条。
        repeats: 每个样本的配对重复次数，至少2次。
        warmup: 正式计时前的预热对数，至少1对，独立于性能配置。
        progress: 请求对之间的可选进度回调。

    Returns:
        独立报告目录；无回调模式不产生 TTFT、TPOT 或逐 token 时间线。

    Raises:
        ValueError: 后端不支持、参数不合法或没有固定输出长度。
        Exception: 诊断运行失败，已产生的报告和配对记录仍保留。
    """
    try:
        if (
            samples < 1
            or repeats < 2
            or warmup < 1
            or not isinstance(backend, NativeTimingBackend)
            or config.transformers is None
            or config.transformers.stop_on_eos
            or config.transformers.generation_path != "hf_generate"
        ):
            raise ValueError("要求固定长度 hf_generate、samples≥1、repeats≥2、warmup≥1")
        raw = config.dataset.read_bytes()
        prompts = parse_prompts(raw.decode("utf-8-sig"), str(config.dataset))[:samples]
        output = config.output_dir / f"{config.name}-overhead-{uuid4().hex[:12]}"
        output.mkdir(parents=True, exist_ok=False)
        snapshot = asdict(config)
        snapshot.update(dataset=str(config.dataset), output_dir=str(config.output_dir))
        report: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "config": snapshot,
            "backend": asdict(backend.info),
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "selected_prompt_ids": [prompt.id for prompt in prompts],
            "samples": len(prompts),
            "repeats": repeats,
            "warmup_pairs": warmup,
            "execution": "paired native generate; order alternates by prompt index plus repeat",
            "scope": (
                "streamer timing diagnostic; not algorithm acceleration or losslessness acceptance"
            ),
            "timing": (
                "generation: synchronized generate only; "
                "request: tokenize, transfer, generate, final decode"
            ),
        }
        (output / "dataset.jsonl").write_bytes(raw)
        write_json(output / "overhead.json", report)
        totals: dict[str, dict[str, list[float]]] = {
            mode: {"generation_elapsed_s": [], "request_elapsed_s": []}
            for mode in ("streamer", "plain")
        }
        matched = 0
        completed = 0
        try:
            phase_started = perf_counter()
            if progress is not None:
                progress(RunProgress("warmup", 0, warmup, None, None, repeats, 0.0))
            for index in range(warmup):
                request = GenerationRequest(prompts[0], config.max_new_tokens, config.seed)
                for observed in (True, False) if index % 2 == 0 else (False, True):
                    _validate(backend.native_timing(request, observed), config.max_new_tokens)
                if progress is not None:
                    progress(
                        RunProgress(
                            "warmup",
                            index + 1,
                            warmup,
                            prompts[0].id,
                            None,
                            repeats,
                            perf_counter() - phase_started,
                        )
                    )
            phase_started = perf_counter()
            if progress is not None:
                progress(
                    RunProgress("measurement", 0, len(prompts) * repeats, None, None, repeats, 0.0)
                )
            for repeat in range(repeats):
                for index, prompt in enumerate(prompts):
                    request = GenerationRequest(prompt, config.max_new_tokens, config.seed + index)
                    order = (True, False) if (repeat + index) % 2 == 0 else (False, True)
                    values: dict[str, NativeTiming] = {}
                    for observed in order:
                        mode = "streamer" if observed else "plain"
                        value = backend.native_timing(request, observed)
                        _validate(value, config.max_new_tokens)
                        values[mode] = value
                        for field in totals[mode]:
                            totals[mode][field].append(getattr(value, field))
                    if values["streamer"].input_tokens != values["plain"].input_tokens:
                        raise ValueError("配对输入 token 数不同")
                    match = values["streamer"].token_ids == values["plain"].token_ids
                    matched += int(match)
                    append_record(
                        output / "pairs.jsonl",
                        {
                            "prompt_id": prompt.id,
                            "repeat": repeat,
                            "seed": request.seed,
                            "order": ["streamer" if flag else "plain" for flag in order],
                            "token_match": match,
                            **{key: asdict(value) for key, value in values.items()},
                        },
                    )
                    completed += 1
                    if progress is not None:
                        progress(
                            RunProgress(
                                "measurement",
                                completed,
                                len(prompts) * repeats,
                                prompt.id,
                                repeat + 1,
                                repeats,
                                perf_counter() - phase_started,
                            )
                        )
            summary: dict[str, object] = {}
            for field in totals["plain"]:
                observed_mean = mean(totals["streamer"][field])
                plain_mean = mean(totals["plain"][field])
                summary[field] = {
                    "streamer_mean_s": observed_mean,
                    "plain_mean_s": plain_mean,
                    "streamer_to_plain_ratio": observed_mean / plain_mean,
                    "streamer_overhead_percent": (observed_mean / plain_mean - 1) * 100,
                }
            summary["token_match_pairs"] = matched
            summary["pair_count"] = completed
            summary["aggregation"] = "equal repeats per prompt; mean of time, then ratio of means"
            report["summary"] = summary
            report["status"] = "completed"
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            report["completed_pairs"] = completed
            report["finished_at_utc"] = datetime.now(UTC).isoformat()
            write_json(output / "overhead.json", report)
        return output
    finally:
        backend.close()
