"""运行有限规模的 HF 提示查找投机可行性实验，保留配对原始数据。"""

import hashlib
import math
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import uuid4

from ainference_optim.config import ExperimentConfig
from ainference_optim.data import parse_prompts
from ainference_optim.domain import (
    Backend,
    GenerationRequest,
    LookupTimingBackend,
    NativeTiming,
    RunProgress,
)
from ainference_optim.results import append_record, code_identity, write_json


def validate_pilot(config: ExperimentConfig, samples: int, repeats: int, draft_tokens: int) -> None:
    """在加载模型前拒绝不适合有限规模试跑的参数。"""
    if (
        not 1 <= samples <= 3
        or not 1 <= repeats <= 3
        or not 1 <= draft_tokens <= 16
        or not 1 <= config.max_new_tokens <= 128
        or config.transformers is None
        or config.transformers.stop_on_eos
        or config.transformers.generation_path != "hf_generate"
    ):
        raise ValueError(
            "投机小试验要求固定长度 hf_generate，samples/repeats=1–3，草稿=1–16，输出≤128"
        )


def _validate(value: NativeTiming, expected: int) -> None:
    if (
        not math.isfinite(value.generation_elapsed_s)
        or value.generation_elapsed_s <= 0
        or not math.isfinite(value.request_elapsed_s)
        or value.request_elapsed_s < value.generation_elapsed_s
        or value.input_tokens < 1
        or len(value.token_ids) != expected
        or any(type(token) is not int or token < 0 for token in value.token_ids)
        or (value.peak_memory_bytes is not None and value.peak_memory_bytes < 0)
    ):
        raise ValueError("投机小试验返回非法耗时、显存或固定长度输出")


def run_speculation_pilot(
    config: ExperimentConfig,
    backend: Backend,
    samples: int = 3,
    repeats: int = 2,
    draft_tokens: int = 4,
    progress: Callable[[RunProgress], None] | None = None,
) -> Path:
    """配对比较普通生成与 HF 提示查找；仅判断可行性，不自动启动正式实验。

    Args:
        config: 固定输出长度的原生 HF 配置，预热固定为每种模式一次。
        backend: 支持提示查找的后端，退出时总会关闭。
        samples: 数据集前1–3条输入，不按结果筛选。
        repeats: 每条输入的配对次数，1–3次。
        draft_tokens: 提示查找草稿上限，1–16。
        progress: 配对完成后的可选进度回调，不在生成计时中执行。

    Returns:
        唯一结果目录，包含输入快照、报告及每对原始记录。

    Raises:
        ValueError: 参数或后端不符合约束，或返回无效观测。
        Exception: 运行失败；已创建的报告保留失败状态，不继续运行。
    """
    try:
        validate_pilot(config, samples, repeats, draft_tokens)
        if not isinstance(backend, LookupTimingBackend):
            raise ValueError("后端不支持投机配对计时")
        raw = config.dataset.read_bytes()
        prompts = parse_prompts(raw.decode("utf-8-sig"), str(config.dataset))[:samples]
        output = config.output_dir / f"{config.name}-lookup-pilot-{uuid4().hex[:12]}"
        output.mkdir(parents=True, exist_ok=False)
        report: dict[str, object] = {
            "schema_version": 1,
            "status": "running",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "code": code_identity(),
            "backend": asdict(backend.info),
            "config": {
                **asdict(config),
                "dataset": str(config.dataset),
                "output_dir": str(config.output_dir),
            },
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "selected_prompt_ids": [prompt.id for prompt in prompts],
            "draft_tokens": draft_tokens,
            "max_matching_ngram_size": 2,
            "repeats": repeats,
            "warmup_pairs": 1,
            "completed_pairs": 0,
            "timing": (
                "synchronized generate only; no streamer; includes prefill and verification; "
                "excludes tokenization and final text decode"
            ),
            "scope": (
                "feasibility only; no TTFT/TPOT, distribution equivalence or task quality claim"
            ),
            "acceptance_rate": None,
            "execution_paths": {
                "plain": "HF greedy generate",
                "lookup": "HF assisted prompt lookup; bounded candidates",
            },
        }
        (output / "dataset.jsonl").write_bytes(raw)
        write_json(output / "pilot.json", report)
        durations: dict[str, list[float]] = {"plain": [], "lookup": []}
        memory: dict[str, list[int]] = {"plain": [], "lookup": []}
        matched = 0
        try:
            start = perf_counter()
            request = GenerationRequest(prompts[0], config.max_new_tokens, config.seed)
            for size in (0, draft_tokens):
                _validate(backend.lookup_timing(request, size), config.max_new_tokens)
            if progress:
                progress(
                    RunProgress(
                        "warmup", 1, 1, prompts[0].id, None, repeats, perf_counter() - start
                    )
                )
            start = perf_counter()
            for repeat in range(repeats):
                for index, prompt in enumerate(prompts):
                    request = GenerationRequest(prompt, config.max_new_tokens, config.seed + index)
                    order = (
                        ("plain", "lookup") if (index + repeat) % 2 == 0 else ("lookup", "plain")
                    )
                    values: dict[str, NativeTiming] = {}
                    for mode in order:
                        value = backend.lookup_timing(
                            request, 0 if mode == "plain" else draft_tokens
                        )
                        _validate(value, config.max_new_tokens)
                        values[mode] = value
                    if values["plain"].input_tokens != values["lookup"].input_tokens:
                        raise ValueError("配对输入 token 数不同")
                    mismatch = next(
                        (
                            i
                            for i, (a, b) in enumerate(
                                zip_longest(values["plain"].token_ids, values["lookup"].token_ids)
                            )
                            if a != b
                        ),
                        None,
                    )
                    matched += int(mismatch is None)
                    append_record(
                        output / "pairs.jsonl",
                        {
                            "prompt_id": prompt.id,
                            "scenario": prompt.scenario,
                            "repeat": repeat,
                            "seed": request.seed,
                            "order": order,
                            "token_match": mismatch is None,
                            "first_mismatch_index": mismatch,
                            **{mode: asdict(value) for mode, value in values.items()},
                        },
                    )
                    for mode, value in values.items():
                        durations[mode].append(value.generation_elapsed_s)
                        if value.peak_memory_bytes is not None:
                            memory[mode].append(value.peak_memory_bytes)
                    report["completed_pairs"] = len(durations["plain"])
                    if progress:
                        progress(
                            RunProgress(
                                "measurement",
                                len(durations["plain"]),
                                len(prompts) * repeats,
                                prompt.id,
                                repeat + 1,
                                repeats,
                                perf_counter() - start,
                            )
                        )
            report["summary"] = {
                "plain_mean_generation_s": mean(durations["plain"]),
                "lookup_mean_generation_s": mean(durations["lookup"]),
                "generation_speedup": mean(durations["plain"]) / mean(durations["lookup"]),
                "token_match_pairs": matched,
                "pair_count": len(durations["plain"]),
                "peak_memory_bytes_max": {
                    mode: max(values) if values else None for mode, values in memory.items()
                },
                "aggregation": (
                    "equal repeats per selected prompt; "
                    "ratio of arithmetic mean generation times; warmup excluded"
                ),
            }
            report["status"] = "completed"
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            report["finished_at_utc"] = datetime.now(UTC).isoformat()
            write_json(output / "pilot.json", report)
        return output
    finally:
        backend.close()
