"""编排 HF 提示查找的配对实验，保留原始数据并生成正式对照表。"""

import hashlib
import math
import random
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
    SpeculationTimingBackend,
)
from ainference_optim.metrics import percentile
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


_METHODS = ("plain", "fixed", "adaptive")


def validate_speculation(config: ExperimentConfig, draft_tokens: int) -> None:
    """在模型加载前校验正式实验采用固定长度、预热和原生生成。"""
    validate_pilot(config, 1, 1, draft_tokens)
    if config.warmup < 1:
        raise ValueError("正式实验要求各方法至少预热一次")


def _measure_methods(
    backend: SpeculationTimingBackend,
    request: GenerationRequest,
    draft_tokens: int,
    offset: int,
) -> dict[str, NativeTiming]:
    # 轮换相对顺序，避免某一方法总在模型刚预热或 GPU 更热时执行。
    order = _METHODS[offset:] + _METHODS[:offset]
    results: dict[str, NativeTiming] = {}
    for mode in order:
        value = backend.speculation_timing(
            request, 0 if mode == "plain" else draft_tokens, adaptive=mode == "adaptive"
        )
        _validate(value, request.max_new_tokens)
        counts = value.draft_counts
        if mode != "plain" and (
            counts is None
            or not 0 <= counts.accepted <= counts.proposed
            or counts.skipped_rounds < 0
        ):
            raise ValueError(f"{mode} 草稿计数缺失或非法")
        results[mode] = value
    if len({value.input_tokens for value in results.values()}) != 1:
        raise ValueError("同一提示的输入 token 数不同")
    return results


def _main_summary(
    groups: dict[str, list[dict[str, NativeTiming]]],
) -> dict[str, dict[str, object]]:
    means = [
        [mean(row[mode].generation_elapsed_s for row in rows) for mode in _METHODS]
        for rows in groups.values()
    ]
    rng = random.Random(42)
    ratios: dict[str, list[float]] = {mode: [] for mode in _METHODS}
    for _ in range(1000):
        resample = rng.choices(means, k=len(means))
        baseline = sum(row[0] for row in resample)
        for index, mode in enumerate(_METHODS):
            ratios[mode].append(baseline / sum(row[index] for row in resample))
    summary: dict[str, dict[str, object]] = {}
    all_rows = [row for rows in groups.values() for row in rows]
    for index, mode in enumerate(_METHODS):
        values = [row[mode] for row in all_rows]
        drafts = [value.draft_counts for value in values if value.draft_counts is not None]
        memory = [
            value.peak_memory_bytes for value in values if value.peak_memory_bytes is not None
        ]
        proposed = sum(draft.proposed for draft in drafts)
        accepted = sum(draft.accepted for draft in drafts)
        summary[mode] = {
            "prompt_count": len(groups),
            "request_count": len(values),
            "mean_generation_ms": 1000 * mean(row[index] for row in means),
            "p50_generation_ms": percentile([1000 * v.generation_elapsed_s for v in values], 0.5),
            "speedup_vs_plain": sum(row[0] for row in means) / sum(row[index] for row in means),
            "speedup_95ci": [percentile(ratios[mode], q) for q in (0.025, 0.975)],
            "peak_memory_bytes": max(memory, default=None),
            "draft_proposed": proposed if drafts else None,
            "draft_accepted": accepted if drafts else None,
            "draft_acceptance_rate": accepted / proposed if proposed else None,
            "skipped_lookup_rounds": sum(draft.skipped_rounds for draft in drafts)
            if drafts
            else None,
            "exact_token_matches_vs_plain": sum(
                row[mode].token_ids == row["plain"].token_ids for row in all_rows
            ),
        }
    return summary


def _main_table(summary: dict[str, dict[str, object]]) -> str:
    lines = [
        "| 方法 | 平均完整生成 (ms) | 加速比 [95% CI] | 峰值显存 (GiB) | 草稿接受率 | token 一致 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, label in zip(_METHODS, ("普通贪心", "固定草稿", "自适应草稿"), strict=True):
        row = summary[mode]
        speed, latency = row["speedup_vs_plain"], row["mean_generation_ms"]
        ci, mem, rate = row["speedup_95ci"], row["peak_memory_bytes"], row["draft_acceptance_rate"]
        assert isinstance(speed, float) and isinstance(latency, float)
        assert isinstance(ci, list)
        mem_text = f"{mem / 2**30:.3f}" if isinstance(mem, int) else "—"
        rate_text = f"{rate:.1%}" if isinstance(rate, float) else "—"
        lines.append(
            f"| {label} | {latency:.2f} | {speed:.3f}× [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{mem_text} | {rate_text} | "
            f"{row['exact_token_matches_vs_plain']}/{row['request_count']} |"
        )
    return "\n".join(lines) + "\n"


def run_speculation_benchmark(
    config: ExperimentConfig,
    backend: Backend,
    draft_tokens: int = 4,
    progress: Callable[[RunProgress], None] | None = None,
) -> Path:
    """在整个输入快照上测三种方法，保存原始记录与论文主表。

    Args:
        config: 固定长度配置，控制输入快照、每方法预热次数及正式轮次。
        backend: 支持投机计时的真实后端，函数退出时统一关闭。
        draft_tokens: 固定草稿和自适应草稿使用的共同上限。
        progress: 每个三路配对结束后的进度回调。

    Returns:
        包含 manifest.json、dataset.jsonl、pairs.jsonl 和 table.md 的新目录。
    """
    try:
        validate_speculation(config, draft_tokens)
        if not isinstance(backend, SpeculationTimingBackend) or backend.info.synthetic:
            raise ValueError("正式实验要求支持草稿计数的真实后端")
        raw = config.dataset.read_bytes()
        prompts = parse_prompts(raw.decode("utf-8-sig"), str(config.dataset))
        output = config.output_dir / f"{config.name}-speculation-{uuid4().hex[:12]}"
        output.mkdir(parents=True, exist_ok=False)
        (output / "dataset.jsonl").write_bytes(raw)
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
            "prompt_count": len(prompts),
            "draft_tokens": draft_tokens,
            "max_matching_ngram_size": 2,
            "adaptive_policy": "half when not fully accepted, +1 on full acceptance, "
            "skip next two lookups after two consecutive zero-acceptance rounds",
            "timing": "synchronized HF generate, includes prefill, no streamer",
            "bootstrap": {"unit": "paired prompt_id", "samples": 1000, "seed": 42},
            "completed_groups": 0,
        }
        write_json(output / "manifest.json", report)
        groups: dict[str, list[dict[str, NativeTiming]]] = {prompt.id: [] for prompt in prompts}
        try:
            started = perf_counter()
            for index in range(config.warmup):
                prompt = prompts[index % len(prompts)]
                request = GenerationRequest(prompt, config.max_new_tokens, config.seed)
                _measure_methods(backend, request, draft_tokens, index % len(_METHODS))
                if progress:
                    progress(
                        RunProgress(
                            "warmup",
                            index + 1,
                            config.warmup,
                            prompt.id,
                            None,
                            config.repeats,
                            perf_counter() - started,
                        )
                    )
            started = perf_counter()
            completed = 0
            total = len(prompts) * config.repeats
            for repeat in range(config.repeats):
                for index, prompt in enumerate(prompts):
                    request = GenerationRequest(prompt, config.max_new_tokens, config.seed + index)
                    values = _measure_methods(backend, request, draft_tokens, (repeat + index) % 3)
                    append_record(
                        output / "pairs.jsonl",
                        {
                            "prompt_id": prompt.id,
                            "scenario": prompt.scenario,
                            "repeat": repeat,
                            "seed": request.seed,
                            "order": list(values),
                            **{mode: asdict(value) for mode, value in values.items()},
                        },
                    )
                    groups[prompt.id].append(values)
                    completed += 1
                    report["completed_groups"] = completed
                    if progress:
                        progress(
                            RunProgress(
                                "measurement",
                                completed,
                                total,
                                prompt.id,
                                repeat + 1,
                                config.repeats,
                                perf_counter() - started,
                            )
                        )
            summary = _main_summary(groups)
            (output / "table.md").write_text(_main_table(summary), encoding="utf-8")
            report.update(status="completed", summary=summary)
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            report["finished_at_utc"] = datetime.now(UTC).isoformat()
            write_json(output / "manifest.json", report)
        return output
    finally:
        backend.close()
