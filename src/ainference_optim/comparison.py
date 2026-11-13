"""校验两个串行实验的可比性，按提示配对汇总真实计时。"""

import hashlib
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

from ainference_optim.domain import GenerationResult
from ainference_optim.metrics import measure, percentile
from ainference_optim.results import SCHEMA_VERSION, write_json


def _load_run(path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("measurement_mode") != "serial_single_request"
    ):
        raise ValueError("只能比较已完成的串行实验")
    raw = (path / "dataset.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("实验输入快照哈希不符")
    groups: dict[str, list[dict[str, Any]]] = {}
    for line in (path / "requests.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["synthetic"] != manifest["backend"]["synthetic"]:
            raise ValueError("请求与后端 synthetic 声明冲突")
        result = GenerationResult(**row["result"])
        metric = measure(result)
        if metric.output_tokens != manifest["config"]["max_new_tokens"] or metric.output_tokens < 2:
            raise ValueError("主对照要求固定输出长度且至少两个 token")
        row["measured"] = {
            "decode_elapsed_s": result.token_times_s[-1] - result.token_times_s[0],
            "tpot_s": metric.tpot_s,
            "elapsed_s": metric.elapsed_s,
            "ttft_s": metric.ttft_s,
            "decode_tokens_per_s": metric.decode_tokens_per_s,
            "peak_memory_bytes": metric.peak_memory_bytes,
        }
        groups.setdefault(row["prompt_id"], []).append(row)
    expected_ids = {
        json.loads(line)["id"] for line in raw.decode("utf-8-sig").splitlines() if line.strip()
    }
    if set(groups) != expected_ids:
        raise ValueError("测量记录未完整覆盖输入快照")
    for rows in groups.values():
        if sorted(row["repeat"] for row in rows) != list(range(manifest["config"]["repeats"])):
            raise ValueError("重复测量缺失或重复")
    return manifest, groups


def _group_summary(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "prompt_count": len(groups),
        "request_count": sum(map(len, groups.values())),
    }
    for key in ("decode_elapsed_s", "tpot_s", "elapsed_s", "ttft_s", "decode_tokens_per_s"):
        values = [row["measured"][key] for rows in groups.values() for row in rows]
        prompt_means = [
            mean(row["measured"][key] for row in rows)
            for rows in groups.values()
            if all(row["measured"][key] is not None for row in rows)
        ]
        valid = [float(value) for value in values if value is not None]
        summary[key] = {
            "mean": mean(prompt_means) if prompt_means else None,
            "valid_prompt_count": len(prompt_means),
            "p50": percentile(valid, 0.5),
            "p99": percentile(valid, 0.99),
        }
    memories = [
        row["measured"]["peak_memory_bytes"]
        for rows in groups.values()
        for row in rows
        if row["measured"]["peak_memory_bytes"] is not None
    ]
    summary["peak_memory_bytes_max"] = max(memories) if memories else None
    return summary


def compare_runs(baseline: Path, candidate: Path, output_root: Path) -> Path:
    """按同一提示和重复编号配对，保存分场景加速比及提示级 bootstrap 区间。

    Args:
        baseline: 已完成的固定长度基线实验目录。
        candidate: 相同输入、环境和生成条件的候选实验目录。
        output_root: 新建唯一 comparison 子目录的父目录。

    Returns:
        comparison.json 所在目录；结果不代表无损验收。

    Raises:
        ValueError: 实验不完整、数据或控制变量不一致。
        OSError: 记录不可读或无法写入报告。
    """
    left, left_groups = _load_run(baseline)
    right, right_groups = _load_run(candidate)
    if left["dataset_sha256"] != right["dataset_sha256"]:
        raise ValueError("对比必须使用相同数据快照")
    if left["backend"]["synthetic"] != right["backend"]["synthetic"]:
        raise ValueError("不能将模拟结果与真实后端比较")
    for field in ("max_new_tokens", "repeats", "seed", "warmup"):
        if left["config"][field] != right["config"][field]:
            raise ValueError(f"对照配置 {field} 不一致")
    for key in ("model", "device"):
        if left["backend"][key] != right["backend"][key]:
            raise ValueError(f"后端 {key} 不一致")
    old, new = left["config"]["transformers"], right["config"]["transformers"]
    if old is not None or new is not None:
        if old is None or new is None:
            raise ValueError("真实后端配置类型不同")
        for key in set(old) | set(new):
            if key != "generation_path" and old.get(key) != new.get(key):
                raise ValueError(f"对照 backend.{key} 不一致")
        if old["stop_on_eos"]:
            raise ValueError("主对照要求 stop_on_eos=false；自然停止需独立分析")
        for key in (
            "torch",
            "transformers",
            "cuda_runtime",
            "gpu_name",
            "driver_version",
            "resolved_revision",
            "token_observation",
        ):
            if left["backend"]["metadata"].get(key) != right["backend"]["metadata"].get(key):
                raise ValueError(f"环境/计时元数据 {key} 不一致")
    comparisons: dict[str, Any] = {}
    for name in sorted({rows[0]["scenario"] for rows in left_groups.values()}):
        ids = sorted(key for key, rows in left_groups.items() if rows[0]["scenario"] == name)
        a = {key: left_groups[key] for key in ids}
        b = {key: right_groups[key] for key in ids}
        for key in ids:
            for x, y in zip(
                sorted(a[key], key=lambda row: row["repeat"]),
                sorted(b[key], key=lambda row: row["repeat"]),
                strict=True,
            ):
                if (x["scenario"], x["seed"], x["result"]["input_tokens"]) != (
                    y["scenario"],
                    y["seed"],
                    y["result"]["input_tokens"],
                ):
                    raise ValueError(f"请求 {key} 的场景、种子或输入 token 数不一致")
        sa, sb = _group_summary(a), _group_summary(b)
        durations_a = [mean(row["measured"]["decode_elapsed_s"] for row in a[key]) for key in ids]
        durations_b = [mean(row["measured"]["decode_elapsed_s"] for row in b[key]) for key in ids]
        if min(durations_a + durations_b) <= 0:
            raise ValueError("解码时间必须为正，不能计算零耗时加速比")
        rng = random.Random(42)
        ratios = []
        for _ in range(1000):
            indices = rng.choices(range(len(ids)), k=len(ids))
            ratios.append(
                sum(durations_a[i] for i in indices) / sum(durations_b[i] for i in indices)
            )
        comparisons[name] = {
            "baseline": sa,
            "candidate": sb,
            "decode_speedup": mean(durations_a) / mean(durations_b),
            "e2e_speedup": sa["elapsed_s"]["mean"] / sb["elapsed_s"]["mean"],
            "decode_speedup_ci95": [percentile(ratios, 0.025), percentile(ratios, 0.975)]
            if len(ids) >= 2
            else None,
        }
    output = output_root.resolve() / f"comparison-{uuid4().hex[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "comparison.json",
        {
            "schema_version": SCHEMA_VERSION,
            "synthetic": left["backend"]["synthetic"],
            "baseline": str(baseline.resolve()),
            "candidate": str(candidate.resolve()),
            "baseline_manifest": left,
            "candidate_manifest": right,
            "aggregation": (
                "repeat mean per prompt, then equal prompt mean; percentiles over requests"
            ),
            "bootstrap": {"unit": "paired prompt_id", "samples": 1000, "seed": 42},
            "scope": "fixed-length timing comparison only; not correctness or quality acceptance",
            "by_scenario": comparisons,
        },
    )
    return output
