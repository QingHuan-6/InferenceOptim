"""纯函数指标计算；时间统一用秒，内存统一用字节。"""

import math
from dataclasses import dataclass
from itertools import pairwise

from ainference_optim.domain import GenerationResult


@dataclass(frozen=True)
class RequestMetrics:
    """单请求指标；不足两个输出 token 时 TPOT 与解码速率为空。"""

    input_tokens: int
    output_tokens: int
    ttft_s: float | None
    tpot_s: float | None
    elapsed_s: float
    decode_tokens_per_s: float | None
    peak_memory_bytes: int | None
    decode_elapsed_s: float | None


def measure(result: GenerationResult) -> RequestMetrics:
    """验证时间线并计算单请求指标。

    Args:
        result: 后端返回的生成结果。

    Returns:
        按项目指标口径计算的单请求测量。

    Raises:
        ValueError: 时间非有限、逆序、越界，或 token/内存计数非法。
    """
    times = result.token_times_s
    if result.token_ids and (
        len(result.token_ids) != len(times)
        or any(type(token) is not int or token < 0 for token in result.token_ids)
    ):
        raise ValueError("token_ids 必须与时间线等长且由非负整数构成")
    if not math.isfinite(result.elapsed_s) or result.elapsed_s < 0:
        raise ValueError("elapsed_s 必须是有限非负数")
    if any(not math.isfinite(value) or value < 0 for value in times):
        raise ValueError("token 时间必须是有限非负数")
    if any(right < left for left, right in pairwise(times)):
        raise ValueError("token 时间必须单调不减")
    if times and times[-1] > result.elapsed_s:
        raise ValueError("最后一个 token 不得晚于请求结束")
    if type(result.input_tokens) is not int or result.input_tokens < 0:
        raise ValueError("input_tokens 必须是非负整数")
    memory = result.peak_memory_bytes
    if memory is not None and (type(memory) is not int or memory < 0):
        raise ValueError("peak_memory_bytes 必须是非负整数或 None")
    tpot = (times[-1] - times[0]) / (len(times) - 1) if len(times) >= 2 else None
    return RequestMetrics(
        input_tokens=result.input_tokens,
        output_tokens=len(times),
        ttft_s=times[0] if times else None,
        tpot_s=tpot,
        elapsed_s=result.elapsed_s,
        decode_tokens_per_s=1 / tpot if tpot is not None and tpot > 0 else None,
        peak_memory_bytes=memory,
        decode_elapsed_s=times[-1] - times[0] if len(times) >= 2 else None,
    )


def percentile(values: list[float], fraction: float) -> float | None:
    """使用线性插值计算分位数；空列表返回 None。"""
    if not 0 <= fraction <= 1:
        raise ValueError("分位数必须处于 [0, 1]")
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rows: list[RequestMetrics]) -> dict[str, object]:
    """聚合请求级指标；不将顺序请求速率解释为并发系统吞吐。"""
    summary: dict[str, object] = {"request_count": len(rows)}
    for name in ("ttft_s", "tpot_s", "elapsed_s", "decode_tokens_per_s", "decode_elapsed_s"):
        values = [float(value) for row in rows if (value := getattr(row, name)) is not None]
        summary[name] = {
            "valid_count": len(values),
            "mean": sum(values) / len(values) if values else None,
            "p50": percentile(values, 0.5),
            "p99": percentile(values, 0.99),
        }
    memory = [row.peak_memory_bytes for row in rows if row.peak_memory_bytes is not None]
    summary["peak_memory_bytes_max"] = max(memory) if memory else None
    for name in ("input_tokens", "output_tokens"):
        counts = [getattr(row, name) for row in rows]
        summary[name] = {
            "min": min(counts) if counts else None,
            "max": max(counts) if counts else None,
            "mean": sum(counts) / len(counts) if counts else None,
            "p50": percentile(counts, 0.5),
        }
    return summary
