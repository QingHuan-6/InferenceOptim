"""核对后端与参考生成的 token 序列，不把验证运行计入性能汇总。"""

import hashlib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ainference_optim.config import ExperimentConfig
from ainference_optim.data import parse_prompts
from ainference_optim.domain import Backend, GenerationRequest, ReferenceBackend
from ainference_optim.metrics import measure
from ainference_optim.results import SCHEMA_VERSION, write_json


def verify_backend(
    config: ExperimentConfig, backend: Backend, samples: int = 3
) -> tuple[Path, bool]:
    """保存逐样本 token 一致性报告，并管理后端生命周期。

    Args:
        config: 与拟运行性能实验相同的配置。
        backend: 支持 reference_tokens 的后端；函数退出时关闭。
        samples: 最多核对的数据集前缀样本数，必须大于零。

    Returns:
        独立报告目录和所有已选样本是否通过；不代表分布等价性。

    Raises:
        ValueError: 参数不合法或后端不支持参考路径。
        Exception: 后端执行失败；已有报告保存失败状态。
    """
    try:
        if samples < 1 or not isinstance(backend, ReferenceBackend):
            raise ValueError("验证需要正样本数和支持参考生成的后端")
        raw = config.dataset.read_bytes()
        prompts = parse_prompts(raw.decode("utf-8-sig"), str(config.dataset))[:samples]
        now = datetime.now(UTC)
        output = config.output_dir / f"{config.name}-verify-{now:%Y%m%dT%H%M%S}-{uuid4().hex[:12]}"
        output.mkdir(parents=True, exist_ok=False)
        snapshot = asdict(config)
        snapshot.update(dataset=str(config.dataset), output_dir=str(config.output_dir))
        rows: list[dict[str, object]] = []
        report: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "passed": False,
            "started_at_utc": now.isoformat(),
            "backend": asdict(backend.info),
            "config": snapshot,
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "scope": "token equality for selected greedy requests, not distribution equivalence",
            "requests": rows,
        }
        write_json(output / "verification.json", report)
        (output / "dataset.jsonl").write_bytes(raw)
        try:
            for index, prompt in enumerate(prompts):
                request = GenerationRequest(prompt, config.max_new_tokens, config.seed + index)
                actual = backend.generate(request)
                measure(actual)
                if not actual.token_ids or len(actual.token_ids) > config.max_new_tokens:
                    raise ValueError("验证路径必须返回非空、未超出上限的实际 token_ids")
                reference = backend.reference_tokens(request)
                first_mismatch = next(
                    (
                        position
                        for position in range(max(len(actual.token_ids), len(reference)))
                        if position >= len(actual.token_ids)
                        or position >= len(reference)
                        or actual.token_ids[position] != reference[position]
                    ),
                    None,
                )
                rows.append(
                    {
                        "prompt_id": prompt.id,
                        "seed": request.seed,
                        "actual": actual.token_ids,
                        "reference": reference,
                        "match": actual.token_ids == reference,
                        "actual_length": len(actual.token_ids),
                        "reference_length": len(reference),
                        "first_mismatch_index": first_mismatch,
                    }
                )
            report["passed"] = all(row["match"] for row in rows)
            report["status"] = "completed"
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            report["finished_at_utc"] = datetime.now(UTC).isoformat()
            write_json(output / "verification.json", report)
        return output, bool(report["passed"])
    finally:
        backend.close()
