"""用失败报告的共同前缀组织原始 logits 探针，不修改原报告或性能记录。"""

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ainference_optim.config import ExperimentConfig
from ainference_optim.data import parse_prompts
from ainference_optim.domain import Backend, GenerationRequest, LogitProbeBackend
from ainference_optim.results import SCHEMA_VERSION, write_json


def diagnose_backend(
    config: ExperimentConfig, backend: Backend, source_report: Path, repeats: int = 3
) -> Path:
    """从首次跨路径不一致处提取共同前缀，保存隔离诊断结果。

    Args:
        config: 本次诊断的配置，可显式改精度作独立对照。
        backend: 支持固定前缀探针的后端，退出时总会关闭。
        source_report: 先前 verify 输出的 verification.json。
        repeats: 每条前缀计算路径的重复次数，至少为 2。

    Returns:
        新建诊断目录；其中状态 completed 仅代表探针执行完成，不代表验证通过。

    Raises:
        ValueError: 缺少可用分叉、快照不符或模型/模板不一致。
        Exception: 文件或后端执行失败，已有诊断报告会保留错误状态。
    """
    try:
        if repeats < 2 or not isinstance(backend, LogitProbeBackend):
            raise ValueError("诊断要求支持 logits 探针的后端和 repeats≥2")
        source_report = source_report.resolve()
        source_bytes = source_report.read_bytes()
        source = json.loads(source_bytes)
        row = next((row for row in source["requests"] if not row["match"]), None)
        if row is None:
            raise ValueError("报告没有跨路径 token 分叉，不能建立该探针对照")
        old_config = source["config"]["transformers"]
        if config.transformers is None or any(
            getattr(config.transformers, key) != old_config[key]
            for key in ("model", "revision", "chat_template", "stop_on_eos")
        ):
            raise ValueError("诊断与原报告的模型路径、版本、模板或停止策略不一致")
        actual, reference = row["actual"], row["reference"]
        position = next(
            (
                index
                for index in range(min(len(actual), len(reference)))
                if actual[index] != reference[index]
            ),
            None,
        )
        if position is None:
            raise ValueError("该报告仅长度不同，尚无可对照的分叉 token")
        raw = (source_report.parent / "dataset.jsonl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["dataset_sha256"]:
            raise ValueError("原验证报告的数据快照哈希不符")
        prompt = next(
            (
                item
                for item in parse_prompts(raw.decode("utf-8-sig"), str(source_report))
                if item.id == row["prompt_id"]
            ),
            None,
        )
        if prompt is None:
            raise ValueError("数据快照中找不到失败的 prompt_id")
        request = GenerationRequest(prompt, source["config"]["max_new_tokens"], row["seed"])
        prefix = tuple(actual[:position])
        output = config.output_dir / f"{config.name}-diagnose-{uuid4().hex[:12]}"
        output.mkdir(parents=True, exist_ok=False)
        snapshot = asdict(config)
        snapshot.update(dataset=str(config.dataset), output_dir=str(config.output_dir))
        report: dict[str, object] = {
            "config": snapshot,
            "probe_repeats": repeats,
            "dataset_sha256": source["dataset_sha256"],
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "backend": asdict(backend.info),
            "source_report": str(source_report),
            "source_report_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_backend": source["backend"],
            "prompt_id": prompt.id,
            "seed": request.seed,
            "probe_position_0based": position,
            "prefix_token_ids": prefix,
            "source_actual_token": actual[position],
            "source_reference_token": reference[position],
            "scope": "fixed-prefix raw-logit diagnostic, not performance or equivalence acceptance",
        }
        write_json(output / "diagnostics.json", report)
        try:
            report["observations"] = [
                asdict(item) for item in backend.probe_logits(request, prefix, repeats)
            ]
            report["status"] = "completed"
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            report["finished_at_utc"] = datetime.now(UTC).isoformat()
            write_json(output / "diagnostics.json", report)
        return output
    finally:
        backend.close()
