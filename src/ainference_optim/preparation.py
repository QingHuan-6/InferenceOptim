"""将本地公开原始数据转为固定抽样的性能输入，并保存来源哈希。"""

import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ainference_optim.data import parse_prompts
from ainference_optim.results import write_json


def prepare_baseline_data(
    alpaca: Path,
    humaneval: Path,
    output: Path,
    seed: int = 42,
    alpaca_count: int = 100,
    humaneval_count: int = 50,
    max_chars: int = 2000,
) -> Path:
    """从已有原始文件抽样，不联网、不截断文本、不执行题目测试代码。

    Args:
        alpaca: Stanford alpaca_data.json 原始文件。
        humaneval: HumanEval.jsonl.gz 或解压后的 JSONL。
        output: 必须不存在的输出目录。
        seed: 专用随机数发生器的种子。
        alpaca_count: 指令样本数，必须为正。
        humaneval_count: 代码样本数，必须为正。
        max_chars: 抽样前的字符长度上限，不是 tokenizer 长度保证。

    Returns:
        包含两个 JSONL 和 preparation.json 的目录。

    Raises:
        ValueError: 原始字段不合法、合格样本不足或计数非法。
        OSError: 输入不可读或输出目录已存在。
    """
    if min(alpaca_count, humaneval_count, max_chars) < 1 or seed < 0:
        raise ValueError("样本数和字符上限须为正，seed 须非负")
    raw_alpaca, raw_humaneval = alpaca.read_bytes(), humaneval.read_bytes()
    instructions = json.loads(raw_alpaca)
    if not isinstance(instructions, list):
        raise ValueError("Alpaca 必须是 JSON 数组")
    code_text = (
        gzip.decompress(raw_humaneval) if raw_humaneval[:2] == b"\x1f\x8b" else raw_humaneval
    ).decode("utf-8-sig")
    pools: dict[str, list[dict[str, str]]] = {"alpaca": [], "humaneval": []}
    for index, row in enumerate(instructions):
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) for key in ("instruction", "input")
        ):
            raise ValueError(f"Alpaca 第 {index} 项缺少字符串 instruction/input")
        prompt = row["instruction"].strip()
        if row["input"].strip():
            prompt += "\n\n" + row["input"].strip()
        if prompt and len(prompt) <= max_chars:
            pools["alpaca"].append(
                {"id": f"alpaca-{index:05d}", "scenario": "chat", "prompt": prompt}
            )
    seen: set[str] = set()
    for line_number, line in enumerate(code_text.splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) for key in ("task_id", "prompt")
        ):
            raise ValueError(f"HumanEval 第 {line_number} 行缺少 task_id/prompt")
        if not item["task_id"] or item["task_id"] in seen:
            raise ValueError("HumanEval 的 task_id 为空或重复")
        seen.add(item["task_id"])
        if item["prompt"].strip() and len(item["prompt"]) <= max_chars:
            pools["humaneval"].append(
                {"id": item["task_id"], "scenario": "code", "prompt": item["prompt"]}
            )
    prepared: dict[str, bytes] = {}
    selections: dict[str, Any] = {}
    for name, count in (("alpaca", alpaca_count), ("humaneval", humaneval_count)):
        if len(pools[name]) < count:
            raise ValueError(f"{name} 合格样本 {len(pools[name])} 少于所需 {count}")
        # 分别重建 RNG，某一数据集的样本数不影响另一集合的抽样。
        selected = random.Random(seed).sample(pools[name], count)
        raw = ("\n".join(json.dumps(row, ensure_ascii=False) for row in selected) + "\n").encode(
            "utf-8"
        )
        parse_prompts(raw.decode("utf-8"), name)
        prepared[name] = raw
        selections[name] = {
            "eligible_count": len(pools[name]),
            "selected_ids": [row["id"] for row in selected],
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in prepared.items():
        (output / f"{name}.jsonl").write_bytes(raw)
    write_json(
        output / "preparation.json",
        {
            "preparation_version": 1,
            "seed": seed,
            "max_chars": max_chars,
            "sampling": "random.Random(seed).sample; source order; independent RNG per dataset",
            "sources": {
                "alpaca": {
                    "path": str(alpaca.resolve()),
                    "sha256": hashlib.sha256(raw_alpaca).hexdigest(),
                },
                "humaneval": {
                    "path": str(humaneval.resolve()),
                    "sha256": hashlib.sha256(raw_humaneval).hexdigest(),
                },
            },
            "selection": selections,
            "scope": "performance prompts only; no task quality scoring",
        },
    )
    return output
