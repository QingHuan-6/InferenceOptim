"""读取 JSONL 输入，不依赖模型分词器。"""

import json
from pathlib import Path

from ainference_optim.domain import Prompt


def load_prompts(path: Path) -> list[Prompt]:
    """读取样本并检查字段、空输入与重复 ID。

    Args:
        path: 每行包含 id、scenario、prompt 的 UTF-8 JSONL 文件。

    Returns:
        按文件顺序排列的非空样本列表。

    Raises:
        ValueError: 样本格式不合法，错误信息包含行号。
        OSError: 文件无法读取。
    """
    return parse_prompts(path.read_text(encoding="utf-8-sig"), str(path))


def parse_prompts(content: str, source: str) -> list[Prompt]:
    """解析固定输入快照；source 用于在校验错误中定位原始文件。"""
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not isinstance(item, dict) or set(item) != {"id", "scenario", "prompt"}:
                raise ValueError("需要且仅允许 id、scenario、prompt 三个字段")
            if any(not isinstance(value, str) or not value.strip() for value in item.values()):
                raise ValueError("所有字段必须为非空字符串")
            if item["id"] in seen:
                raise ValueError(f"重复样本 ID：{item['id']}")
            seen.add(item["id"])
            prompts.append(Prompt(item["id"], item["scenario"], item["prompt"]))
        except ValueError as error:
            raise ValueError(f"{source}:{line_number}: {error}") from error
    if not prompts:
        raise ValueError(f"数据集为空：{source}")
    return prompts
