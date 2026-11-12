"""负责实验输出的 JSON 序列化与状态记录。"""

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


def write_json(path: Path, value: Any) -> None:
    """以严格 JSON 写入，并通过同目录替换避免留下半个状态文件。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_record(path: Path, value: Any) -> None:
    """追加一条完整请求记录，便于失败后保留已完成的数据。"""
    line = json.dumps(value, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
