"""下载固定官方版本的原始数据，并生成第一轮实验的固定输入。"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from ainference_optim.preparation import prepare_baseline_data

SOURCES = {
    "alpaca_data.json": (
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
        "761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json"
    ),
    "HumanEval.jsonl.gz": (
        "https://raw.githubusercontent.com/openai/human-eval/"
        "b9453e8f1865df0a0057a1383767caa400acfbcf/data/HumanEval.jsonl.gz"
    ),
}
EXPECTED_SHA256 = {
    "alpaca_data.json": "2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419",
    "HumanEval.jsonl.gz": "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
}


def main() -> None:
    """只下载公开数据；不加载模型、不使用 GPU、不覆盖已有输入快照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/baseline-v1"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"输入目录已存在，请复用原快照或指定新目录：{output}")
    raw_dir = root / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for name, url in SOURCES.items():
        path = raw_dir / name
        # 使用专用文件名，已有文件保留；固定 URL 与实际字节哈希共同记录。
        downloaded_now = not path.exists()
        if downloaded_now:
            print(f"下载 {name}：{url}", flush=True)
            with urlopen(url, timeout=90) as response:
                content = response.read()
        else:
            content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != EXPECTED_SHA256[name]:
            raise ValueError(f"{path} 不符合固定官方版本的 SHA256；保留原文件，请核对来源")
        if downloaded_now:
            with path.open("xb") as handle:
                handle.write(content)
        source_hashes[name] = {"url": url, "sha256": digest, "downloaded_now": downloaded_now}
    result = prepare_baseline_data(
        raw_dir / "alpaca_data.json", raw_dir / "HumanEval.jsonl.gz", output
    )
    # 仅在本次新建目录中补充固定下载入口，已有快照不会走到这里。
    (result / "download_sources.json").write_text(
        json.dumps(source_hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"固定输入已生成：{result}；Alpaca 100 条，HumanEval 50 条；未使用 GPU")


if __name__ == "__main__":
    main()
