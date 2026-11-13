"""按白名单生成远程实验上传包，不打包环境、权重、输出或登录资料。"""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    """将可审查的源文件写入唯一 ZIP，并打印绝对路径。"""
    root = Path(__file__).resolve().parents[1]
    files = [
        root / name
        for name in (
            "README.md",
            "pyproject.toml",
            "AGENTS.md",
            ".gitignore",
            ".gitattributes",
            ".editorconfig",
            "scripts/setup_4090.sh",
            "scripts/package_remote.py",
            "docs/remote_4090.md",
            "docs/engineering.md",
            "docs/git_workflow.md",
            "docs/experiment_protocol.md",
            "docs/benchmark_references.md",
            "docs/evaluation_metrics.md",
            "data/smoke.jsonl",
            "configs/mock.toml",
            "configs/4090_smoke.toml",
            "configs/4090_baseline.toml",
            "configs/4090_deterministic.toml",
        )
    ]
    for directory in (root / "src/ainference_optim", root / "tests"):
        files.extend(directory.rglob("*.py"))
    if any(path.is_symlink() for path in files):
        raise ValueError("上传白名单中含符号链接，请人工核对")
    output = root / "outputs/deploy"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    archive = output / f"AInferenceOptim-{stamp}-{uuid4().hex[:8]}.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(set(files)):
            content = path.read_bytes()
            if path.suffix == ".sh":
                content = content.replace(b"\r\n", b"\n")
            bundle.writestr("AInferenceOptim/" + path.relative_to(root).as_posix(), content)
    print(archive)


if __name__ == "__main__":
    main()
