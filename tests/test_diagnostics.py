"""验证固定前缀提取、输入溯源、失败留存和资源释放。"""

import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from ainference_optim.backends.mock import MockBackend
from ainference_optim.config import load_config
from ainference_optim.diagnostics import diagnose_backend
from ainference_optim.domain import GenerationRequest, LogitProbe

ROOT = Path(__file__).resolve().parents[1]


class ProbeFixture(MockBackend):
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.closed = False
        self.received: tuple[GenerationRequest, tuple[int, ...], int] | None = None

    def probe_logits(
        self, request: GenerationRequest, prefix: tuple[int, ...], repeats: int
    ) -> tuple[LogitProbe, ...]:
        self.received = request, prefix, repeats
        if self.fail:
            raise RuntimeError("probe failed")
        return (
            LogitProbe("cache_replay", 0, True, 0, (2, 3), (1.0, 0.9), 0.1, "test", True, 0.0, 2),
        )

    def close(self) -> None:
        self.closed = True


class DiagnosticTests(unittest.TestCase):
    def test_extracts_common_prefix_without_modifying_source_and_retains_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(
                load_config(ROOT / "configs/4090_deterministic.toml"), output_dir=root / "outputs"
            )
            raw = config.dataset.read_bytes()
            (root / "dataset.jsonl").write_bytes(raw)
            source = {
                "config": {"transformers": asdict(config.transformers), "max_new_tokens": 128},
                "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                "backend": {},
                "requests": [
                    {
                        "prompt_id": "chat-001",
                        "seed": 42,
                        "match": False,
                        "actual": [1, 2, 3],
                        "reference": [1, 2, 4],
                    }
                ],
            }
            source_path = root / "verification.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            before = source_path.read_bytes()
            backend = ProbeFixture()
            output = diagnose_backend(config, backend, source_path)
            report = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
            self.assertTrue(backend.closed)
            self.assertIsNotNone(backend.received)
            assert backend.received is not None
            request, prefix, repeats = backend.received
            self.assertEqual((prefix, repeats, request.seed), ((1, 2), 3, 42))
            self.assertEqual(report["probe_position_0based"], 2)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(source_path.read_bytes(), before)
            failed = ProbeFixture(fail=True)
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                diagnose_backend(config, failed, source_path)
            self.assertTrue(failed.closed)
            reports = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in config.output_dir.glob("*/diagnostics.json")
            ]
            self.assertEqual(sorted(item["status"] for item in reports), ["completed", "failed"])
            (root / "dataset.jsonl").write_bytes(b"changed")
            corrupt = ProbeFixture()
            with self.assertRaisesRegex(ValueError, "哈希"):
                diagnose_backend(config, corrupt, source_path)
            self.assertTrue(corrupt.closed)
            self.assertIsNone(corrupt.received)


if __name__ == "__main__":
    unittest.main()
