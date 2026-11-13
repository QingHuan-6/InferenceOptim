"""检查投机试跑的配对边界、失败保留与加载前校验。"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ainference_optim.backends.mock import MockBackend
from ainference_optim.cli import main
from ainference_optim.config import load_config
from ainference_optim.domain import GenerationRequest, NativeTiming
from ainference_optim.speculation import run_speculation_pilot

ROOT = Path(__file__).resolve().parents[1]


class LookupFixture(MockBackend):
    def __init__(self, invalid: bool = False) -> None:
        self.calls: list[tuple[int, int]] = []
        self.closed = False
        self.invalid = invalid

    def lookup_timing(self, request: GenerationRequest, draft_tokens: int) -> NativeTiming:
        self.calls.append((request.seed, draft_tokens))
        if self.invalid:
            raise RuntimeError("fixture failure")
        ids = tuple(range(request.max_new_tokens))
        if draft_tokens:
            ids = (99,) + ids[1:]
        return NativeTiming(1.0 if draft_tokens else 2.0, 3.0, 10, ids, 1024)

    def close(self) -> None:
        self.closed = True


class SpeculationTests(unittest.TestCase):
    def test_paired_order_warmup_exclusion_and_mismatch_reporting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                load_config(ROOT / "configs/4090_hf_smoke.toml"), output_dir=Path(directory)
            )
            backend = LookupFixture()
            path = run_speculation_pilot(config, backend, samples=2, repeats=2)
            report = json.loads((path / "pilot.json").read_text(encoding="utf-8"))
            pairs = [json.loads(row) for row in (path / "pairs.jsonl").read_text().splitlines()]
            self.assertTrue(backend.closed)
            self.assertEqual(len(backend.calls), 10)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["summary"]["generation_speedup"], 2.0)
            self.assertEqual(report["summary"]["token_match_pairs"], 0)
            self.assertTrue(all(row["first_mismatch_index"] == 0 for row in pairs))
            self.assertEqual(
                [row["order"] for row in pairs],
                [
                    ["plain", "lookup"],
                    ["lookup", "plain"],
                    ["lookup", "plain"],
                    ["plain", "lookup"],
                ],
            )
            self.assertIsNone(report["acceptance_rate"])
            self.assertNotIn("tpot_s", report["summary"])
            self.assertEqual(pairs[0]["lookup"]["peak_memory_bytes"], 1024)
            for a, b in zip(backend.calls[::2], backend.calls[1::2], strict=True):
                self.assertEqual(a[0], b[0])

    def test_failed_warmup_keeps_report_and_releases_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                load_config(ROOT / "configs/4090_hf_smoke.toml"), output_dir=Path(directory)
            )
            backend = LookupFixture(invalid=True)
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                run_speculation_pilot(config, backend)
            report = json.loads(
                next(Path(directory).glob("*/pilot.json")).read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["completed_pairs"], 0)
            self.assertTrue(backend.closed)

    def test_cli_rejects_large_pilot_before_loading_model(self) -> None:
        with (
            patch("ainference_optim.cli.create_backend") as factory,
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(
                main(
                    [
                        "speculation-pilot",
                        "--config",
                        str(ROOT / "configs/4090_hf_smoke.toml"),
                        "--samples",
                        "100",
                    ]
                ),
                1,
            )
            factory.assert_not_called()
