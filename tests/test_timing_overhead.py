"""验证开销对照的配对次序、计数、非法观测与资源释放。"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ainference_optim.backends.mock import MockBackend
from ainference_optim.cli import main
from ainference_optim.config import load_config
from ainference_optim.domain import GenerationRequest, NativeTiming
from ainference_optim.timing_overhead import measure_timing_overhead

ROOT = Path(__file__).resolve().parents[1]


class TimingFixture(MockBackend):
    def __init__(self, invalid: bool = False) -> None:
        self.calls: list[tuple[int, bool]] = []
        self.closed = False
        self.invalid = invalid

    def native_timing(self, request: GenerationRequest, streamer: bool) -> NativeTiming:
        self.calls.append((request.seed, streamer))
        return NativeTiming(
            2.0 if streamer else 1.0,
            2.5 if streamer else 1.5,
            5,
            tuple(range(request.max_new_tokens - int(self.invalid))),
        )

    def close(self) -> None:
        self.closed = True


class TimingOverheadTests(unittest.TestCase):
    def test_alternating_pairs_have_same_seed_and_warmup_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                load_config(ROOT / "configs/4090_hf_alpaca.toml"),
                dataset=ROOT / "data/smoke.jsonl",
                output_dir=Path(directory),
            )
            backend = TimingFixture()
            output = measure_timing_overhead(config, backend, samples=2, repeats=2, warmup=1)
            self.assertTrue(backend.closed)
            report = json.loads((output / "overhead.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["completed_pairs"], 4)
            self.assertEqual(len(backend.calls), 10)
            pairs = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
            self.assertEqual(
                [row["order"] for row in pairs],
                [
                    ["streamer", "plain"],
                    ["plain", "streamer"],
                    ["plain", "streamer"],
                    ["streamer", "plain"],
                ],
            )
            for offset in range(0, len(backend.calls), 2):
                self.assertEqual(backend.calls[offset][0], backend.calls[offset + 1][0])
            summary = report["summary"]
            self.assertEqual(summary["generation_elapsed_s"]["streamer_overhead_percent"], 100.0)
            self.assertEqual(summary["token_match_pairs"], 4)
            self.assertNotIn("ttft_s", summary)

    def test_invalid_observation_keeps_failure_report_and_closes_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                load_config(ROOT / "configs/4090_hf_alpaca.toml"),
                dataset=ROOT / "data/smoke.jsonl",
                output_dir=Path(directory),
            )
            backend = TimingFixture(invalid=True)
            with self.assertRaisesRegex(ValueError, "固定长度"):
                measure_timing_overhead(config, backend)
            self.assertTrue(backend.closed)
            path = next(Path(directory).glob("*/overhead.json"))
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual((report["status"], report["completed_pairs"]), ("failed", 0))

    def test_cli_and_invalid_parameters_do_not_load_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                load_config(ROOT / "configs/4090_hf_alpaca.toml"),
                dataset=ROOT / "data/smoke.jsonl",
                output_dir=Path(directory),
            )
            with (
                patch("ainference_optim.cli.load_config", return_value=config),
                patch(
                    "ainference_optim.cli.create_backend", return_value=TimingFixture()
                ) as factory,
                redirect_stdout(io.StringIO()) as display,
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    main(["timing-overhead", "--config", "unused", "--samples", "0"]), 1
                )
                factory.assert_not_called()
                self.assertEqual(
                    main(
                        [
                            "timing-overhead",
                            "--config",
                            "unused",
                            "--samples",
                            "1",
                            "--repeats",
                            "2",
                            "--warmup",
                            "1",
                            "--quiet",
                        ]
                    ),
                    0,
                )
            self.assertIn("合成计时", display.getvalue())
            self.assertIn("streamer_overhead_percent", display.getvalue())


if __name__ == "__main__":
    unittest.main()
