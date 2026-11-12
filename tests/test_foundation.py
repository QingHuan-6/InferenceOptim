"""验证实验契约、错误路径与关键指标口径，不依赖设备或第三方库。"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from ainference_optim.backends import create_backend
from ainference_optim.backends.mock import MockBackend
from ainference_optim.cli import main
from ainference_optim.config import ExperimentConfig, load_config
from ainference_optim.data import load_prompts
from ainference_optim.domain import GenerationRequest, GenerationResult
from ainference_optim.metrics import measure, percentile
from ainference_optim.runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]


class MetricsTests(unittest.TestCase):
    def test_ttft_is_excluded_from_tpot(self) -> None:
        metrics = measure(GenerationResult("abc", 4, (1.0, 1.2, 1.6), 1.8))
        self.assertEqual(metrics.ttft_s, 1.0)
        self.assertAlmostEqual(metrics.tpot_s, 0.3)
        self.assertAlmostEqual(metrics.decode_tokens_per_s, 1 / 0.3)

    def test_empty_single_and_burst_outputs(self) -> None:
        empty = measure(GenerationResult("", 1, (), 0.1))
        self.assertIsNone(empty.ttft_s)
        self.assertIsNone(empty.tpot_s)
        single = measure(GenerationResult("x", 1, (0.1,), 0.2))
        self.assertIsNone(single.tpot_s)
        burst = measure(GenerationResult("xx", 1, (0.1, 0.1), 0.2))
        self.assertEqual(burst.tpot_s, 0)
        self.assertIsNone(burst.decode_tokens_per_s)

    def test_invalid_measurements_are_rejected(self) -> None:
        baseline = GenerationResult("", 1, (0.1, 0.2), 0.3)
        for result in (
            replace(baseline, elapsed_s=float("nan")),
            replace(baseline, token_times_s=(0.2, 0.1)),
            replace(baseline, token_times_s=(-0.1,)),
            replace(baseline, token_times_s=(float("inf"),)),
            replace(baseline, elapsed_s=0.1),
            replace(baseline, input_tokens=-1),
            replace(baseline, peak_memory_bytes=-1),
        ):
            with self.subTest(result=result), self.assertRaises(ValueError):
                measure(result)

    def test_percentile_interpolation(self) -> None:
        self.assertEqual(percentile([0, 10], 0.99), 9.9)
        self.assertIsNone(percentile([], 0.5))
        with self.assertRaises(ValueError):
            percentile([1], 2)


class InputTests(unittest.TestCase):
    def test_paths_are_relative_to_config_not_working_directory(self) -> None:
        config = load_config(ROOT / "configs/mock.toml")
        self.assertEqual(config.dataset, ROOT / "data/smoke.jsonl")

    def test_config_rejects_typo_and_boolean_count(self) -> None:
        source = (ROOT / "configs/mock.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.toml"
            for text in (
                source.replace("repeats = 2", "repeat = 2"),
                source.replace("repeats = 2", "repeats = true"),
                source.replace('name = "mock-smoke"', 'name = "../escape"'),
                source.replace("max_new_tokens = 16", "max_new_tokens = 0"),
            ):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_config(path)

    def test_dataset_reports_line_of_duplicate_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            line = json.dumps({"id": "x", "scenario": "chat", "prompt": "你好"})
            path.write_text(f"{line}\n{line}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r":2:.*重复"):
                load_prompts(path)

    def test_invalid_or_empty_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            for text in ("", "[]", "{bad", '{"id":"x"}'):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_prompts(path)

    def test_unknown_backend_never_falls_back(self) -> None:
        with self.assertRaises(ValueError):
            create_backend("cuda")


class RecordingBackend(MockBackend):
    def __init__(self, fail_at: int | None = None) -> None:
        self.seeds: list[int] = []
        self.closed = False
        self.fail_at = fail_at

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.seeds.append(request.seed)
        if len(self.seeds) == self.fail_at:
            raise RuntimeError("device failed")
        return super().generate(request)

    def close(self) -> None:
        self.closed = True


class RunnerTests(unittest.TestCase):
    def config(self, output: Path) -> ExperimentConfig:
        return replace(load_config(ROOT / "configs/mock.toml"), output_dir=output)

    def test_warmup_excluded_and_results_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            backend = RecordingBackend()
            output = run_experiment(config, backend)
            self.assertTrue(backend.closed)
            self.assertEqual(backend.seeds, list(range(42, 49)))
            rows = [
                json.loads(line)
                for line in (output / "requests.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["seed"], 43)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["synthetic"])
            self.assertEqual(summary["by_scenario"]["code"]["request_count"], 2)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual((output / "dataset.jsonl").read_bytes(), config.dataset.read_bytes())
            second = run_experiment(config, RecordingBackend())
            self.assertNotEqual(output, second)
            self.assertEqual(
                (output / "requests.jsonl").read_bytes(),
                (second / "requests.jsonl").read_bytes(),
            )

    def test_failure_retains_partial_records_and_closes_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            backend = RecordingBackend(fail_at=3)
            with self.assertRaisesRegex(RuntimeError, "device failed"):
                run_experiment(config, backend)
            self.assertTrue(backend.closed)
            output = next(Path(directory).iterdir())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["completed_requests"], 1)
            self.assertFalse((output / "summary.json").exists())

    def test_dataset_failure_still_closes_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = RecordingBackend()
            config = replace(self.config(Path(directory)), dataset=Path(directory) / "missing")
            with self.assertRaises(OSError):
                run_experiment(config, backend)
            self.assertTrue(backend.closed)

    def test_over_budget_output_is_rejected_during_warmup_and_measurement(self) -> None:
        class OverBudgetBackend(RecordingBackend):
            def generate(self, request: GenerationRequest) -> GenerationResult:
                return super().generate(replace(request, max_new_tokens=request.max_new_tokens + 1))

        with tempfile.TemporaryDirectory() as directory:
            for warmup in (0, 1):
                backend = OverBudgetBackend()
                config = replace(self.config(Path(directory)), warmup=warmup)
                with self.subTest(warmup=warmup), self.assertRaises(ValueError):
                    run_experiment(config, backend)
                self.assertTrue(backend.closed)

    def test_cli_report_labels_synthetic_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = run_experiment(self.config(Path(directory)), RecordingBackend())
            stream = io.StringIO()
            with redirect_stdout(stream):
                status = main(["summarize", str(output)])
            self.assertEqual(status, 0)
            self.assertIn("合成数据", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
