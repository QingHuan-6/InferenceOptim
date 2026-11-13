"""验证数据抽样可复现、实验配对、聚合口径及不可比输入的拒绝。"""

import gzip
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ainference_optim.backends.mock import MockBackend
from ainference_optim.comparison import compare_runs
from ainference_optim.config import load_config
from ainference_optim.domain import GenerationRequest, GenerationResult
from ainference_optim.preparation import prepare_baseline_data
from ainference_optim.runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]


class ScaledMock(MockBackend):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        original = super().generate(request)
        return replace(
            original,
            token_times_s=tuple(value / 2 for value in original.token_times_s),
            elapsed_s=original.elapsed_s / 2,
        )


class BaselineTests(unittest.TestCase):
    def test_sampling_preserves_code_and_provenance_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alpaca, code = root / "alpaca.json", root / "code.gz"
            alpaca.write_text(
                json.dumps(
                    [
                        {"instruction": f"question {i}", "input": "context", "output": "unused"}
                        for i in range(12)
                    ]
                ),
                encoding="utf-8",
            )
            code.write_bytes(
                gzip.compress(
                    (
                        "\n".join(
                            json.dumps(
                                {
                                    "task_id": f"HumanEval/{i}",
                                    "prompt": f"def f{i}():\n    ",
                                    "test": "must not execute",
                                    "canonical_solution": "hidden",
                                }
                            )
                            for i in range(12)
                        )
                    ).encode("utf-8")
                )
            )
            first = prepare_baseline_data(
                alpaca, code, root / "a", alpaca_count=5, humaneval_count=4
            )
            second = prepare_baseline_data(
                alpaca, code, root / "b", alpaca_count=5, humaneval_count=4
            )
            for name in ("alpaca", "humaneval"):
                self.assertEqual(
                    (first / f"{name}.jsonl").read_bytes(), (second / f"{name}.jsonl").read_bytes()
                )
            row = json.loads((first / "humaneval.jsonl").read_text().splitlines()[0])
            self.assertTrue(row["prompt"].endswith("\n    "))
            self.assertEqual(set(row), {"id", "scenario", "prompt"})
            config = replace(
                load_config(ROOT / "configs/mock.toml"),
                dataset=first / "alpaca.jsonl",
                output_dir=root / "runs",
            )
            run = run_experiment(config, MockBackend())
            self.assertEqual(
                (run / "preparation.json").read_bytes(), (first / "preparation.json").read_bytes()
            )
            (first / "alpaca.jsonl").write_text(
                (first / "alpaca.jsonl").read_text(encoding="utf-8").replace("context", "changed"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "抽样哈希"):
                run_experiment(config, MockBackend())
            with self.assertRaises(FileExistsError):
                prepare_baseline_data(alpaca, code, first, alpaca_count=5, humaneval_count=4)
            with self.assertRaisesRegex(ValueError, "合格样本"):
                prepare_baseline_data(alpaca, code, root / "not-created", alpaca_count=100)
            self.assertFalse((root / "not-created").exists())

    def test_comparison_pairs_prompts_and_rejects_incomplete_or_changed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(load_config(ROOT / "configs/mock.toml"), output_dir=Path(directory))
            dataset = Path(directory) / "prompts.jsonl"
            dataset.write_text(
                "\n".join(
                    json.dumps({"id": f"p-{i}", "scenario": "chat", "prompt": f"prompt {i}"})
                    for i in range(3)
                ),
                encoding="utf-8",
            )
            config = replace(config, dataset=dataset)
            left = run_experiment(config, MockBackend())
            right = run_experiment(config, ScaledMock())
            output = compare_runs(left, right, Path(directory))
            report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertTrue(report["synthetic"])
            for group in report["by_scenario"].values():
                self.assertAlmostEqual(group["decode_speedup"], 2.0)
                self.assertAlmostEqual(group["e2e_speedup"], 2.0)
                self.assertEqual(group["decode_speedup_ci95"], [2.0, 2.0])
                self.assertIsNone(group["baseline"]["peak_memory_bytes_max"])
            manifest_path = right / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("code", manifest)
            manifest["config"]["seed"] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed"):
                compare_runs(left, right, Path(directory))
            manifest["config"]["seed"] -= 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            records = (right / "requests.jsonl").read_text(encoding="utf-8").splitlines()
            (right / "requests.jsonl").write_text("\n".join(records[:-1]), encoding="utf-8")
            with self.assertRaises(ValueError):
                compare_runs(left, right, Path(directory))

    def test_presets_keep_old_manual_path_and_validate_native_path(self) -> None:
        old = load_config(ROOT / "configs/4090_baseline.toml")
        assert old.transformers is not None
        self.assertEqual(old.transformers.generation_path, "manual")
        for name in ("alpaca", "humaneval"):
            config = load_config(ROOT / f"configs/4090_hf_{name}.toml")
            assert config.transformers is not None
            self.assertEqual(config.transformers.generation_path, "hf_generate")
            self.assertFalse(config.transformers.stop_on_eos)
            self.assertEqual((config.warmup, config.repeats, config.max_new_tokens), (5, 3, 128))


if __name__ == "__main__":
    unittest.main()
