"""覆盖参考验证报告、错误路径、配置边界及可选依赖隔离。"""

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ainference_optim.backends.mock import MockBackend
from ainference_optim.config import load_config
from ainference_optim.domain import GenerationRequest, GenerationResult
from ainference_optim.verification import verify_backend

ROOT = Path(__file__).resolve().parents[1]


class ReferenceFixture(MockBackend):
    def __init__(self, match: bool = True, fail: bool = False) -> None:
        self.match = match
        self.fail = fail
        self.closed = False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return replace(super().generate(request), token_ids=tuple(range(request.max_new_tokens)))

    def reference_tokens(self, request: GenerationRequest) -> tuple[int, ...]:
        if self.fail:
            raise RuntimeError("reference failed")
        return tuple(range(request.max_new_tokens)) if self.match else (999,)

    def close(self) -> None:
        self.closed = True


class VerificationTests(unittest.TestCase):
    def test_repeated_verification_uses_same_seed_per_prompt(self) -> None:
        class RecordingReference(ReferenceFixture):
            def __init__(self) -> None:
                super().__init__()
                self.seeds = []

            def generate(self, request: GenerationRequest) -> GenerationResult:
                self.seeds.append(request.seed)
                return super().generate(request)

        with tempfile.TemporaryDirectory() as directory:
            config = replace(load_config(ROOT / "configs/mock.toml"), output_dir=Path(directory))
            backend = RecordingReference()
            output, passed = verify_backend(config, backend, samples=2, repeats=3)
            report = json.loads((output / "verification.json").read_text(encoding="utf-8"))
            self.assertTrue(passed)
            self.assertEqual(backend.seeds, [42, 42, 42, 43, 43, 43])
            self.assertEqual(len(report["requests"]), 6)
            self.assertEqual(report["verification_repeats"], 3)

    def test_matching_pairs_cannot_hide_repeat_instability(self) -> None:
        class DriftingReference(ReferenceFixture):
            def __init__(self) -> None:
                super().__init__()
                self.offset = 0

            def generate(self, request: GenerationRequest) -> GenerationResult:
                self.offset += 1
                return replace(super().generate(request), token_ids=self.reference_tokens(request))

            def reference_tokens(self, request: GenerationRequest) -> tuple[int, ...]:
                return tuple(token + self.offset for token in range(request.max_new_tokens))

        with tempfile.TemporaryDirectory() as directory:
            config = replace(load_config(ROOT / "configs/mock.toml"), output_dir=Path(directory))
            output, passed = verify_backend(config, DriftingReference(), samples=1, repeats=3)
            rows = json.loads((output / "verification.json").read_text(encoding="utf-8"))[
                "requests"
            ]
            self.assertFalse(passed)
            self.assertTrue(all(row["match"] for row in rows))
            self.assertFalse(rows[1]["actual_repeat_match"])
            self.assertFalse(rows[1]["reference_repeat_match"])

    def test_invalid_repeat_count_closes_backend(self) -> None:
        backend = ReferenceFixture()
        with self.assertRaises(ValueError):
            verify_backend(load_config(ROOT / "configs/mock.toml"), backend, repeats=0)
        self.assertTrue(backend.closed)

    def test_report_persists_match_and_mismatch_and_closes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(load_config(ROOT / "configs/mock.toml"), output_dir=Path(directory))
            for match in (True, False):
                backend = ReferenceFixture(match)
                path, passed = verify_backend(config, backend, samples=2)
                report = json.loads((path / "verification.json").read_text(encoding="utf-8"))
                self.assertEqual(passed, match)
                self.assertEqual(report["passed"], match)
                self.assertEqual(len(report["requests"]), 2)
                first = report["requests"][0]
                self.assertEqual(first["first_mismatch_index"], None if match else 0)
                self.assertEqual(first["actual_length"], config.max_new_tokens)
                self.assertEqual(first["reference_length"], config.max_new_tokens if match else 1)
                self.assertEqual(report["status"], "completed")
                self.assertTrue(backend.closed)

    def test_failure_is_not_reported_as_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(load_config(ROOT / "configs/mock.toml"), output_dir=Path(directory))
            backend = ReferenceFixture(fail=True)
            with self.assertRaisesRegex(RuntimeError, "reference failed"):
                verify_backend(config, backend)
            report = json.loads(
                next(Path(directory).glob("*/verification.json")).read_text(encoding="utf-8")
            )
            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "failed")
            self.assertTrue(backend.closed)

    def test_late_and_length_only_mismatches_remain_failures(self) -> None:
        class LateDifference(ReferenceFixture):
            def __init__(self, position: int, truncate: bool) -> None:
                super().__init__()
                self.position = position
                self.truncate = truncate

            def reference_tokens(self, request: GenerationRequest) -> tuple[int, ...]:
                tokens = list(super().reference_tokens(request))
                if self.truncate:
                    return tuple(tokens[: self.position])
                tokens[self.position] = 999
                return tuple(tokens)

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                load_config(ROOT / "configs/mock.toml"),
                output_dir=Path(directory),
                max_new_tokens=128,
            )
            for position, truncate in ((99, False), (127, False), (99, True)):
                with self.subTest(position=position, truncate=truncate):
                    output, passed = verify_backend(config, LateDifference(position, truncate), 1)
                    report = json.loads((output / "verification.json").read_text(encoding="utf-8"))
                    self.assertFalse(passed)
                    self.assertEqual(report["requests"][0]["first_mismatch_index"], position)

    def test_cli_import_does_not_load_torch_or_transformers(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import ainference_optim.cli; "
                "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class TransformersConfigTests(unittest.TestCase):
    def test_presets_and_invalid_fields(self) -> None:
        source = (ROOT / "configs/4090_smoke.toml").read_text(encoding="utf-8")
        self.assertIsNotNone(load_config(ROOT / "configs/4090_baseline.toml").transformers)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            for text in (
                source.replace('dtype = "bfloat16"', 'dtype = "int4"'),
                source.replace('device = "cuda:0"', 'device = "auto"'),
                source.replace('device = "cuda:0"', 'device = "cpu"'),
                source.replace('attention = "sdpa"', 'attention = "bad"'),
                source.replace("stop_on_eos = true", 'stop_on_eos = "true"'),
                source.replace("max_input_tokens = 2048", "max_input_tokens = 0"),
                source.replace("chat_template = true", "chat_templat = true"),
                source.replace('attention = "sdpa"', 'attention = "sdpa"\nsdpa_backend = "flash"'),
                source.replace('attention = "sdpa"', 'attention = "eager"\nsdpa_backend = "math"'),
                source.replace('attention = "sdpa"', 'attention = "sdpa"\ndeterministic = "true"'),
            ):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_config(path)

    def test_deterministic_preset_resolves_local_model_and_preserves_bf16(self) -> None:
        config = load_config(ROOT / "configs/4090_deterministic.toml").transformers
        self.assertTrue(config.deterministic)
        self.assertEqual(config.sdpa_backend, "math")
        self.assertEqual(config.dtype, "bfloat16")
        self.assertTrue(config.local_files_only)
        self.assertEqual(config.model, str(ROOT / "models/Qwen2.5-1.5B-Instruct"))

    def test_local_model_path_resolves_against_config(self) -> None:
        source = (ROOT / "configs/4090_smoke.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                source.replace('model = "Qwen/Qwen2.5-1.5B-Instruct"', 'model = "./model"'),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.transformers.model, str(Path(directory) / "model"))


if __name__ == "__main__":
    unittest.main()
