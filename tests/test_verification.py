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
            ):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_config(path)

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
