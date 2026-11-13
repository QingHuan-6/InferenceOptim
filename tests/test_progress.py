"""验证预热与正式进度、失败计数，以及显示不改变请求测量。"""

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
from ainference_optim.domain import GenerationRequest, GenerationResult, RunProgress
from ainference_optim.runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]


class FailingBackend(MockBackend):
    def __init__(self) -> None:
        self.closed = False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise RuntimeError("injected failure")

    def close(self) -> None:
        self.closed = True


class ProgressTests(unittest.TestCase):
    def test_phase_totals_and_callbacks_leave_measurements_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                load_config(ROOT / "configs/mock.toml"),
                output_dir=Path(directory),
                warmup=2,
                repeats=2,
            )
            events: list[RunProgress] = []
            visible = run_experiment(config, MockBackend(), events.append)
            quiet = run_experiment(config, MockBackend())
            warmup = [event for event in events if event.phase == "warmup"]
            formal = [event for event in events if event.phase == "measurement"]
            self.assertEqual([event.completed for event in warmup], [0, 1, 2])
            self.assertEqual([event.completed for event in formal], list(range(7)))
            self.assertTrue(all(event.total == 6 for event in formal))
            self.assertEqual([event.repeat for event in formal], [None, 1, 1, 1, 2, 2, 2])
            self.assertEqual(
                (visible / "requests.jsonl").read_bytes(), (quiet / "requests.jsonl").read_bytes()
            )
            self.assertEqual(
                (visible / "summary.json").read_bytes(), (quiet / "summary.json").read_bytes()
            )

    def test_failure_is_not_counted_as_completed_and_closes_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                load_config(ROOT / "configs/mock.toml"), output_dir=Path(directory), warmup=0
            )
            events: list[RunProgress] = []
            backend = FailingBackend()
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                run_experiment(config, backend, events.append)
            self.assertTrue(backend.closed)
            self.assertEqual([event.completed for event in events], [0])
            path = next(Path(directory).glob("*/manifest.json"))
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual((manifest["status"], manifest["completed_requests"]), ("failed", 0))

    def test_cli_defaults_to_progress_and_quiet_disables_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                load_config(ROOT / "configs/mock.toml"),
                output_dir=Path(directory),
                warmup=1,
                repeats=1,
            )
            for quiet in (False, True):
                with (
                    patch("ainference_optim.cli.load_config", return_value=config),
                    redirect_stdout(io.StringIO()) as output,
                    redirect_stderr(io.StringIO()) as progress,
                ):
                    args = ["run", "--config", "unused.toml"] + (["--quiet"] if quiet else [])
                    self.assertEqual(main(args), 0)
                self.assertIn("结果目录", output.getvalue())
                if quiet:
                    self.assertEqual(progress.getvalue(), "")
                else:
                    self.assertIn("[预热] 1/1", progress.getvalue())
                    self.assertIn("[正式] 3/3 (100.0%)", progress.getvalue())
                    self.assertIn("轮次=1/1", progress.getvalue())


if __name__ == "__main__":
    unittest.main()
