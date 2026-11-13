"""可选真实库集成测试：随机小型 Qwen2 与本地分词器，不下载权重、不需要 GPU。"""

import importlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from ainference_optim.backends.transformers import (
    TransformersBackend,
    _configure_fp32,
    _prepare_cublas_workspace,
)
from ainference_optim.cli import main
from ainference_optim.config import TransformersConfig
from ainference_optim.domain import GenerationRequest, Prompt
from ainference_optim.metrics import measure

HAS_INFERENCE = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "tokenizers")
)


class WorkspaceTests(unittest.TestCase):
    def fake_torch(self, initialized: bool) -> ModuleType:
        module = ModuleType("test_torch")
        module.cuda = SimpleNamespace(is_initialized=lambda: initialized)
        return module

    def test_workspace_defaults_before_init_and_preserves_valid_value(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            _prepare_cublas_workspace(self.fake_torch(False))
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            _prepare_cublas_workspace(self.fake_torch(True))
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":16:8")

    def test_workspace_rejects_late_or_invalid_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            _prepare_cublas_workspace(self.fake_torch(True))
        with (
            patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": "bad"}),
            self.assertRaises(ValueError),
        ):
            _prepare_cublas_workspace(self.fake_torch(False))


@unittest.skipUnless(HAS_INFERENCE, "安装 .[inference] 后运行真实库的 CPU 集成测试")
class TransformersIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name)
        torch = importlib.import_module("torch")
        hf = importlib.import_module("transformers")
        tokenizers = importlib.import_module("tokenizers")
        torch.set_num_threads(1)
        torch.manual_seed(17)
        vocabulary = {
            "[PAD]": 0,
            "[BOS]": 1,
            "[EOS]": 2,
            "[UNK]": 3,
            "hello": 4,
            "world": 5,
            "user": 6,
            "assistant": 7,
            "answer": 8,
        }
        raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]"))
        raw.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        tokenizer = hf.PreTrainedTokenizerFast(
            tokenizer_object=raw,
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{message['role']}} {{message['content']}} "
            "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
        )
        tokenizer.save_pretrained(cls.path)
        model = hf.Qwen2ForCausalLM(
            hf.Qwen2Config(
                vocab_size=9,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=256,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                tie_word_embeddings=False,
            )
        )
        model.save_pretrained(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def setUp(self) -> None:
        self.config = TransformersConfig(
            model=str(self.path),
            device="cpu",
            dtype="float32",
            local_files_only=True,
            stop_on_eos=False,
        )
        self.backend = TransformersBackend(self.config)
        self.addCleanup(self.backend.close)
        self.request = GenerationRequest(Prompt("local", "chat", "hello world"), 8, 42)

    def test_cached_greedy_matches_reference_for_chat_and_plain_inputs(self) -> None:
        result = self.backend.generate(self.request)
        self.assertEqual(result.token_ids, self.backend.reference_tokens(self.request))
        self.assertEqual(len(result.token_ids), 8)
        self.assertEqual(result.input_tokens, 4)
        self.assertEqual(len(result.token_times_s), 8)
        self.assertFalse(self.backend.info.synthetic)
        self.assertIsNone(measure(result).peak_memory_bytes)
        plain = TransformersBackend(replace(self.config, chat_template=False, attention="eager"))
        try:
            self.assertEqual(
                plain.generate(self.request).token_ids, plain.reference_tokens(self.request)
            )
        finally:
            plain.close()

    def test_forward_only_receives_full_prompt_once(self) -> None:
        lengths = []

        def observe(module, args, kwargs):
            lengths.append(int(kwargs["input_ids"].shape[-1]))

        hook = self.backend._model.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            self.backend.generate(self.request)
        finally:
            hook.remove()
        self.assertEqual(lengths, [4] + [1] * 7)

    def test_long_fixed_output_matches_reference(self) -> None:
        request = replace(self.request, max_new_tokens=128)
        result = self.backend.generate(request)
        self.assertEqual(len(result.token_ids), 128)
        self.assertEqual(result.token_ids, self.backend.reference_tokens(request))

    def test_fixed_prefix_probes_are_repeatable_and_do_not_change_generation(self) -> None:
        expected = self.backend.generate(self.request).token_ids
        for prefix in ((), expected[:4]):
            observations = self.backend.probe_logits(self.request, prefix, repeats=3)
            self.assertEqual(len(observations), 6)
            for path in ("full_context_no_cache", "cache_replay"):
                rows = [item for item in observations if item.path == path]
                self.assertEqual(len({item.digest for item in rows}), 1)
                self.assertTrue(all(item.finite and item.equal_to_first for item in rows))
                self.assertTrue(all(item.max_abs_difference == 0 for item in rows))
                self.assertTrue(all(item.top_margin >= 0 for item in rows))
        self.assertEqual(self.backend.generate(self.request).token_ids, expected)

    def test_probe_rejects_invalid_prefix_and_training_mode(self) -> None:
        for prefix, repeats in (((9,), 3), ((-1,), 3), ((1,) * 8, 3), ((), 1)):
            with self.assertRaises(ValueError):
                self.backend.probe_logits(self.request, prefix, repeats)
        self.backend._model.train()
        with self.assertRaisesRegex(RuntimeError, "eval"):
            self.backend.probe_logits(self.request, (), 3)

    def test_deterministic_math_repeats_and_restores_runtime_flags(self) -> None:
        torch = importlib.import_module("torch")
        previous = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
        )
        stable = TransformersBackend(replace(self.config, deterministic=True, sdpa_backend="math"))
        try:
            with stable._execution_context():
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
                self.assertFalse(torch.backends.cudnn.benchmark)
                self.assertTrue(torch.backends.cudnn.deterministic)
                self.assertTrue(torch.backends.cuda.math_sdp_enabled())
                self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
                self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
            request = replace(self.request, max_new_tokens=128)
            actual = stable.generate(request).token_ids
            self.assertEqual(actual, stable.reference_tokens(request))
            self.assertEqual(actual, stable.generate(request).token_ids)
            with (
                patch.object(stable._model, "generate", side_effect=RuntimeError("injected")),
                self.assertRaisesRegex(RuntimeError, "injected"),
            ):
                stable.reference_tokens(request)
            current = (
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
            )
            self.assertEqual(current, previous)
            self.assertEqual(stable.info.metadata["sdpa_backend"], "math")
        finally:
            stable.close()

    def test_forward_positions_and_last_logits_match_reference(self) -> None:
        observations = []

        def observe(module, args, kwargs):
            observations.append(
                {
                    "ids": kwargs["input_ids"].tolist(),
                    "mask": kwargs["attention_mask"].tolist(),
                    "position": kwargs["position_ids"].tolist(),
                    "cache_position": kwargs["cache_position"].tolist(),
                    "logits_to_keep": kwargs["logits_to_keep"],
                    "cache_type": type(kwargs["past_key_values"]).__name__,
                }
            )

        hook = self.backend._model.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            self.backend.generate(self.request)
            actual = observations.copy()
            observations.clear()
            self.backend.reference_tokens(self.request)
        finally:
            hook.remove()
        self.assertEqual(actual, observations)
        self.assertTrue(all(item["logits_to_keep"] == 1 for item in actual))

    def test_new_tf32_api_does_not_emit_deprecation_warning(self) -> None:
        import warnings

        torch = importlib.import_module("torch")
        matmul = torch.backends.cuda.matmul
        cudnn = torch.backends.cudnn
        previous = (matmul.fp32_precision, cudnn.fp32_precision)
        try:
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                _configure_fp32(torch)
            self.assertEqual(matmul.fp32_precision, "ieee")
            self.assertEqual(cudnn.fp32_precision, "ieee")
            self.assertFalse(any("allow_tf32" in str(warning.message) for warning in seen))
        finally:
            matmul.fp32_precision, cudnn.fp32_precision = previous

    def test_eos_stops_and_is_included_in_output(self) -> None:
        first = self.backend.generate(self.request).token_ids[0]
        self.backend._eos_ids = [first, 2]
        self.backend._config = replace(self.config, stop_on_eos=True)
        result = self.backend.generate(self.request)
        self.assertEqual(result.token_ids, (first,))
        self.assertEqual(result.token_ids, self.backend.reference_tokens(self.request))
        self.assertIsNone(measure(result).tpot_s)

    def test_reference_does_not_inherit_sampling_penalties_or_eos_stop(self) -> None:
        actual = self.backend.generate(self.request).token_ids
        defaults = self.backend._model.generation_config
        defaults.do_sample = True
        defaults.repetition_penalty = 2.0
        defaults.forced_bos_token_id = (actual[0] + 1) % 9
        defaults.eos_token_id = actual[0]
        self.assertEqual(self.backend.reference_tokens(self.request), actual)

    def test_input_and_context_overflow_fail_without_truncation(self) -> None:
        self.backend._config = replace(self.config, max_input_tokens=2)
        with self.assertRaisesRegex(ValueError, "不自动截断"):
            self.backend.generate(self.request)
        self.backend._config = self.config
        with self.assertRaisesRegex(ValueError, "上下文上限"):
            self.backend.generate(replace(self.request, max_new_tokens=256))

    def test_close_is_idempotent_and_prevents_reuse(self) -> None:
        self.backend.close()
        self.backend.close()
        with self.assertRaisesRegex(RuntimeError, "已关闭"):
            self.backend.generate(self.request)

    def test_cuda_request_does_not_fall_back_to_cpu(self) -> None:
        torch = importlib.import_module("torch")
        unavailable = torch.cuda.device_count()
        with self.assertRaisesRegex(RuntimeError, "CUDA 设备不可用"):
            TransformersBackend(replace(self.config, device=f"cuda:{unavailable}"))

    def test_cli_verification_and_benchmark_persist_real_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preset = Path(__file__).resolve().parents[1] / "configs/4090_smoke.toml"
            text = preset.read_text(encoding="utf-8")
            substitutions = {
                'model = "Qwen/Qwen2.5-1.5B-Instruct"': f'model = "{self.path.as_posix()}"',
                'dataset = "../data/smoke.jsonl"': 'dataset = "./data.jsonl"',
                'output_dir = "../outputs"': 'output_dir = "./outputs"',
                'device = "cuda:0"': 'device = "cpu"',
                'dtype = "bfloat16"': 'dtype = "float32"',
                "stop_on_eos = true": "stop_on_eos = false",
                "local_files_only = false": "local_files_only = true",
                "max_new_tokens = 32": "max_new_tokens = 8",
            }
            for before, after in substitutions.items():
                text = text.replace(before, after)
            config_path = root / "config.toml"
            config_path.write_text(text, encoding="utf-8")
            (root / "data.jsonl").write_text(
                json.dumps({"id": "local", "scenario": "chat", "prompt": "hello world"}),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(["verify", "--config", str(config_path), "--repeats", "2"]), 0
                )
                self.assertEqual(main(["run", "--config", str(config_path)]), 0)
            output = next((root / "outputs").glob("*/manifest.json")).parent
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            row = json.loads((output / "requests.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertFalse(row["synthetic"])
            self.assertEqual(len(row["result"]["token_ids"]), 8)
            self.assertEqual(row["metrics"]["input_tokens"], 4)
            verification = next((root / "outputs").glob("*/verification.json"))
            report = json.loads(verification.read_text(encoding="utf-8"))
            self.assertEqual(report["verification_repeats"], 2)
            self.assertEqual(len(report["requests"]), 2)
            # 构造已知位置的分叉，覆盖 CLI 从原快照到真实库探针的完整装配。
            report["requests"][0]["match"] = False
            report["requests"][0]["reference"][3] = (report["requests"][0]["actual"][3] + 1) % 9
            verification.write_text(json.dumps(report), encoding="utf-8")
            with redirect_stdout(io.StringIO()) as display:
                self.assertEqual(
                    main(
                        [
                            "diagnose",
                            "--config",
                            str(config_path),
                            "--report",
                            str(verification),
                            "--dtype",
                            "float32",
                            "--repeats",
                            "2",
                        ]
                    ),
                    0,
                )
            self.assertIn("equal_to_first=True", display.getvalue())
            probe_file = next((root / "outputs").glob("*/diagnostics.json"))
            diagnostic = json.loads(probe_file.read_text(encoding="utf-8"))
            self.assertEqual(diagnostic["probe_position_0based"], 3)
            self.assertEqual(len(diagnostic["observations"]), 4)


if __name__ == "__main__":
    unittest.main()
