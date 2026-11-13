"""单设备、batch=1 的 Transformers 贪心基线；第三方张量仅在此模块使用。"""

import gc
import hashlib
import importlib
import inspect
import os
import subprocess
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from time import perf_counter
from types import ModuleType
from typing import Any

from ainference_optim.config import TransformersConfig
from ainference_optim.domain import BackendInfo, GenerationRequest, GenerationResult, LogitProbe


class _TokenClock:
    """只接收 token ID 的同步 streamer，跳过 HF 首次发送的输入提示。"""

    def __init__(self, started: float, prompt_ids: tuple[int, ...]) -> None:
        self.started = started
        self.prompt_ids = prompt_ids
        self.received_prompt = False
        self.ended = False
        self.token_ids: list[int] = []
        self.times: list[float] = []

    def put(self, value: Any) -> None:
        """在 token 已到达 CPU 后记录可用时间；同批释放共享时间戳。"""
        tokens = tuple(int(token) for token in value.reshape(-1).tolist())
        if not self.received_prompt:
            if tokens != self.prompt_ids:
                raise RuntimeError("HF streamer 首次发送的提示与实际输入不一致")
            self.received_prompt = True
            return
        self.token_ids.extend(tokens)
        self.times.extend([perf_counter() - self.started] * len(tokens))

    def end(self) -> None:
        """记录 HF 已完成发送；不做额外分词或文本流式解码。"""
        self.ended = True


def _configure_fp32(torch: ModuleType) -> None:
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"


def _prepare_cublas_workspace(torch: ModuleType) -> None:
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is None:
        if torch.cuda.is_initialized():
            raise RuntimeError("设置 CUBLAS_WORKSPACE_CONFIG 前 CUDA 已初始化，请用全新进程运行")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif workspace not in {":4096:8", ":16:8"}:
        raise ValueError("确定性模式要求 CUBLAS_WORKSPACE_CONFIG 为 :4096:8 或 :16:8")


@contextmanager
def _deterministic_algorithms(torch: ModuleType) -> Iterator[None]:
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.benchmark = previous[2]
        torch.backends.cudnn.deterministic = previous[3]


def _driver_version() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


class TransformersBackend:
    """加载已有 causal LM，显式管理 KV Cache 和 host 可用时间。

    当前只提供无 logits processor 的贪心生成，不支持采样、量化或 offload。
    Any 限于无统一静态接口的第三方模型、分词器和张量边界。
    """

    def __init__(self, config: TransformersConfig) -> None:
        """检查设备并加载模型；CUDA 不可用时直接失败，不回退 CPU。

        Args:
            config: 经配置层校验的模型与设备参数。

        Raises:
            RuntimeError: 缺少依赖或指定设备不可用。
            ValueError: 模型不是受支持的 decoder-only 模型或缺少聊天模板。
        """
        self._config = config
        try:
            self._torch = importlib.import_module("torch")
            self._hf = importlib.import_module("transformers")
        except ImportError as error:
            raise RuntimeError(
                "缺少推理依赖，请按 docs/remote_4090.md 安装 .[inference]"
            ) from error
        torch = self._torch
        self._device = torch.device(config.device)
        if config.deterministic and self._device.type == "cuda":
            # 必须早于设备检查、模型加载等可能建立 CUDA/cuBLAS 状态的操作。
            _prepare_cublas_workspace(torch)
        if self._device.type == "cuda":
            if not torch.cuda.is_available() or self._device.index >= torch.cuda.device_count():
                raise RuntimeError(f"指定 CUDA 设备不可用：{config.device}")
            with torch.cuda.device(self._device):
                if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                    raise RuntimeError("当前设备不支持 BF16；请显式修改 dtype")
            _configure_fp32(torch)
        options = {
            "revision": config.revision,
            "local_files_only": config.local_files_only,
            "trust_remote_code": False,
        }
        model_config = self._hf.AutoConfig.from_pretrained(config.model, **options)
        if model_config.is_encoder_decoder:
            raise ValueError("当前基线仅支持 decoder-only causal LM")
        # 先解析 commit，再让权重和分词器使用同一版本，避免 main 分支更新造成混用。
        options["revision"] = getattr(model_config, "_commit_hash", None) or config.revision
        self._tokenizer: Any = self._hf.AutoTokenizer.from_pretrained(config.model, **options)
        if config.chat_template and not self._tokenizer.chat_template:
            raise ValueError("模型没有聊天模板，请显式关闭 chat_template 或选择 instruct 模型")
        self._model: Any = (
            self._hf.AutoModelForCausalLM.from_pretrained(
                config.model,
                config=model_config,
                dtype=getattr(torch, config.dtype),
                attn_implementation=config.attention,
                use_safetensors=True,
                **options,
            )
            .to(self._device)
            .eval()
        )
        self._closed = False
        # generate 会在受支持模型上只投影末位 hidden state，手写循环须保持相同形状。
        self._forward_options = (
            {"logits_to_keep": 1}
            if "logits_to_keep" in inspect.signature(self._model.forward).parameters
            else {}
        )
        eos = self._model.generation_config.eos_token_id
        if eos is None:
            eos = self._tokenizer.eos_token_id
        self._eos_ids: list[int] = (
            [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        )
        self._context_limit = getattr(model_config, "max_position_embeddings", None)
        metadata = {
            "torch": str(torch.__version__),
            "transformers": str(self._hf.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "requested_revision": config.revision,
            "resolved_revision": str(options["revision"]),
            "dtype": config.dtype,
            "attention": config.attention,
            "strategy": "raw_greedy_no_logits_processors",
            "stop_on_eos": str(config.stop_on_eos),
            "chat_template": str(config.chat_template),
            "model_type": str(model_config.model_type),
            "validation_status": "requires_reference_verification_on_target_device",
            "decode_path": (
                "hf_generate_streamer_v1"
                if config.generation_path == "hf_generate"
                else "prepared_dynamic_cache_v2"
            ),
            "generation_path": config.generation_path,
            "token_observation": "host_available_each_token_v1",
            "input_preparation": "model.prepare_inputs_for_generation",
            "logits_to_keep": str(self._forward_options.get("logits_to_keep", "model_default")),
            "fp32_matmul_precision": "ieee" if self._device.type == "cuda" else "unchanged",
            "deterministic_requested": str(config.deterministic),
            "deterministic_algorithms": str(
                config.deterministic or torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_warn_only": str(
                False
                if config.deterministic
                else torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "sdpa_backend": config.sdpa_backend,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
        }
        if self._device.type == "cuda":
            properties = torch.cuda.get_device_properties(self._device)
            metadata.update(
                {
                    "gpu_name": properties.name,
                    "total_memory_bytes": str(properties.total_memory),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "driver_version": _driver_version(),
                }
            )
        self._info = BackendInfo(
            name="transformers",
            model=config.model,
            device=config.device,
            synthetic=False,
            timing_notes=(
                "host perf_counter；含聊天模板、分词、输入传输及最终文本解码；"
                "逐 token 到达 CPU 后计时（manual 使用 item，hf_generate 使用 streamer）；"
                "包含同步与回调开销，不代表无观测的设备极限速度；"
                "请求前后 CUDA 同步；不含模型加载；EOS 计入输出 token；"
                "peak_memory_bytes 为请求期间 max_memory_allocated，包含常驻权重"
            ),
            metadata=metadata,
        )

    @property
    def info(self) -> BackendInfo:
        """返回真实执行身份；synthetic=False 不表示设备性能或等价性已经验收。"""
        return self._info

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("后端已关闭，不能再次生成")

    @contextmanager
    def _execution_context(self) -> Iterator[None]:
        with ExitStack() as stack:
            stack.enter_context(self._torch.inference_mode())
            if self._config.deterministic:
                stack.enter_context(_deterministic_algorithms(self._torch))
            if self._config.sdpa_backend == "math":
                attention = importlib.import_module("torch.nn.attention")
                stack.enter_context(attention.sdpa_kernel(attention.SDPBackend.MATH))
            yield

    def _synchronize(self) -> None:
        if self._device.type == "cuda":
            self._torch.cuda.synchronize(self._device)

    def _prepare(self, request: GenerationRequest) -> dict[str, Any]:
        if request.max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须为正整数")
        if self._config.chat_template:
            encoded = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": request.prompt.text}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            encoded = self._tokenizer(request.prompt.text, return_tensors="pt")
        count = int(encoded["input_ids"].shape[-1])
        if count == 0 or count > self._config.max_input_tokens:
            raise ValueError(
                f"实际输入 {count} token，合法范围为 1–{self._config.max_input_tokens}；不自动截断"
            )
        if self._context_limit and count + request.max_new_tokens > self._context_limit:
            raise ValueError("输入与最大输出总长度超过模型原生上下文上限")
        return {name: encoded[name].to(self._device) for name in ("input_ids", "attention_mask")}

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """执行 KV Cache 贪心解码，记录每个输出 token 的 host 可用时间。

        Args:
            request: 单条输入、输出上限和种子；贪心路径不进行随机采样。

        Returns:
            包含真实 token ID、时间线及设备峰值内存的生成结果。

        Raises:
            ValueError: 输入长度不合法或超过模型上下文上限。
            RuntimeError: 后端已关闭、设备失败或模型不返回 KV Cache。
        """
        self._ensure_open()
        if self._config.generation_path == "hf_generate":
            return self._generate_native(request)
        torch = self._torch
        torch.manual_seed(request.seed)
        self._synchronize()
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        started = perf_counter()
        token_ids: list[int] = []
        times: list[float] = []
        with self._execution_context():
            inputs = self._prepare(request)
            input_tokens = int(inputs["input_ids"].shape[-1])
            current = inputs["input_ids"]
            attention = inputs["attention_mask"]
            cache = self._hf.DynamicCache(config=self._model.config)
            cache_position = torch.arange(input_tokens, device=self._device)
            for index in range(request.max_new_tokens):
                # 与参考路径共用输入规范化，显式对齐位置、cache 和张量 stride。
                model_inputs = self._model.prepare_inputs_for_generation(
                    current,
                    attention_mask=attention,
                    past_key_values=cache,
                    cache_position=cache_position,
                    use_cache=True,
                    **self._forward_options,
                )
                output = self._model(**model_inputs, return_dict=True)
                next_token = output.logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
                token = int(next_token.item())
                times.append(perf_counter() - started)
                token_ids.append(token)
                cache = output.past_key_values
                # 不保留上一轮 logits，避免下一轮 forward 与旧 logits 同时占显存。
                del output
                if self._config.stop_on_eos and token in self._eos_ids:
                    break
                if index + 1 < request.max_new_tokens:
                    if cache is None:
                        raise RuntimeError("模型没有返回 KV Cache，禁止退化为全序列重算")
                    current = next_token
                    attention = torch.cat((attention, attention.new_ones((1, 1))), dim=-1)
                    cache_position = cache_position[-1:] + 1
            text = str(self._tokenizer.decode(token_ids, skip_special_tokens=True))
            self._synchronize()
            elapsed = perf_counter() - started
            memory = (
                int(torch.cuda.max_memory_allocated(self._device))
                if self._device.type == "cuda"
                else None
            )
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            token_times_s=tuple(times),
            elapsed_s=elapsed,
            peak_memory_bytes=memory,
            token_ids=tuple(token_ids),
        )

    def _native_forward(
        self,
        request: GenerationRequest,
        inputs: dict[str, Any],
        streamer: _TokenClock | None = None,
    ) -> Any:
        generation = self._hf.GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            cache_implementation="dynamic",
            eos_token_id=(self._eos_ids or None) if self._config.stop_on_eos else None,
            pad_token_id=self._tokenizer.pad_token_id,
            bos_token_id=self._tokenizer.bos_token_id,
        )
        # 显式覆盖 EOS，避免 HF 将模型默认停止条件重新合并进来。
        return self._model.generate(
            **inputs,
            generation_config=generation,
            use_model_defaults=False,
            eos_token_id=(self._eos_ids or None) if self._config.stop_on_eos else None,
            streamer=streamer,
            **self._forward_options,
        )

    def _generate_native(self, request: GenerationRequest) -> GenerationResult:
        self._torch.manual_seed(request.seed)
        self._synchronize()
        if self._device.type == "cuda":
            self._torch.cuda.reset_peak_memory_stats(self._device)
        started = perf_counter()
        with self._execution_context():
            inputs = self._prepare(request)
            prompt_ids = tuple(int(token) for token in inputs["input_ids"][0].tolist())
            clock = _TokenClock(started, prompt_ids)
            result = self._native_forward(request, inputs, clock)
            ids = tuple(int(token) for token in result[0, len(prompt_ids) :].tolist())
            if not clock.ended or tuple(clock.token_ids) != ids:
                raise RuntimeError("HF streamer 时间线与最终输出 token 不一致")
            text = str(self._tokenizer.decode(ids, skip_special_tokens=True))
            self._synchronize()
            elapsed = perf_counter() - started
            memory = (
                int(self._torch.cuda.max_memory_allocated(self._device))
                if self._device.type == "cuda"
                else None
            )
        return GenerationResult(text, len(prompt_ids), tuple(clock.times), elapsed, memory, ids)

    def reference_tokens(self, request: GenerationRequest) -> tuple[int, ...]:
        """用独立 GenerationConfig 调用库内 generate，排除模型默认采样与惩罚参数。"""
        self._ensure_open()
        self._torch.manual_seed(request.seed)
        with self._execution_context():
            inputs = self._prepare(request)
            length = int(inputs["input_ids"].shape[-1])
            result = self._native_forward(request, inputs)
            return tuple(int(token) for token in result[0, length:].tolist())

    def _probe_forward(self, inputs: dict[str, Any], prefix: Any, cached: bool) -> Any:
        torch = self._torch
        input_ids = inputs["input_ids"]
        attention = inputs["attention_mask"]
        if not cached:
            input_ids = torch.cat((input_ids, prefix), dim=-1)
            attention = torch.cat((attention, attention.new_ones(prefix.shape)), dim=-1)
            prepared = self._model.prepare_inputs_for_generation(
                input_ids,
                attention_mask=attention,
                use_cache=False,
                cache_position=torch.arange(input_ids.shape[-1], device=self._device),
                **self._forward_options,
            )
            return self._model(**prepared, return_dict=True).logits[:, -1, :]
        cache = self._hf.DynamicCache(config=self._model.config)
        position = torch.arange(input_ids.shape[-1], device=self._device)
        for index in range(prefix.shape[-1] + 1):
            prepared = self._model.prepare_inputs_for_generation(
                input_ids,
                attention_mask=attention,
                past_key_values=cache,
                cache_position=position,
                use_cache=True,
                **self._forward_options,
            )
            output = self._model(**prepared, return_dict=True)
            if index == prefix.shape[-1]:
                return output.logits[:, -1, :]
            cache = output.past_key_values
            if cache is None:
                raise RuntimeError("诊断模型未返回 KV Cache")
            input_ids = prefix[:, index : index + 1]
            attention = torch.cat((attention, attention.new_ones((1, 1))), dim=-1)
            position = position[-1:] + 1
        raise RuntimeError("前缀重放未产生 logits")

    def probe_logits(
        self, request: GenerationRequest, prefix: tuple[int, ...], repeats: int
    ) -> tuple[LogitProbe, ...]:
        """固定前缀，重复观测全上下文和逐步缓存计算的末位原始 logits。

        Args:
            request: 提供原始输入和种子，不进行随机采样。
            prefix: 已选定的共同输出前缀，重放时不根据本次 argmax 改动它。
            repeats: 每条计算路径重复次数，至少为 2。

        Returns:
            按路径和轮次排列的 top-5、非有限值、原始值哈希与重复差异。

        Raises:
            ValueError: 次数、token ID 或上下文长度不合法。
            RuntimeError: 模型处于训练模式或后端已经关闭。
        """
        self._ensure_open()
        if repeats < 2 or len(prefix) >= request.max_new_tokens:
            raise ValueError("探针要求 repeats≥2 且前缀短于输出上限")
        if any(
            type(token) is not int or not 0 <= token < self._model.config.vocab_size
            for token in prefix
        ):
            raise ValueError("诊断前缀包含非法 token ID")
        if any(module.training for module in self._model.modules()):
            raise RuntimeError("诊断要求模型及所有子模块处于 eval 模式")
        torch = self._torch
        observations: list[LogitProbe] = []
        with self._execution_context():
            inputs = self._prepare(request)
            tokens = torch.tensor([prefix], dtype=torch.long, device=self._device)
            for cached in (False, True):
                baseline = None
                for repeat in range(repeats):
                    torch.manual_seed(request.seed)
                    self._synchronize()
                    # 同一个输入 tensor 重复使用，去除文本生成分叉和分词差异。
                    logits = (
                        self._probe_forward(inputs, tokens, cached).detach().float().cpu().clone()
                    )
                    finite = bool(torch.isfinite(logits).all().item())
                    if baseline is None:
                        baseline = logits.clone()
                    values, ids = logits[0].topk(min(5, logits.shape[-1]))
                    scores = tuple(
                        float(value) if torch.isfinite(value).item() else None for value in values
                    )
                    difference = logits - baseline
                    valid_difference = finite and bool(torch.isfinite(baseline).all().item())
                    observations.append(
                        LogitProbe(
                            path="cache_replay" if cached else "full_context_no_cache",
                            repeat=repeat,
                            finite=finite,
                            nonfinite_count=int((~torch.isfinite(logits)).sum().item()),
                            top_token_ids=tuple(int(token) for token in ids.tolist()),
                            top_logits=scores,
                            top_margin=(
                                scores[0] - scores[1]
                                if len(scores) >= 2
                                and scores[0] is not None
                                and scores[1] is not None
                                else None
                            ),
                            digest=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
                            equal_to_first=bool(torch.equal(logits, baseline)),
                            max_abs_difference=(
                                float(difference.abs().max().item()) if valid_difference else None
                            ),
                            # topk 对并列项的顺序不保证与生成使用的 argmax 一致。
                            greedy_token_id=int(logits.argmax(dim=-1).item()),
                        )
                    )
        return tuple(observations)

    def close(self) -> None:
        """释放本实例持有的模型与分词器；重复关闭无副作用。"""
        if self._closed:
            return
        self._model = None
        self._tokenizer = None
        self._closed = True
        gc.collect()
        if self._device.type == "cuda":
            with self._torch.cuda.device(self._device):
                self._torch.cuda.empty_cache()
