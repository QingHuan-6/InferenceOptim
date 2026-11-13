"""单设备、batch=1 的 Transformers 贪心基线；第三方张量仅在此模块使用。"""

import gc
import importlib
import inspect
import subprocess
from time import perf_counter
from types import ModuleType
from typing import Any

from ainference_optim.config import TransformersConfig
from ainference_optim.domain import BackendInfo, GenerationRequest, GenerationResult


def _configure_fp32(torch: ModuleType) -> None:
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"


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
            "decode_path": "prepared_dynamic_cache_v2",
            "input_preparation": "model.prepare_inputs_for_generation",
            "logits_to_keep": str(self._forward_options.get("logits_to_keep", "model_default")),
            "fp32_matmul_precision": "ieee" if self._device.type == "cuda" else "unchanged",
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
                "每个 argmax token 经 item() 到达 CPU 后计时（包含隐式同步开销）；"
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
        torch = self._torch
        torch.manual_seed(request.seed)
        self._synchronize()
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        started = perf_counter()
        token_ids: list[int] = []
        times: list[float] = []
        with torch.inference_mode():
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

    def reference_tokens(self, request: GenerationRequest) -> tuple[int, ...]:
        """用独立 GenerationConfig 调用库内 generate，排除模型默认采样与惩罚参数。"""
        self._ensure_open()
        self._torch.manual_seed(request.seed)
        with self._torch.inference_mode():
            inputs = self._prepare(request)
            length = int(inputs["input_ids"].shape[-1])
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
            # EOS=None 在模型默认值合并时仍可能被补回，必须在最终 kwargs 中显式覆盖。
            result = self._model.generate(
                **inputs,
                generation_config=generation,
                use_model_defaults=False,
                eos_token_id=(self._eos_ids or None) if self._config.stop_on_eos else None,
                **self._forward_options,
            )
            return tuple(int(token) for token in result[0, length:].tolist())

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
