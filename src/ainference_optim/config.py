"""加载严格 TOML 配置，所有相对路径均相对于配置文件。"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TransformersConfig:
    """真实后端参数；CPU 只用于显式的功能测试，不作 GPU 回退。"""

    model: str
    revision: str = "main"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attention: str = "sdpa"
    chat_template: bool = True
    stop_on_eos: bool = True
    local_files_only: bool = False
    max_input_tokens: int = 2048
    deterministic: bool = False
    sdpa_backend: str = "auto"


@dataclass(frozen=True)
class ExperimentConfig:
    """已完成验证和路径解析的实验配置。"""

    name: str
    dataset: Path
    output_dir: Path
    warmup: int
    repeats: int
    seed: int
    backend: str
    max_new_tokens: int
    transformers: TransformersConfig | None = None


def _keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{location} 字段不匹配：需要 {sorted(expected)}，得到 {sorted(value)}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def _integer(value: Any, name: str, minimum: int) -> int:
    # bool 是 int 的子类，但不应接受为实验次数或随机种子。
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} 必须是 >= {minimum} 的整数")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是布尔值")
    return value


def _transformers_config(raw: dict[str, Any], base: Path) -> TransformersConfig:
    allowed = {
        "name",
        "model",
        "revision",
        "device",
        "dtype",
        "attention",
        "chat_template",
        "stop_on_eos",
        "local_files_only",
        "max_input_tokens",
        "deterministic",
        "sdpa_backend",
    }
    if set(raw) - allowed or "model" not in raw:
        raise ValueError("transformers 后端必须指定 model，且不能有未知字段")
    model = _string(raw["model"], "backend.model")
    if model.startswith((".", "/", "~")) or re.match(r"^[A-Za-z]:", model):
        model = str((base / Path(model).expanduser()).resolve())
    device = _string(raw.get("device", "cuda:0"), "backend.device")
    dtype = _string(raw.get("dtype", "bfloat16"), "backend.dtype")
    attention = _string(raw.get("attention", "sdpa"), "backend.attention")
    sdpa_backend = _string(raw.get("sdpa_backend", "auto"), "backend.sdpa_backend")
    if device != "cpu" and not re.fullmatch(r"cuda:[0-9]+", device):
        raise ValueError("device 仅支持 cpu 或 cuda:<非负索引>，不支持自动分配")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype 仅支持 float32、float16、bfloat16")
    if device == "cpu" and dtype != "float32":
        raise ValueError("CPU 功能测试必须使用 float32")
    if attention not in {"sdpa", "eager"}:
        raise ValueError("attention 仅支持 sdpa 或 eager")
    if sdpa_backend not in {"auto", "math"} or (attention != "sdpa" and sdpa_backend != "auto"):
        raise ValueError("sdpa_backend 仅支持 auto/math；math 要求 attention=sdpa")
    return TransformersConfig(
        model=model,
        revision=_string(raw.get("revision", "main"), "backend.revision"),
        device=device,
        dtype=dtype,
        attention=attention,
        chat_template=_boolean(raw.get("chat_template", True), "chat_template"),
        stop_on_eos=_boolean(raw.get("stop_on_eos", True), "stop_on_eos"),
        local_files_only=_boolean(raw.get("local_files_only", False), "local_files_only"),
        max_input_tokens=_integer(raw.get("max_input_tokens", 2048), "max_input_tokens", 1),
        deterministic=_boolean(raw.get("deterministic", False), "deterministic"),
        sdpa_backend=sdpa_backend,
    )


def load_config(path: Path) -> ExperimentConfig:
    """读取配置并拒绝未知字段，避免拼写错误被静默忽略。

    Args:
        path: TOML 配置文件路径。

    Returns:
        已验证的实验配置。

    Raises:
        ValueError: 配置结构或字段值不合法。
        OSError: 配置文件无法读取。
    """
    path = path.resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    _keys(raw, {"experiment", "backend", "generation"}, "root")
    for section in raw.values():
        if not isinstance(section, dict):
            raise ValueError("配置节必须是 TOML table")
    exp, backend, generation = raw["experiment"], raw["backend"], raw["generation"]
    _keys(exp, {"name", "dataset", "output_dir", "warmup", "repeats", "seed"}, "experiment")
    backend_name = _string(backend.get("name"), "backend.name")
    model_config = None
    if backend_name == "mock":
        _keys(backend, {"name"}, "backend")
    elif backend_name == "transformers":
        model_config = _transformers_config(backend, path.parent)
    else:
        raise ValueError(f"未知后端：{backend_name}")
    _keys(generation, {"max_new_tokens"}, "generation")
    name = _string(exp["name"], "name")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("name 只允许 1–64 位字母、数字、连字符和下划线，首位须为字母或数字")
    return ExperimentConfig(
        name=name,
        dataset=(path.parent / _string(exp["dataset"], "dataset")).resolve(),
        output_dir=(path.parent / _string(exp["output_dir"], "output_dir")).resolve(),
        warmup=_integer(exp["warmup"], "warmup", 0),
        repeats=_integer(exp["repeats"], "repeats", 1),
        seed=_integer(exp["seed"], "seed", 0),
        backend=backend_name,
        max_new_tokens=_integer(generation["max_new_tokens"], "max_new_tokens", 1),
        transformers=model_config,
    )
