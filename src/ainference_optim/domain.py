"""定义硬件无关的请求、生成结果与后端契约。"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Prompt:
    """一条输入样本；scenario 仅表示任务类别，不代表质量已验证。"""

    id: str
    scenario: str
    text: str


@dataclass(frozen=True)
class GenerationRequest:
    """单次生成参数；seed 由 runner 按实际调用顺序分配。"""

    prompt: Prompt
    max_new_tokens: int
    seed: int


@dataclass(frozen=True)
class GenerationResult:
    """由后端测量的单次生成结果。

    token_times_s 是相对请求开始、每个输出 token 可用时的秒数。
    投机验证一次接受多个 token 时，可用时间可以相同。
    elapsed_s 包含请求内分词、输入传输及最终同步，排除模型加载。
    peak_memory_bytes 为该请求的设备峰值内存；不支持时填写 None。
    """

    text: str
    input_tokens: int
    token_times_s: tuple[float, ...]
    elapsed_s: float
    peak_memory_bytes: int | None = None
    token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class BackendInfo:
    """后端身份及测量声明，随每次实验永久保存。"""

    name: str
    model: str
    device: str
    synthetic: bool
    timing_notes: str
    metadata: dict[str, str] = field(default_factory=dict)


class Backend(Protocol):
    """可替换后端；实现方负责设备同步、随机种子和正确的计时。"""

    @property
    def info(self) -> BackendInfo:
        """返回可序列化的身份信息。"""
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """生成一次输出，并返回已完成设备同步的测量结果。"""
        ...

    def close(self) -> None:
        """释放资源；即使生成失败，runner 也会调用此方法。"""
        ...


@runtime_checkable
class ReferenceBackend(Backend, Protocol):
    """支持通过库内参考生成路径核对输出的后端。"""

    def reference_tokens(self, request: GenerationRequest) -> tuple[int, ...]:
        """返回相同生成设置下参考实现的输出 token，不计入性能实验。"""
        ...


@dataclass(frozen=True)
class LogitProbe:
    """同一前缀下的一次原始 logits 观测，不作为性能或质量指标。"""

    path: str
    repeat: int
    finite: bool
    nonfinite_count: int
    top_token_ids: tuple[int, ...]
    top_logits: tuple[float | None, ...]
    top_margin: float | None
    digest: str
    equal_to_first: bool
    max_abs_difference: float | None
    greedy_token_id: int


@runtime_checkable
class LogitProbeBackend(Backend, Protocol):
    """支持使用固定输出前缀探查原始 logits 的后端。"""

    def probe_logits(
        self, request: GenerationRequest, prefix: tuple[int, ...], repeats: int
    ) -> tuple[LogitProbe, ...]:
        """分别重复无缓存与缓存重放，不使用新生成的 token 更新前缀。"""
        ...
