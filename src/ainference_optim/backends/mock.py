"""确定性模拟后端，仅验证框架流程，不执行模型推理。"""

from ainference_optim.domain import BackendInfo, GenerationRequest, GenerationResult


class MockBackend:
    """返回固定的合成 token 时间线，不使用 sleep 伪造真实性能。"""

    @property
    def info(self) -> BackendInfo:
        """声明所有测量均为合成值。"""
        return BackendInfo(
            name="mock",
            model="synthetic-no-model",
            device="none",
            synthetic=True,
            timing_notes="合成时间线：TTFT=0.02s，token 间隔=0.005s；不代表任何硬件性能",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """产生指定数量的合成 token，输入计数仅为字符数占位。"""
        times = tuple(0.02 + index * 0.005 for index in range(request.max_new_tokens))
        return GenerationResult(
            text=" ".join(f"mock_{index}" for index in range(request.max_new_tokens)),
            input_tokens=len(request.prompt.text),
            token_times_s=times,
            elapsed_s=times[-1] + 0.001,
        )

    def close(self) -> None:
        """模拟后端没有设备资源需要释放。"""
