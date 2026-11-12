"""后端装配入口；框架其余模块不导入具体后端。"""

from ainference_optim.backends.mock import MockBackend
from ainference_optim.config import TransformersConfig
from ainference_optim.domain import Backend


def create_backend(name: str, config: TransformersConfig | None = None) -> Backend:
    """创建已实现的后端；未知后端直接报错，不回退到模拟实现。"""
    if name == "mock":
        if config is not None:
            raise ValueError("mock 后端不接受真实模型参数")
        return MockBackend()
    if name == "transformers":
        if config is None:
            raise ValueError("transformers 后端缺少模型配置")
        from ainference_optim.backends.transformers import TransformersBackend

        return TransformersBackend(config)
    raise ValueError(f"后端尚未实现：{name}")
