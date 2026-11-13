# 工程与注释规范

GitHub 同步、分支与 AI 提交边界另见 [Git 工作流规范](git_workflow.md)。

## 设计边界

按变化原因拆模块：模型和设备变化只影响后端，指标口径变化只影响 metrics，
文件格式变化集中在 results，实验执行顺序集中在 runner。
避免通用 utils 大杂烩、全局可变状态、隐式设备选择和业务逻辑散落到 CLI。

依赖方向如下，禁止反向导入：

```text
domain ← backends
domain ← data
domain ← metrics
config + data + domain + metrics + results ← runner
config + data + domain + metrics + results ← verification
config + data + domain + results ← diagnostics
data + results ← preparation
domain + metrics + results ← comparison
backends + config + data + runner + verification + diagnostics + preparation + comparison ← cli
```

domain 仅使用标准库。runner 只认识 Backend 协议，不识别 CUDA、NPU、模型名称
或具体生成算法。后端通过 create_backend 显式装配，未知名称报错，禁止静默回退。
后端通过显式工厂注册，不引入插件框架、复杂继承树或依赖注入容器。

## Python 代码规范

- Python >= 3.11，UTF-8，4 空格缩进，100 字符行宽，由 Ruff 格式化。
- 模块、函数和变量用 snake_case，类型用 PascalCase，常量用 UPPER_SNAKE_CASE。
- 名称表达领域含义；时间以 `_s` 结尾、内存以 `_bytes` 结尾，不使用模糊的 time/cost。
- 所有函数参数和返回值写类型标注。领域对象优先使用不可变 dataclass。
- Any 仅限 JSON/TOML 以及第三方张量/模型等外部边界；完成解析后使用确定类型。
  第三方张量和模型的 Any 只能留在具体后端，不得流入领域对象或 runner。
- 公共 API 不返回含糊的多用途 dict；指标汇总与序列化边界允许 JSON 对象。
- 函数聚焦一个职责。需要同时理解设备、文件格式和调度的函数必须拆分。
- 核心模块不 print、不配置 logging；CLI 负责用户输出。
- 可预期输入错误用 ValueError，并给出字段、样本 ID 或行号等定位信息。
- 不吞异常、不以空结果表示失败，不捕获 Exception 后静默继续。
  runner 只为持久化失败状态捕获异常，随后原样抛出。
- 生命周期通过 try/finally 管理；后端 generate 失败后仍应释放资源。
- 不在模块导入时加载模型、初始化设备、联网或读取实验配置。

## 注释和文档字符串

项目维护说明、docstring 和注释统一使用中文，保留标准英文术语；标识符使用英文。

每个模块写一句说明其职责的 docstring。每个公共类、公共函数和协议方法必须写
docstring；复杂接口使用 Google 风格的 Args、Returns、Raises。
描述行为、单位、边界条件和必要副作用，不逐句翻译函数名。
私有的短小函数可以省略 docstring，但算法或异常约束仍需说明。

```python
def percentile(values: list[float], fraction: float) -> float | None:
    """使用线性插值计算分位数；空输入返回 None。

    Args:
        values: 以相同单位记录的有限数值。
        fraction: [0, 1] 范围内的分位比例。

    Returns:
        插值得到的分位值，单位与输入一致。

    Raises:
        ValueError: fraction 超出合法范围。
    """
```

行内注释只解释“为什么”、设备同步要求、公式依据或非直观边界。
禁止 `# 循环列表`、`# 返回结果` 等复述代码的注释。
复杂公式要说明数学含义及单位；引自论文的算法应补充可核验来源。
TODO 必须明确缺失行为和完成条件，禁止空泛的“以后优化”。
接口行为变更时同步更新 docstring、实验协议和测试，避免注释与实现分离。

## 配置与结果规范

实验参数进入 TOML，不散落在代码中。路径相对于配置文件解析。
配置拒绝未知字段，避免输入拼写错误后继续使用默认值。
新参数需同时更新配置模型、验证、示例和 manifest 序列化。

输出结构带 schema_version；修改字段含义或单位时升级版本并记录迁移方式。
JSON 不写 NaN/Infinity，无法定义或未测量的指标使用 null，不以 0 代替缺失值。
每条记录保留原始时间线，汇总不得成为唯一证据。

## 测试与质量门槛

行为变化需覆盖有实际风险的边界：配置错拼、重复样本、预热排除、计时公式、
不合法测量、失败记录和资源释放。测试不依赖模型下载、网络或 GPU。
真实后端后续增加独立的设备集成检查，不能用 mock 测试代替设备测试。

提交前执行 unittest、Ruff lint、Ruff format 检查和 mypy strict。
核心依赖保持最小；真实后端依赖应放在独立可选依赖组，确认版本兼容后固定环境。
