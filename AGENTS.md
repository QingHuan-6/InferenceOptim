# 项目开发约定

在修改代码前阅读 `docs/engineering.md`。这些规范适用于整个仓库。
涉及 Git 操作时还必须阅读 `docs/git_workflow.md`。

- 保持 domain → 无外部依赖；metrics → domain；backends → domain；
  runner、verification → domain/config/data/metrics/results；cli → 应用入口的单向依赖。
- 框架层不得导入 torch、CUDA 或昇腾运行时；设备代码只放在具体后端。
- 新后端必须实现 Backend 协议，并准确标注 synthetic 与计时边界。
- 不得把模拟性能作为真实性能，不得默认宣称分布等价或加速收益。
- 修改生成/计时/投机验证前，先审阅固定版本的成熟实现并记录源码位置、计时边界与差异。
  付费 GPU 实验须有明确假设、最小样本、判定规则和预计成本；先排除源码与 CPU 测试能回答的问题。
  禁止将未证实的慢点当根因，或以整套基线反复试参数；流程见 docs/source_review.md。
- Python 公共接口使用完整类型标注和 Google 风格中文 docstring。
  行内注释解释原因、约束和边界，不复述代码；规则详见工程规范。
- 行为变化后运行 `python -m unittest discover -s tests -v`。
  有开发工具时同时运行 `ruff check .`、`ruff format --check .`、`mypy`。
- 结果目录不可覆盖；新增指标必须记录单位、定义、聚合口径和缺失值规则。
- 不提交模型权重、真实私有输入、密钥或实验输出。
- GitHub 使用私有仓库；本地开发、服务器执行，工作分支格式为 `ai/<type>-<topic>`。
- 提交格式为 `type(scope): 中文说明`，一次提交一个目的，检查和审阅后再提交。
- AI 仅明确路径暂存，禁止盲目 `git add .`；保护用户已有修改。
- 已授权开发任务可本地提交；推送、建远程仓库、PR、合并须有请求或既有明确授权。
- 不自行强推、硬重置、清理未跟踪文件、改写共享历史或修改全局 Git 配置。
- 不伪造作者身份、测试与实测结论；完整约定见 Git 工作流规范。
