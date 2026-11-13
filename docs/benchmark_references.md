# 解码性能评测的论文依据

本记录核对论文正文中的数据集、工作负载与指标，不以博客或开源框架现有默认配置替代论文。
系统论文调研日期为 2026-11-12，DeepSeek 指标补充日期为 2026-11-13。
首表论文发表于 SOSP、OSDI、ASPLOS；这些会议属于 CCF A 类，
分类入口见 [CCF 目录](https://www.ccf.org.cn/Academic_Evaluation/By_category/)，
可交叉查阅 [北大 CECA 的 CCF 分类对照](https://ceca.pku.edu.cn/research/topconf/index.htm)。

## 论文与原文定位

| 论文 | 会议 | 数据集及用途 | 指标与原文定位 |
| --- | --- | --- | --- |
| Efficient Memory Management for Large Language Model Serving with PagedAttention（vLLM） | SOSP 2023 | ShareGPT、Alpaca；分词后依据输入/输出长度构造请求，用 Poisson 到达过程测试负载 | 服务吞吐及 normalized latency；§6.1、Fig. 11–12；[论文](https://arxiv.org/pdf/2309.06180)、[正式发表信息](https://doi.org/10.1145/3600006.3613165) |
| DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | OSDI 2024 | ShareGPT 对话、HumanEval 代码补全、LongBench 摘要；采样请求并生成 Poisson 到达时间 | TTFT/TPOT 双 SLO、SLO attainment；§6.1、Table 1、Fig. 7；[论文](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) |
| Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve | OSDI 2024 | openchat_sharegpt4、arxiv_summarization；使用请求长度特征构造轨迹，多轮对话中的每轮作为请求 | TTFT 中位数、逐 token TBT 的 P99、满足延迟目标的 capacity；§2.4、§5、Table 2；[论文](https://www.usenix.org/system/files/osdi24-agrawal.pdf) |
| SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification | ASPLOS 2024 | Chatbot Instruction Prompts（CIP）、ChatGPT Prompts（CP）、WebQA、Alpaca、PIQA；提取真实提示/问题 | 平均每 token 延迟、投机验证推进效率；§6.1–6.2、Fig. 7、Table 2；[论文](https://arxiv.org/pdf/2305.09781)、[会议正式程序](https://www.asplos-conference.org/asplos2024/main-program/abstracts/index.html) |

## 从这些论文能得出的结论

测速度也需要明确工作负载，但不意味着每次都要计算任务准确率。
vLLM 的 normalized latency 是每个请求端到端时延除以输出长度后再取均值，
不能直接当作本项目排除首 token 后的 TPOT。
Sarathi-Serve 的 P99 TBT 是连续输出 token 间隔的尾部统计，
不能替换为“请求平均 TPOT 的 P99”。这些是不同统计对象。

投机解码中，接受率取决于实际提示、模型和生成分布；仅复现输入输出长度不足以
验证投机收益。SpecInfer 使用真实问题提示，生成上限设为 128 token，超出部分截断。
因此本项目后续投机实验应保留真实文本，不能仅使用重复字符构造全部研究输入。

系统吞吐实验还需要到达过程、请求率、并发、SLO 及持续时间；单请求 tokens/s
不能代替这些结果。Poisson 到达过程是上述部分系统论文的实验设置，不是数据集自带字段。
目前项目只提供串行单请求执行，尚不具备这些服务容量评测能力。

## 对本项目的建议

以下是结合论文和单张 4090 条件提出的计划，不是已实现能力或对原论文的完整复现。

1. 对话/指令先选一个明确版本的 ShareGPT 或 Alpaca，代码使用 HumanEval。
   先完成真实文本的串行基线，再测投机解码。
2. 长上下文扩展参考 LongBench 的摘要子任务，固定子集和上下文处理方法。
   不能把 LongBench v2、LongBench v1 或不同子任务混称为同一数据集。
3. 固定长度微基准和真实任务分开报告；公开输入/输出 token 长度分布，
   固定模型/分词器 revision、模板、停止条件、精度、输入集合与种子。
4. 性能和质量分别评测。HumanEval 提示可以用于性能负载，但提取 prompt 并测时延
   不等于标准 pass@k 评测；当前项目没有执行其测试代码或实现该评分。
5. JSON/工具调用需另外选择有依据的数据集与评分规则；不能用上述论文的数据集
   推导“JSON 场景已覆盖”。当前只有自带 JSON 提示用于流程检查。

## DeepSeek 解码指标补充

以下区分方法论文、模型技术报告和官方工程文档；不统一标为 CCF A 论文。
这些资料支持指标选择，不代表本项目复现了相应集群或收益。

| 资料 | 与指标有关的原文定位 | 本项目采用的边界 |
| --- | --- | --- |
| [DeepSeek-V2](https://arxiv.org/html/2405.04434v5)（2024） | §3.4，8×H800、实际服务长度分布；报告系统生成吞吐与 KV cache | 总吞吐不是单请求速度，跨模型的吞吐增益不是固定模型的算法加速比 |
| [DeepSeek-V3](https://arxiv.org/html/2412.19437v2)（2024） | §5.4.3，MTP 第二预测 token 接受率 85%–90%，TPS 为 1.8× | 接受率和真实速度同时记录；该小节未给出完整测量协议，不直接沿用其倍率 |
| [Insights into DeepSeek-V3](https://arxiv.org/html/2505.09343v1#S2.SS3)（2025） | §2.3，区分系统最大吞吐、单请求延迟，讨论 MTP 及计算通信重叠 | 单请求和服务并发分阶段测量 |
| [Native Sparse Attention](https://arxiv.org/html/2502.11089v1#S5)（2025） | §5、表4，算子计时与解码 KV 访存量；11.6× 为 64K 条件下访存量推算 | 算子收益、估计收益和整个模型实测分别标注 |
| [DeepSeek-V3.2](https://arxiv.org/html/2512.02556v1#S2.SS3)（2025） | §2.3、图3，H800 实际服务基准，按每 GPU 小时 2 美元折算 token 成本 | 若扩展成本指标，记录卡数、计费单价、时间边界及实际输出，不使用 API 价格冒充硬件成本 |
| [DSpark](https://arxiv.org/html/2607.05147v1)（2026） | §4.1、表1：温度1、数学/代码/聊天负载，每轮接受长度；§5.4、图7：总吞吐与单用户速度 | 默认接受长度含 bonus；本项目草稿接受率排除 bonus，每轮推进包含 bonus；接受长度不能替代真实计时 |
| [DeepInfer benchmark 文档](https://github.com/deepseek-ai/DeepInfer/blob/main/examples/README.md#benchmark) | 稳态 SPS、token/s/GPU、预热/取样窗口；decode 吞吐按 full-acceptance accounting 计算 | 该微基准计数假设草稿全接受；本项目正式有效速度只使用实际输出 |

DSpark 离线负载包括 GSM8K、MATH500、AIME25、MBPP、HumanEval、LiveCodeBench、
MT-Bench、Alpaca 和 Arena-Hard。其使用真实任务输入测接受长度，不代表每个速度表都报告任务分数。
线上报告来自实际用户流量；不能把线上流量实验误写成某个公开数据集的复现。

当前单卡采用解码耗时、加速比和实际 token/s 为性能主线，TTFT、TPOT、E2E 与显存并列，
后续投机后端补充接受率、每轮推进与耗时分解；GSM8K/HumanEval 用于独立质量评测。
这套组合是本项目的实验设计，不声称 DeepSeek 所有文章都使用同一套指标。

## 数据来源与版本

| 数据 | 获取入口 | 使用约定 |
| --- | --- | --- |
| Alpaca | [Stanford 官方仓库](https://github.com/tatsu-lab/stanford_alpaca) 的 alpaca_data.json | 保存下载版本与 SHA256；合并 instruction/input 作为提示，output 不作为当前加载器输入 |
| HumanEval | [OpenAI 官方仓库](https://github.com/openai/human-eval) 的 data/HumanEval.jsonl.gz | 保留原始压缩包、版本和 SHA256；只取 task_id、prompt，转换后才能交给当前程序 |
| ShareGPT | [ShareGPT Vicuna 清洗版数据卡](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) | 有多个清洗版本，必须注明具体文件、commit 和抽取轮次；不能默认等同于所有论文所用版本 |
| LongBench | [THUDM 官方仓库](https://github.com/THUDM/LongBench) | 选择对应 v1 版本、摘要子集和模板，保存处理策略；完整原始长输入未必适合初期显存与长度限制 |

数据下载不代表格式已转换或质量评测已完成。当前自带且直接可运行的数据只有
data/smoke.jsonl 的 3 条流程样本；README 提供 HumanEval 的可选下载和提示转换示例。
