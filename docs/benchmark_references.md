# 解码性能评测的论文依据

本记录核对论文正文中的数据集、工作负载与指标，不以博客或开源框架现有默认配置替代论文。
调研日期为 2026-11-12。下列论文发表于 SOSP、OSDI、ASPLOS；这些会议属于 CCF A 类，
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

## 数据来源与版本

| 数据 | 获取入口 | 使用约定 |
| --- | --- | --- |
| Alpaca | [Stanford 官方仓库](https://github.com/tatsu-lab/stanford_alpaca) 的 alpaca_data.json | 保存下载版本与 SHA256；合并 instruction/input 作为提示，output 不作为当前加载器输入 |
| HumanEval | [OpenAI 官方仓库](https://github.com/openai/human-eval) 的 data/HumanEval.jsonl.gz | 保留原始压缩包、版本和 SHA256；只取 task_id、prompt，转换后才能交给当前程序 |
| ShareGPT | [ShareGPT Vicuna 清洗版数据卡](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) | 有多个清洗版本，必须注明具体文件、commit 和抽取轮次；不能默认等同于所有论文所用版本 |
| LongBench | [THUDM 官方仓库](https://github.com/THUDM/LongBench) | 选择对应 v1 版本、摘要子集和模板，保存处理策略；完整原始长输入未必适合初期显存与长度限制 |

数据下载不代表格式已转换或质量评测已完成。当前自带且直接可运行的数据只有
data/smoke.jsonl 的 3 条流程样本；README 提供 HumanEval 的可选下载和提示转换示例。
