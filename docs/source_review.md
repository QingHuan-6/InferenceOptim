# 源码审阅与实验成本约定

审阅日期：2026-11-13。目标是先借鉴成熟代码，区分算法、引擎和测量方式，
避免把实现常识和没有决策用途的参数猜测转化为付费 GPU 实验。
本次仅下载、阅读公开源码并检查本地已安装 Transformers，没有运行 GPU、安装新框架或修改原结果。
固定版本代表本次审阅快照，不声称全部复现对应论文发表时的版本。

## 实际阅读的实现

| 实现及版本 | 源码入口与观测 | 对本项目的约束 |
| --- | --- | --- |
| EAGLE，cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b | [Qwen3 评测 L213–221](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/evaluation/gen_ea_answer_qwen3.py#L213-L221)：同步后测完整 eagenerate，完成同步后停止计时；分词在计时前，输入 tensor 上卡在计时内；最终文本 decode 在计时外。另读 [普通基线 L265–278](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/evaluation/gen_baseline_answer_vicuna.py#L265-L278) | 方法对照使用相同边界；不把我们的含分词/最终解码 E2E 与该源码 wall_time 直接混比 |
| DFlash，07ebd93db9f472af339b644bb70221ad8428328a | [model.py L174–223](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L174-L223) 分开记录 prefill/首token 与完整 decode 阶段，_cuda_time 做同步；[L315–323](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L315-L323) 用 decode 时间除以 num_output_tokens。benchmark.py 将 tokenization 移到生成前，使用 block_size=1 和草稿块分别测量，并对两种模式预热 | 可以借鉴阶段计时与共享路径，不能照抄指标名称：其分母 N 与本项目排除首token后的 N−1 不同；内部仍可有接受/停止逻辑的 CPU 同步，不称其为完全异步 |
| vLLM，10a854ed6b3bbd7411c65ccbf87594e71c4d322c | [latency.py L96–140](https://github.com/vllm-project/vllm/blob/10a854ed6b3bbd7411c65ccbf87594e71c4d322c/vllm/benchmarks/latency.py#L96-L140)：预先构造 token 输入、ignore_eos、预热，perf_counter 包围阻塞式 llm.generate，use_tqdm=False；独立统计重复时延 | 离线完整调用延迟是一种有效主测量，不需要为获取此指标增加逐 token Python 回调；引擎调用已阻塞完成，不能机械地把其无显式 synchronize 写法搬到原始异步 CUDA 提交 |
| vLLM，同上版本 | [backend_request_func.py L306–359](https://github.com/vllm-project/vllm/blob/10a854ed6b3bbd7411c65ccbf87594e71c4d322c/benchmarks/backend_request_func.py#L306-L359)：在客户端收到 HTTP 流块时记 TTFT/ITL，检查空响应、错误及 usage 输出计数 | 服务流式延迟属于客户端观测，含传输/服务调度，且 chunk 不自动等于单个 token；不能拿它代替离线 GPU 阶段时间 |
| SGLang，1b2870d33de7b5f31844fd74e97eea6d7896c53a | [one_batch.py L798–877](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/benchmark/one_batch.py#L798-L877)：分别测 extend/prefill 和逐步 decode，在步骤边界同步，汇总 median decode latency 与 batch_size/latency | 逐步同步并非一概错误，它服务于单批次微基准；必须标明是在测引擎步骤，不能把其总吞吐公式或中位数与我们的请求级均值直接互换 |
| Transformers v4.57.1，本项目固定安装版本 | [generation/utils.py](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/utils.py) 的 _sample：streamer.put(next_tokens.cpu())；随后计算 unfinished_sequences.max()==0，并在 _has_unfinished_sequences 中用于 Python 控制流 | streamer 确有 CPU 传输/回调成本，但移除它并不等于消除所有同步；仅从源码不能估算它占总耗时多少，更不能把约58.7 token/s的原因归结于它 |

上述代码是功能和计时参考，不照单全收：版本、停止条件、统计分母、模板、框架优化和输入类型
必须逐项核对。被审阅的源码副本存于本地忽略目录 outputs/source-review/，不向仓库复制第三方实现。

## 对当前代码与已完成实验的结论

1. hf_generate_streamer_v1 属于 host token 可用时间观测，现有 TTFT/TPOT 定义自洽，
   数据无需删除；它不是无观测开销的离线引擎极限性能。
2. 约58.7 token/s和两次约0.2%的均值差异只能说明该配置下的观测结果，
   没有 profiler 或设备阶段证据就不能判断 CPU、GPU、streamer 各占多少。
3. 撤销“完成两组基线后默认跑 timing-overhead”的流程；该功能保留为按需诊断，
   本次审阅没有实测其开销比例，不给出修复或提速结论。
4. 后续算法实验优先选一个成熟生成栈中的普通/投机两条路径，以相同模型、输入和计时边界比较。
   对固定安装的 HF，先审阅内置 prompt_lookup_num_tokens、PromptLookupCandidateGenerator 和
   _assisted_decoding，再决定是否需要自行实现 n-gram 候选与缓存回退。
   不能一开始就手写已有可靠实现，也不默认它在 Alpaca 聊天上一定有收益。
5. 主性能计时、流式响应观测、调试正确性、任务质量分别记录用途；切换计时模式后，
   新旧结果不直接相除，不以测量路径变更冒充解码算法加速。

## 后续开发与付费实验顺序

这些是工程责任和停止规则，不增加逐次用户确认的审批流程。

1. 先查已有结果、当前环境与固定版本成熟实现。记录文件/commit/调用路径，
   明确预热、同步、首token、计时范围、输出计数和聚合方式。
2. 先在本地用随机小模型验证生成参数、mask/位置/缓存、停止规则和计时边界；
   运行必要的 unittest、Ruff、mypy。CPU 测试不证明 GPU 吞吐或数值确定性。
3. 提出一个会改变后续决策的假设：写明唯一主要变量、原始记录、判别规则、
   预计调用数/耗时、无法判断时的停止条件。阈值须在看结果前确定，不事后挑数据。
4. 设备检查先用最小的1–3条输入、短输出；确认路径/计数/内存没有问题再扩展。
   需要长上下文或晚期分叉的检查才增加输出长度，并说明短输出不能回答的原因。
5. 成本先按已有逐请求耗时×调用数估计，另计模型加载和预热。超出预计预算或出现
   OOM、输出计数不符、配置未生效时停止，不继续整套扫描。区分预期的数值分叉与执行错误。
6. 一轮只回答一个问题，完成后先分析；只有新证据、代码变化或明确的统计不确定性才加测。
   不能为了“看起来更稳”反复重跑相同基线。能从落盘记录计算的指标离线算，不重载模型。

## 下一步的最小交付

先离线分析已有 Alpaca/HumanEval 原始记录，形成同一口径的基线表和长度/重复波动统计；
服务器结果尚未提供给本地时，只需传回 summary/manifest 和必要原始记录，无需重新推理。
随后为选定的成熟投机路径写清源码对照与小模型测试，再给出一次有明确目标的 GPU 小实验。
当前不要求新增整套基线、安装 vLLM/SGLang、训练草稿模型或复现 DSpark 集群。
