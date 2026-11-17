# 首次提示查找投机小试验

4090 已经跑通时，可直接使用[正式三路对照](speculation_benchmark.md)产出主表。

本轮只判断现有 Qwen2.5-1.5B、HF 4.57.1 路径能否执行，以及少量输入上的完整生成耗时和显存。
使用内置 Prompt Lookup，从已有 token 序列的重复 n-gram 提出候选，由目标模型验证；
无需额外草稿模型，不训练权重。它不是 EAGLE、DSpark 或本项目自适应算法的复现。

## 源码依据与兼容处理

固定 Transformers v4.57.1，实际阅读本地安装源码，并对照：

- [generation/utils.py](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/utils.py)：
  `_get_candidate_generator` 选择 PromptLookupCandidateGenerator；`_assisted_decoding` 中
  L3555–3647 获取候选、将 logits_to_keep 改为候选数+1、验证最长匹配前缀并裁剪 KV。
  不自行重写候选生成、验证或 cache 回退。
- [candidate_generator.py](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/candidate_generator.py)：
  PromptLookupCandidateGenerator.get_candidates 在历史序列寻找重复，未命中返回空候选；
  update_candidate_strategy 不做自适应。本轮草稿上限4、匹配 n-gram 上限2。
- 固定版本候选命中后会调用 `isin(chosen_ids, eos_token_id)`；关闭 EOS 时传 None 会报错。
  本地 CPU 集成测试已复现，因此投机固定长度模式显式传空 EOS 列表，普通路径仍用 None；
  二者均不因 EOS 提前停止，不修改第三方包、不启用模型默认采样参数。
- 同版本候选长度未始终按剩余输出预算裁剪，本地重复输入测试复现8 token上限返回9 token。
  后端仅对当前实例临时包装候选工厂，把候选截到总长度上限减1，给验证产生的额外token留位。
  工厂在正常或异常退出后恢复，不改HF全局类；候选搜索、匹配验证和KV裁剪仍调用HF原实现。
- 与 [源码审阅记录](source_review.md) 一致：无 streamer；generate 前后同步，包含 prefill、
  候选查找和验证，不含分词、输入上卡和最终文本 decode；模型加载在计时之外。
  这与已有 hf_generate_streamer_v1 不同，不与旧结果直接计算加速比。

本地验证：58项 unittest 通过，含 torch 2.9.0+cpu / Transformers 4.57.1 的随机小型 Qwen2
集成检查；覆盖候选命中、多位置验证、1/2/7/8/32 token边界、普通路径恢复与异常恢复。
Ruff、格式检查和 mypy 通过。尚未进行4090设备实测，未测收益或真实模型质量。

## 假设、规模和停止规则

- 假设：此内置路径能执行固定长度生成，且在本轮输入中可能降低完整生成延迟。
- 唯一主要变量：普通贪心 vs 草稿上限4的提示查找；同模型、精度、输入、种子、输出长度。
- 默认从 smoke.jsonl 取前3条，固定输出32 token，各重复2次；按样本和轮次交替 AB/BA。
  两种模式各预热一次，共14次生成。CLI 限制最多3条×3轮，输出最多128 token，防止误开大跑。
- 以已有约2.18秒/128 token粗略估算，同等时长14次约31秒；这不是设备承诺，另加加载。
  建议为本次命令预留5分钟，按2元/小时折合约0.17元运行预算；租机准备时间另计。
  若单请求异常耗时或总时长超过5分钟，人工 Ctrl+C 停止，保留失败报告，不自动扫参。
- OOM、配置不生效、输出计数不符或执行异常即停止，保留失败状态与已完成配对。
  token 差异单独记录，不自动当作已知数值误差，也不因此重新跑完整基线。
- 预先规定：完成仅表示路径可用；完整生成加速比>1.10仅作为进入小规模代表性评测的候选信号，
  不是正式结论。≤1.10时先停止扩大规模，结合原始记录分析，不盲目增加重复。
  少量短聊天/代码/JSON不能代表输入复用任务，也不能据此否定所有投机方法。

## 服务器执行

需先把包含本功能的代码同步到服务器，再从项目根目录运行。依赖沿用现有环境，无需重新 setup：

```bash
.venv-4090/bin/python -m ainference_optim speculation-pilot \
  --config configs/4090_hf_smoke.toml \
  --samples 3 --repeats 2 --draft-tokens 4
```

模型来自 `models/Qwen2.5-1.5B-Instruct`，输入来自 `data/smoke.jsonl`，不需额外数据下载。
命令显示加载提示及配对进度，成功后打印汇总和唯一结果目录。
如服务器使用自己修改的模型路径，复制配置到 configs/local/ 后需重新核对相对路径：
该目录中的模型、数据、输出路径分别应从 `../../models/`、`../../data/`、`../../outputs/` 起算。

## 输出与解释

输出 `outputs/<实验名>-lookup-pilot-<唯一标识>/`：

- pilot.json：代码提交与dirty状态、原始配置、输入哈希、后端版本、选中样本、预热与完成数。
- pairs.jsonl：每对执行顺序、完整输出ID、输入长度、耗时、峰值显存及首次输出分叉位置。
- dataset.jsonl：原始数据快照，实际选用ID在报告中列出。

指标定义：

| 字段 | 单位与定义 | 聚合及缺失规则 |
| --- | --- | --- |
| generation_elapsed_s | 秒；同步 generate 完整调用 | 每条每轮原始值；预热不计入汇总 |
| request_elapsed_s | 秒；含分词、上卡、generate、最终解码 | 仅原始记录，与 generation 不混用 |
| generation_speedup | 无量纲；普通生成平均耗时/投机平均耗时 | 等重复次数、先算耗时算术均值再相除；不是纯decode加速比 |
| peak_memory_bytes | 字节；CUDA 请求峰值 allocated，含常驻权重 | 每次独立重置峰值；CPU为null，不代表reserved或整卡占用 |
| first_mismatch_index | 从0开始的首个输出分叉位置 | 无差异为null；不自动解释分叉原因 |
| token_match_pairs | 对；完整输出ID相同的配对数 | 与pair_count同时报告；不等于分布或任务质量验收 |
| acceptance_rate | 未测量 | 固定为null，不将输出数冒充接受数 |

本轮不提供 TTFT、TPOT、KL或任务质量分数；不得从完整耗时伪造逐token指标。
请保留整个结果目录，先看本次报告再决定下一项实验，不连续启动两套完整数据集。
