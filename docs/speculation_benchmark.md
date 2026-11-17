# 正式投机解码性能对照

使用已冻结的 Alpaca 100 条和 HumanEval 50 条输入，在同一 4090 上比较普通贪心、
固定草稿 Prompt Lookup 与自适应草稿 Prompt Lookup。每条固定生成 128 token，
每种方法测 3 次，同一个输入与轮次下旋转三种方法的顺序。两组共 1350 次正式生成，
另有每组每方法 5 次预热。以旧基线约 2.2 秒/次粗估约 50 分钟，按 2 元/小时约 1.67 元，
另计加载和平台计费。按现有路径预计可沿用同一份权重与环境。

固定草稿的上限为 4。自适应在每个请求开始时也用 4，候选全部接受时加 1（最多 4），
有候选但未完全接受时将长度折半（最少 1）。连续两轮零接受时，接下来两轮跳过候选查找，
只由目标模型推进一个 token，随后恢复查找。跳过时仍通过 HF 辅助生成循环执行。
该策略组合为第三种方法，独立变量是整套策略。加速比及配对区间决定是否有可测收益，
接受率与跳过轮数用于解释结果。低于 1 的加速比同样进入表格。

生成逻辑沿用 [HF v4.57.1 的 `_assisted_decoding`](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/utils.py#L3555-L3647)
及 [PromptLookupCandidateGenerator](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/candidate_generator.py#L1053-L1149)。
草稿计数在候选裁剪后取实际验证数量。固定长度且为目标 token 预留一位时，每轮已接受草稿数
等于输出序列增长量减 1。通过张量形状计数，避免为统计额外读取 GPU 标量。由目标模型产生的
额外 token 排除在接受数之外。计时包含候选查找、验证、策略和轻量计数。

```bash
.venv-4090/bin/python -m ainference_optim speculation-benchmark \
  --config configs/4090_hf_alpaca.toml --draft-tokens 4 &&
.venv-4090/bin/python -m ainference_optim speculation-benchmark \
  --config configs/4090_hf_humaneval.toml --draft-tokens 4
```

每组将打印 `table.md` 并保存 `pairs.jsonl`、`manifest.json` 和数据快照。
预热不计入汇总。输出长度不符、计数无效或设备错误会停止并保存失败记录。

| 字段 | 定义及聚合 |
| --- | --- |
| 完整生成耗时 (ms) | 同步 `generate()` 的墙钟时间，包含 prefill，分词和最终文本解码在区间外。先对每条输入的重复取均值，再对输入取均值。原始记录为秒 |
| 加速比 | 普通方法平均完整生成耗时除以当前方法平均耗时。按提示 ID 配对重采样 1000 次，固定种子 42，取 2.5% 和 97.5% 分位 |
| 峰值显存 (GiB) | 每个请求期间 CUDA 峰值 allocated 含常驻权重，取正式请求中最大值；CPU 为缺失 |
| 草稿接受率 | 正式请求的已接受候选 token 总数除以实际送入验证的候选 token 总数；无候选或普通方法为缺失 |
| token 一致 | 完整输出与同输入、同轮次普通贪心完全一致的请求数和总请求数 |

此表适用于单请求完整生成延迟。任务书中的纯解码阶段、TPOT、质量和分布等价各需相应测量。
