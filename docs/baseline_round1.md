# 第一轮：4090 原生生成基线

本轮固定 Qwen2.5-1.5B-Instruct、单张 4090、BF16、SDPA、batch=1、raw greedy，
使用 HF 原生 generate。先建立原生实现性能参照，不要求手写路径先通过所有长序列核对。
旧 verification.json 保留；本轮性能结果不能替代严格一致性或无损验收。

## 更新代码

以下命令在服务器项目根目录执行。已有 .venv-4090 和模型无需重装或重新下载。
首次获取本轮工作分支（若同名本地分支已存在，直接 switch 到该分支后 pull --ff-only）：

```bash
git status --short --branch
git fetch origin
git switch --track origin/ai/feat-hf-baseline
git rev-parse HEAD
```

新配置使用项目内 models/Qwen2.5-1.5B-Instruct，沿用用户已下载的权重，
不会覆盖旧 smoke/baseline 配置。若模型位于其他路径，将本轮配置复制到 configs/local/，
修改为绝对模型路径，并相应修正 dataset/output_dir 的相对路径后调用。
模型来源版本、ModelScope 下载记录应与结果共同保存；目前不计算完整权重哈希。

## 准备固定数据（不占用 GPU）

```bash
.venv-4090/bin/python scripts/prepare_baseline.py
```

脚本从固定 commit 的 Stanford Alpaca 和 OpenAI HumanEval 官方仓库下载，
校验内置 SHA256，保留 data/raw/ 原始文件；生成 data/baseline-v1/：

- alpaca.jsonl：100 条指令/上下文，scenario=chat。
- humaneval.jsonl：50 条原始代码提示，scenario=code；保留末尾缩进，不含答案和测试代码。
- preparation.json：原始文件哈希、抽样种子、合格池数量、选中 ID 和输出哈希。
- download_sources.json：固定下载 URL、实际 SHA256 与是否本次下载。

先筛选非空且字符数≤2000的完整提示，再用独立 Random(42).sample 抽样。
这是为初始短输入实验预先声明的筛选，不代表完整 Alpaca/HumanEval 分布，不支持长上下文结论。
字符上限不等于 tokenizer 上限；实际输入含聊天模板，配置上限 4096 token，超限报错，不截断。
两组均采用相同 chat template；HumanEval 此处只作为性能输入，不能当作标准 pass@1。

已有同名输出目录时脚本拒绝覆盖，复用之前的快照即可；需要新版本时用 --output-dir 指定新目录，
并让配置 dataset 指向该目录。已有原始文件校验失败时保留文件并停止，不自动替换。
数据准备失败先解决下载/格式问题，再运行 GPU 实验；不要反复启动模型等待网络。

## 先运行原生路径冒烟，再测两组基线

```bash
.venv-4090/bin/python -m ainference_optim run --config configs/4090_hf_smoke.toml
.venv-4090/bin/python -m ainference_optim run --config configs/4090_hf_alpaca.toml
.venv-4090/bin/python -m ainference_optim run --config configs/4090_hf_humaneval.toml
```

第一条仅 3 条输入、32-token 输出，确认新的计时路径可在服务器运行。
首条出错则停止，不继续后两条。两组基线均预热 5 次、每条正式重复 3 次，固定输出128 token、
忽略 EOS；分别保存 300 和 150 条正式请求。先顺序完成一组再启动另一组，不并行抢占同一张卡。
每条命令都会加载模型；模型加载不计时，但会占用租机时间。
run 默认在 stderr 显示模型加载提示、预热与正式请求完成数、百分比、已完成样本 ID、
重复轮次、阶段已用时间和估算剩余时间。每条请求完成并落盘后更新一次，不是逐 token 进度。
阶段墙钟时间包含请求间处理与打印，不等于性能指标；进度回调不进入后端请求计时。
不同输入长度会影响剩余时间估算。需要安静运行时，在 run 命令末尾加 --quiet。

HF generate 的 streamer 首次发送 prompt，计时器核验并跳过它；之后 token 到达 CPU 时记录时间。
末尾核对 streamer ID 与最终输出，防止把输入算成输出或漏记 token。
此模式含每 token CPU 传输、回调和同步成本，不是无观测开销的 GPU kernel 峰值测量。
HF 自身验证是同一个生成实现，不能用它与 reference_tokens 一致来宣称独立验证了生成算法。

## 查看和保存结果

每次运行打印独立结果目录；替换下面的目录名查看汇总，无需再次加载模型：

```bash
.venv-4090/bin/python -m ainference_optim summarize outputs/<实际基线目录>
```

检查 manifest.json 的 status=completed，completed_requests 分别为300/150，
backend.metadata.decode_path=hf_generate_streamer_v1；查看 summary.json 的 input_tokens、output_tokens，
后者应 min=max=128。查看 mean/P50 的 decode_elapsed_s、tpot_s、ttft_s、elapsed_s、
decode_tokens_per_s，以及 peak_memory_bytes_max（字节除以2³⁰为GiB）。
小样本 p99 只作记录，不作服务尾延迟结论。

manifest 自动保存 Git commit/dirty/status；无 Git 环境时写 null 及原因。
输入快照、抽样与下载记录会复制到各结果目录，模型环境和解析后的配置也保留。
旧 summary 均值按请求汇总，本轮每题重复次数相同，等同于先对题目取重复均值再等权平均。
输出仍含实际输入和模型文本，备份整个结果目录，不只保存一张截图。

本轮应交付两组性能基线及原始记录；不存在优化结果时，不会产生加速收益结论。
后续先加入不训练额外模型的 n-gram/prompt lookup 投机方法，再测草稿长度1/2/4/8。
当前代码尚未实现投机后端、GSM8K/HumanEval质量评分或并发服务。

## 后续配对比较

两组基线完成后，先用小规模计时开销检查，区分原生执行与逐 token 观测成本：

```bash
.venv-4090/bin/python -m ainference_optim timing-overhead \
  --config configs/4090_hf_alpaca.toml --samples 10 --repeats 3
.venv-4090/bin/python -m ainference_optim timing-overhead \
  --config configs/4090_hf_humaneval.toml --samples 10 --repeats 3
```

每条命令只加载一次模型，使用原数据前10条，每条3对正式测量（每对分别有/无 streamer），
共60次正式生成；另有默认2对预热。进度的30/30表示配对数，不是单次生成数。
同一对使用同一输入、种子、输出长度和生成规则，按样本索引与重复编号交替 AB/BA 顺序。
预热与正式记录分离。仍有CPU循环、HF实现和驱动开销，无回调不代表最优推理引擎。

独立 outputs/*-overhead-*/overhead.json 保存摘要、配置、后端身份、输入快照哈希和选中ID，
pairs.jsonl 保存执行顺序、双方原始 token ID 与计时。原来两组基线结果不会被覆盖。
与性能 run 不同，此诊断暂未自动采集 Git，启动前另存 git rev-parse HEAD 及 git status 记录。

- generation_elapsed_s：对完整 generate 前后做设备同步，单位秒，排除分词和最终文本解码。
- request_elapsed_s：包括分词、输入传输、生成和最终文本解码，单位秒，排除模型加载。
- 每项分别报告两种模式的平均值及 streamer/plain 均值之比。
- streamer_overhead_percent = (streamer均值/plain均值−1)×100，单位%；可能为负，不强行归零。
- 正式重复数相同，均值等价于先对每题重复求均值，再对题目等权；不对单次比值取平均。
- token_match_pairs/pair_count 仅描述本次配对是否得到相同输出，不代表通过独立正确性验收。

无回调模式不保存 token 时间线，也不生成 TTFT、TPOT、解码阶段耗时或峰值显存估计。
其计时边界不同于旧基线，不能把无回调的完整 generate 时间与旧 TPOT 直接相除。
这里只比较新实验内双方同口径计时；小子集结果不能自动代表整个数据集，且没有置信区间验收。
若输出不同，保留差异；先检查固定长度、生成设置与运行状态，不能简单将耗时差归因于回调。
completed只表示诊断完成，不将观测开销下降称为解码算法加速收益。

两种模式差别很小，则不支持“主要被streamer拖慢”的猜测；若差别明显，先调整后续测量模式，
再用统一模式重测优化方法及相应参照，不把两套计时方式混成加速比。

后续方法对照使用：

```bash
.venv-4090/bin/python -m ainference_optim compare \
  outputs/<同一数据集基线目录> outputs/<候选方法目录>
```

当前工具适用于固定输出长度的同负载对照，拒绝数据哈希、种子、重复数、模型、精度、
attention、停止策略、设备及记录的环境差异。不能把 Alpaca 与 HumanEval 两组直接相除。
对比工具按提示先求重复均值，再等权汇总，给出解码耗时之比、E2E加速比和提示级配对
bootstrap 95%区间（1000次，seed=42；不足2题时区间为null）；峰值显存取所有请求最大值。
只按场景报告主结果，不将不同场景混出一个总加速比。输出会新建 comparison-*/comparison.json。
它不核验任务质量或无损，不替代数值诊断；也不能消除两次分开运行的设备状态差异。
正式方法对照仍需交错重测，当前工具不负责交错执行。未来投机算法增加新配置字段后，
需明确哪些字段是算法变量并扩展可比性检查，不能通过随意放宽全部配置比较来接入。
