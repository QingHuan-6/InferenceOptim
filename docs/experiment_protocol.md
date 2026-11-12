# 实验协议

研究评价方案已固定在 [评价指标约定 V1.0](evaluation_metrics.md)。
本文件描述当前代码已有的计算行为；尚未实现的比较、置信区间、TBT 和并发统计
不能由当前 summary.json 推定。正式重复测量的分层聚合需要后续对比工具支持。

## 当前能力与计时边界

当前实验是顺序执行的单请求测量，没有请求到达队列、并发负载或服务端调度。
因此不能用当前汇总证明系统吞吐、双 SLO 达成率或线上 P99 性能。
mock 的全部数字是合成值，包含字符数形式的 input_tokens 占位；只能验证流程。
Transformers 后端执行真实单设备贪心生成；4090 首次运行步骤见 remote_4090.md。

真实后端在请求开始前清空先前设备工作并完成同步，再启动单调时钟。
请求起点在分词和输入传输之前；模型加载不计入请求时间。
每个 token 可供调用方使用时记录相对请求起点的时间，最终同步后记录 elapsed_s。
计时不能只包住异步 kernel 提交。需要说明同步/流式返回本身的测量开销。
若后端无法提供 token 可用时间，应另设测量模式，禁止从总时间均摊出 TTFT。

投机解码一次释放多个 token 时允许相同时间戳；应记录实际可用时刻，
不能人为均匀摊开时间。token 指目标分词器的输出 token，而非文本块或字符。
EOS 等特殊 token 是否包含在输出计数中，真实后端必须明确并保持对照一致。
当前 Transformers 后端计入所有生成 token（包含 EOS），每次 argmax 的 item() 隐式同步
并取得 host token ID 后记录时间；最终文本解码计入 elapsed_s。
该观测方式包含逐 token 同步开销，不能直接当作设备端 kernel 极限吞吐。

## 指标定义

设输出 token 数为 N，各 token 相对可用时间为 t₁…tₙ。

| 字段 | 定义 | 单位与边界 |
| --- | --- | --- |
| input_tokens | 后端分词后的实际输入长度 | token，mock 除外 |
| output_tokens | 时间线长度 N | token |
| ttft_s | t₁ | 秒；N=0 时 null |
| tpot_s | (tₙ−t₁)/(N−1) | 秒/token；N<2 时 null |
| elapsed_s | 请求起点到完成的耗时 | 秒，包含最终同步和收尾 |
| decode_tokens_per_s | (N−1)/(tₙ−t₁) | token/秒；N<2 或间隔为零时 null |
| peak_memory_bytes | 本次请求的设备峰值内存 | 字节；不支持时 null |

当前 P50/P99 对“请求级 TTFT/平均 TPOT 等值”作线性插值，
不等同于逐 token 间隔分位数。小样本 P99 仅供流程检查，不能支撑尾时延结论。
场景汇总先按 scenario 分组；整体均值按请求等权，不按输出 token 加权。
当前不会自动评价正确性、计算 KL 散度或给出加速比。

## 可复现性

配置中 warmup 指总预热请求数，按数据集顺序循环取样；预热不进入请求记录和汇总。
repeats 指正式数据集完整遍历次数。每次调用 seed 为基础 seed 加调用索引，
包括预热；相同配置和执行顺序可以重现种子分配。
仅记录种子不保证真实硬件确定性，后端需要声明其采样与确定性设置。

manifest 保存解析后的配置、Python/平台信息、包版本、后端身份、数据集 SHA256、
UTC 起止时间与状态。数据集快照和逐请求文本同步保存。目录名称使用 UTC 和随机 ID。
Transformers metadata 记录设备型号、驱动（可读取时）、运行时版本、精度、attention、
模型及分词器共用的 resolved revision 和生成策略。当前未记录 Git 提交号或完整权重哈希；
使用本地权重时还需人工记录来源版本。

固定模型、分词器、采样参数、输入集合、停止条件与设备条件后才能比较方法。
普通解码与投机解码可通过各自后端实现接入；不要将算法分支加入 runner。
当前仅提供 raw greedy，不做采样、惩罚、强制 token 或模型默认 generation_config 的策略。
stop_on_eos 控制自然停止或固定长度性能测量；输出文本会跳过特殊 token，原始 token_ids 保留。
输入计数包含聊天模板；超过 max_input_tokens 或原生上下文上限会报错，不自动截断。
采样基准需后续扩展请求配置，不能由当前结果宣称分布等价。

## 输出格式与参考验证

schema_version=2 新增 result.token_ids 和 backend.metadata，保留旧指标单位与字段。
mock 仍可不提供 token_ids；真实后端返回与时间线等长的实际 ID。
旧版结果不自动迁移，summarize 仍可读取既有的 summary.json。

verify 命令调用参考 generate，独立保存 verification.json，不加入预热或性能汇总。
它核对所选样本的贪心 token 序列，不核验任务质量或采样分布。
synthetic 只表示数据是否由模拟产生：真实计算为 false，设备验收状态另外记录。

## 后端扩展步骤

1. 在 backends 内新增模块，实现 Backend 协议及 BackendInfo。
2. 在 create_backend 中显式注册；仅在该后端被选择时导入设备依赖。
3. 将模型参数加入独立类型配置，并更新严格校验和复现信息。
4. 用契约测试验证时间线、token 计数、资源释放和超长输出拒绝。
5. 准确标记 synthetic，并在目标设备验证计时边界和生成正确性后用于正式性能结论。
6. 保存完整环境信息，再执行基线与优化对照。
