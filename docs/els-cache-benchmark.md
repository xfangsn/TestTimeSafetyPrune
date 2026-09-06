# ELS 不变计算缓存：Hazel 对照

2026-09-06。只缓存原始模型上的 BLADE-G 分数经每矩阵候选筛选后的结果；方向、输入矩、Q、λ、权重与评分设置固定。每个候选层组合仍重新进行全局排序。此次没有更改生成、PPL、搜索阈值，也没有引入提前停止或小样本近似。

## 实现与约束

`src/ttsafety/els_cache.py` 提供 `ELSCandidateCache(score_fn)`，通过 `ranking_fn=cache.rank` 接入 `solo_layer_pool` 和 `bestfirst_layers`。同一次拟合的两个阶段可以共用一个缓存。原函数默认路径保持原样。

每个矩阵缓存原排序实现的 `torch.topk(..., sorted=False)` 输出，包括 FP32 分数、原权重索引、矩阵元素数。按名称排序后重新拼接候选池并执行 global top-k，保持原来的并列分数处理。没有缓存或拼接单层最终 mask。

缓存仅用于恢复后的原模型，不能在剪枝上下文中建立。若模型权重、方向、输入矩、Q、λ 或评分设置改变，必须新建缓存或 `clear()`；对象身份检查不能检测原地修改及闭包内容变化。缓存会占用额外主机内存，统计中报告实际候选张量字节数。

## 测试范围

- 本地：缓存与原版逐 tensor 比较，覆盖多层组合、多种比例、并列/零/负无穷分数、B/G 真实打分函数、缓存失效及 ELS 调用；31 项通过。基准输入解析另有 5 项通过。
- GPU 排序：重放原 BLADE+ITI 作业的全部非空候选轨迹；保留 solo 与第一轮的重复调用。分别记录从空缓存起步的一整段轨迹，以及同一缓存再次重放轨迹的耗时。两者均与交替执行的原版逐 tensor 比较。
- GPU 搜索：候选池固定为 `[6,22,24]`，两轮按原版/缓存版、缓存版/原版顺序执行。每个搜索使用空缓存，完整运行真实生成与 PPL。比较所有候选 mask、输出、PPL、行为率及最终选层。

缩小候选池的搜索不是全 36 层 ELS 的端到端加速实验。全轨迹重放仅测排序，不重新执行全轨迹的生成。不可将排序加速比当作完整方法加速比。

## 固定设置

Qwen3-8B，thinking OFF；原训练/选层划分与方向拟合；Q=g1scalar，C4 上限 65,536 token、窗口 2,048、batch=2。原离线文本实际产生 17 个完整窗口，即 34,816 token。

ELS 保留 test_frac=.005、beta=.05、eps=.005；42 个选层问题，greedy 128 token、batch=16；PPL max_tokens=5000、window=1024、batch=8。额外比较 `.01` 比例的 mask 时仍使用同一个 ELS scorer，不代表生产版 strict-positive 最终 mask 的独立审计。

## 执行记录

首次快照试跑 JobID：735581，已在启动约一分钟内取消，原因是用户明确项目应通过 Git 同步代码与结果。该作业不作为性能结论来源。

隔离目录：`/share/jekml/xfang23/ttsafety_els_cache/cache_20260906`。
参考目录：`/share/jekml/xfang23/ttsafety_blade_iti/rho001_a2_20260906/output`。
本地记录：`results/els_cache/20260906/`。

快照 SHA256：`84029bc285ae0f6bd0b00201df70f598bd4dabf52ff57308c184b677f90eeba6`，远程解包前及逐文件校验均通过。

正式测试改用 Git 分支 `codex/els-cache-20260906`，Hazel fetch 后创建独立工作树。输入、参考轨迹和最终结果 JSON 通过该分支同步；模型、Q、大型 tensor、环境和日志不纳入 Git。单张 L40S，8 CPU，128G 主机内存，2 小时上限。

结果由 `results/els_cache/20260906/gpu/summary.json` 与 `controlled_*.json` 逐步保存。当前这份记录只说明实现、验证范围和提交信息；性能结论待正式作业完成后补充。
