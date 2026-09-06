# 实际执行：完整 BLADE ρ=0.01、α=2 + ITI

2026-09-06。以用户最新指令为准：只生成 transfer/refit 两类组合，旧单方法结果不重跑。
用户明确要求完整 BLADE 流程、其余超参数不变；先前拟复用旧 L* 的方案已撤销且未提交。

## BLADE

基准流程为 `scripts/blade_ood_run.py` 的 BLADE-G 分支：重新提取 direction/moments、
重新计算 g1scalar Q、运行完整 solo_layer_pool + bestfirst_layers、构建最终 mask。
不使用任何历史 L* 作为输入或失败后的替代值。

- 最终编辑：rho=0.01，alpha=2，components=both。
- 方向与矩：沿用原 `split_3way(seed=0)`；direction batch=16，moments batch=8。
- Q：g1scalar，C4 65536 tokens，seqlen=2048，batch=2，no_grad。
- lambda：全层 positive-median behavior score / positive-median Q；alpha=2 时 scale factor=1。
- ELS：screen_frac=0.005、test_frac=0.005、beta=0.05、eps=0.005，均是原脚本固定值。
- ELS 生成：128 new tokens、greedy、batch=16；C4 PPL max_tokens=5000、window=1024、batch=8。
- ranking：max_fraction=0.05、per_matrix_cap=0.10，与现有代码一致；最终 mask 增加正/有限值审计，
  如不满足即报告失败，不补出一个不同方法的 mask。

## 两类组合

1. transfer：在原模型重建 ITI 方向/投影 std，保留历史 baseline 中记录的 48 个 head ID。
2. refit：在完整 BLADE 重加权模型内重新提取激活，按原 mean-diff L2 规则重选 48 heads、
   重估 direction 与 projection std。

两者均 c={2,4,6}，共 6 条件；thinking OFF，生成128 tokens，greedy，batch=12。
沿用已有 ITI 的 legacy decode-only 规则（跳过整个 prefill），以配合现成 baseline。
本次不引入 plan v3 的 probe-CV、新位置策略、新数据划分或扩大参数搜索。

使用既有 ITI 对比的相同 280 题（SelfAware/FalseQA 四个 cell 各70），顺序和题目哈希冻结。
这是使用已看过数据的补充实验，不宣称全新 held-out test。
原方向/尺度 tensor 未保存，所以 transfer 是按相同配方重建，并非原 tensor 的逐位复用；
保存 historical heads 与本机重算 ranking 的 overlap、所有方向与 sigma，便于审计。

## 验证及执行

新增 `scripts/blade_plus_iti.py`、`src/ttsafety/iti_composition.py`；7 项本地测试通过，
覆盖异常恢复、组合确实使用编辑后的 o_proj、prefill 跳过与 decode 注入、正分数候选约束、
transfer/refit head 来源，以及新 runner 的划分与现有 BLADE 原函数完全一致。
GPU fit 中另做数值 smoke；这些小型训练题检查不属于重跑 OOD baseline。

Hazel 隔离快照目录：
`/share/jekml/xfang23/ttsafety_blade_iti/rho001_a2_20260906`

拟合作业完整执行 BLADE + 两份 ITI 拟合。成功后 afterok 自动启动 array 的 transfer/refit
两个任务，各跑三个剂量，逐 batch 原子保存结果。使用 L40S；fit 128G host RAM、
generate 64G，每任务2小时。原实验目录与结果不覆盖，不提交/推送 Git。

Hazel 环境目前 torch 2.5.1+cu121、transformers 5.0.0；旧 OOD 为 torch 2.11.0+cu128、
transformers 5.15.1、RTX5090。环境差异必须随结果呈现。
PPL 延续旧 all-position stress 口径，明确区别于 decode-only 生成；同时报告相对本次
ELS 已计算 base PPL 和相对旧 baseline 的增幅，不混淆两个分母。

输入与快照清单：`results/blade_iti/rho001_a2_20260906/`。
更新后的传输快照 SHA256：`b270985289ede923d98f1444c04a1d2907cc45e36436f19aaa61fc44b93eb98f`。
已提交：fit JobID **735481**，组合 array **735482**（0=transfer，1=refit）。
首次启动检查：735481 在 **gpu18** RUNNING，735482 因 afterok dependency 等待。
状态记录：`results/blade_iti/rho001_a2_20260906/submission.json`。
运行目录为上述快照路径下 `output/`；日志位于 `output/logs/`。
此处只记录提交/启动证据，不代表实验已完成或组合已验证有效。
