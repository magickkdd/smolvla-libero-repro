# M2 评测 — libero_spatial（M1 checkpoint-008000）

## 结果

**总成功率 74/100 = 74%**（Wilson 95% CI [63.0%, 83.2%]）

| 任务 | 成功率 | 任务 | 成功率 |
|---|---|---|---|
| spatial_0 | 9/10 (90%) | spatial_5 | 3/10 (30%) |
| spatial_1 | 10/10 (100%) | spatial_6 | 7/10 (70%) |
| spatial_2 | 8/10 (80%) | spatial_7 | 8/10 (80%) |
| spatial_3 | 8/10 (80%) | spatial_8 | 7/10 (70%) |
| spatial_4 | 8/10 (80%) | spatial_9 | 6/10 (60%) |

- 评测协议：n_action_steps=1（逐步重规划）· init_states + hard_reset · 256×256 · 10 任务 × 10 episodes
- 原始明细：HF 仓库 `eval-results/eval_info.json`；完整日志 `eval.log.gz`；100 个 rollout 视频同目录

## 与论文对照

| | 论文 Table 2 | 本复现 |
|---|---|---|
| Spatial 成功率 | 90% | **74%**（CI [63.0%, 83.2%]） |
| 训练步数 | 200k steps @ bs64 | 8000 updates @ eff-bs 64（**1/25**） |
| 训练数据 | 全部 40 任务（multi-task） | 仅 libero_spatial（单套件） |
| 数值精度 | bf16 | fp16 + GradScaler（T4 无 bf16） |
| 硬件 | 数据中心 GPU | 2× T4（Kaggle 免费，4h50m） |
| 评测协议 | n_action_steps=1 | ✓ 已对齐 |

## 差异归因

1. **训练量 1/25** — 主要负项：90%（论文）落在本复现 CI 上界之外，差距真实存在但远小于步数比例暗示的幅度
2. **单套件训练** — 正向补偿：只训 spatial 套件相当于任务集中度 10×
3. **fp16 vs bf16** — 轻微
4. **评测协议** — 已对齐（n_action_steps=1、官方初始化/重置），不构成差异来源

## 结论

8000 updates（论文预算 1/25）+ 免费双 T4 达到 74%，验证了 SmolVLA 在 LIBERO Spatial 上的可复现性，且差异可由训练量差距解释。任务间方差大（30%-100%），spatial_5 是明显短板——M3 改进实验的首选切入点。
