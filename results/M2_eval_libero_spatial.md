# M2 评测 — libero_spatial（M1 checkpoint-008000）

## 评测协议

| 项 | 值 |
|---|---|
| checkpoint | 008000（8000 updates @ eff-bs 64，fp16 + GradScaler） |
| 任务 | LIBERO Spatial（`libero_spatial` 任务套件） |
| 重规划频率 | **n_action_steps = 1**（每执行一个动作后重新观测-规划，与论文协议一致） |
| episodes | 10（快速档；统计区间见下） |
| eval.batch_size | 1 |
| 初始状态 | `init_states=true` + `hard_reset=true` |
| 渲染分辨率 | 256×256 |
| 评测命令 | `lerobot-eval`（lerobot 0.6.2 @ commit `146d8474`） |

## 结果

- 成功率：**待填**（`EVAL-DONE` 信标回传后更新）
- 10 episodes 的 95% Wilson 区间极宽（例如 9/10 ≈ [60%, 98%]）——本档位结果仅作方向性参考，CI 级结论需 30-50 trials/任务（M2 完整矩阵）

## 与论文对照

| | 论文 Table 2 | 本复现 |
|---|---|---|
| Spatial 成功率 | 90% | 待填 |
| 训练步数 | 200k steps @ bs64 | 8000 updates @ eff-bs 64（**1/25**） |
| 训练数据 | 全部 40 任务（multi-task） | 仅 libero_spatial（单套件） |
| 数值精度 | bf16（数据中心 GPU） | fp16 + GradScaler（T4 无 bf16） |
| 硬件 | 数据中心 GPU | 2× T4（Kaggle 免费） |
| 评测协议 | n_action_steps=1 | n_action_steps=1 ✓ 已对齐 |

## 差异归因（预期方向，待结果验证）

1. **训练量 1/25** — 预期主要负项
2. **单套件训练** — 对该套件成功率偏正（论文为 multi-task 训练后分套件报告）
3. **fp16 vs bf16** — 轻微影响
4. **评测协议** — 已对齐，不构成差异来源
