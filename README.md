# SmolVLA × LIBERO — 零成本复现与改进实验

在**免费的 Kaggle T4×2** 上完整复现 HuggingFace [SmolVLA](https://huggingface.co/papers/2506.01844)（450M 参数机器人操作基础模型）在 [LIBERO](https://libero-project.github.io/) 仿真基准上的微调，并构建"训练 → 无人值守监控 → 协议评测 → 改进实验"的全链路。

## 结果速览

| 项 | 值 |
|---|---|
| 训练 | 8000 updates @ eff-bs 64（512k 样本 ≈ 1.9 epoch），libero_spatial |
| 硬件 | 2 × T4（torchrun DDP），fp16 + GradScaler |
| 耗时 | **4h 50m**（2.16 s/step，双卡 100%） |
| loss | 1.14 → ~0.5（flow matching MSE） |
| 可训练参数 | 100M / 450M（冻结 VLM 骨干，官方 preset） |
| 成功率 | M2 协议评测进行中 |

## 为什么是这套组合

| 维度 | 选型 | 一句话理由 |
|---|---|---|
| 模型 | SmolVLA-base 450M | 唯一能在免费 T4 上全量微调的 VLA 梯队 |
| 数据 | lerobot/libero v3.0 (1.9GB) | 唯一同时满足格式兼容 + Kaggle 磁盘 + 官方复现文档 |
| 框架 | lerobot 0.6.2 @ commit `146d8474` | 官方实现，训练/评测/数据全链路 |
| 并行 | torchrun DDP ×2，bs32/卡 | 450M 放得下 → DDP 优于 FSDP；双卡 1.9× 提速 |
| 精度 | fp16 + GradScaler | T4（Turing）无 bf16 的唯一加速路径 |
| 微调 | 冻结 VLM 骨干 + 专家全量 100M | 官方 preset，显存 9.7/15GiB 富余 |
| 运行 | Kaggle 批处理 + 前台训练 + 崩溃自动 resume | 无空闲计时器，结构上消灭静默失败 |
| 监控 | ntfy.sh HTTP 信标（每 10 分钟） | 零账号、双向可达、5/5 实测可靠 |

完整的选型决策记录（含替代方案对比与预案）见 [PLAN.md](PLAN.md)。

## 复现步骤

1. Kaggle 新建 Notebook（GPU T4 ×2），导入 [kaggle_bootstrap_v11_ddp.ipynb](kaggle_bootstrap_v11_ddp.ipynb)
2. **Save Version → Save & Run All**（批处理模式，关浏览器不影响）
3. 日志健康标志依次出现：`OK-setup` → `Effective batch size: 32 x 2 dp workers` → tqdm 走 8000 步 → `DONE`
4. 每 2000 步落一个 checkpoint（含优化器/调度器/RNG 全状态），崩溃自动 resume

训练入口命令（等价展开）：

```bash
torchrun --standalone --nproc-per-node=2 $(which lerobot-train) \
  --policy.path=smolvla_base_fp32 --policy.push_to_hub=false \
  --policy.input_features=null --policy.output_features=null \
  --dataset.repo_id=lerobot/libero --dataset.revision=a1aaacb7 \
  --batch_size=32 --num_workers=2 --accelerator.mixed_precision=fp16 \
  --accelerator.gradient_accumulation.steps=1 --steps=8000 \
  --policy.scheduler_warmup_steps=2000 --policy.scheduler_decay_steps=32000 \
  --save_freq=2000 --seed=1000 \
  --output_dir=outputs/m1_base_libero --job_name=m1_base_libero
```

## 项目结构

```
├── kaggle_bootstrap.ipynb          # v9 极简单卡版（基础设施参考实现）
├── kaggle_bootstrap_v10_speed.ipynb # v10 单卡 bs32 对照实验（证明 T4 算力饱和）
├── kaggle_bootstrap_v11_ddp.ipynb   # v11 双卡 DDP 最终版（本次训练所用）
├── PLAN.md                          # 三阶段计划 + 差异归因表
├── KAGGLE_SSH.md                    # SSH/隧道方案踩坑记录（已弃用，存档）
├── paper/SmolVLA_paper.md           # 论文笔记
└── logs/                            # 训练日志（从 Kaggle 产物归档）
```

## 无人值守监控架构

训练作为 cell 的**前台进程**运行（崩溃 traceback 直接进日志，结构上消灭静默失败），独立心跳进程每 10 分钟把步数/GPU/磁盘推送到 ntfy.sh 私有 topic，agent 从信标远程接管监控与诊断。每 2000 步一个全状态 checkpoint（优化器/调度器/GradScaler/RNG/采样器偏移），崩溃自动 resume。

## License

MIT
