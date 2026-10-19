# SmolVLA 复现计划（Kaggle 版）v6

> 项目：SmolVLA 轻量级 VLA 的微调与泛化改进 · 论文 arXiv:2506.01844
> 执行环境：Kaggle Notebook（T4 16GB）；本地仅资料/代码梳理，不训练。
> 代码基线：lerobot main @ `146d8474932c8b566717eb2738ba459337ab71c4`（pyproject version 0.6.2，无 tag，以 commit 指代）。
> 配套文档：SSH/开关机 → [KAGGLE_SSH.md](KAGGLE_SSH.md)（v3，含 SSH 侧 token 来源与 MagicDNS）。

**v6 修订说明**：v5 的 C1–C4/§8 三条/双校验/周数自洽经复核全部成立。本版修：①**预算 seed 维度缺口**——item 3b 训练了基线第二 seed 但 §5 没有它的评测（协议 800 + OOD 1000 = 1800 eps），新增 item 5b；同时修正 v5 高端的加总错误（117.8h 误写 108h）→ 核心 GPU **57–130h ≈ 1.9–4.3 周配额**、墙钟 62–140h，item 5/6 标题限定"seed#1"防再漏；②**`--policy.private=true`**——`private=None` 默认建**公开** repo（[train_utils.py:562](lerobot/src/lerobot/common/train_utils.py#L562) docstring 原话），10.4GB checkpoint + 全部超参不能裸奔；③**后台模式删不了本地 ckpt**——save→push 顺序无任何本地清理且无交互 shell，分账表改"按 10.4GB 峰值规划"（或 save_freq=16000 换余量）；④**U>6 分支不再静默违背 seed≥2**——保留 3b（训练 26k updates），评测数字给出收紧后口径，极端时才显式降级；⑤**SSH 侧 HF_TOKEN 来源**——kernel 的 os.environ 不进 SSH shell（[KAGGLE_SSH §1.4](KAGGLE_SSH.md)），两文档各补 `/tmp/hf_token` 方案；⑥精度——"5 项中 3 项为库默认"、几何臂补 RandomRotation 消增强密度混淆、协议档曲线变体命令、交叉引用改正、pip 缓存机制改正（PIP_CACHE_DIR ≠ --find-links 目录）。

---

## 1. 目标与范围

- **不复现预训练**（论文全项目约 30k GPU 小时）。本项目"复现" = 预训练权重 + 微调 pipeline + 论文评测协议（每任务 10 trials）+ 消融方法论。
- **复现目标**：主臂 = `lerobot/smolvla_base` 微调后的 LIBERO 表现，**以自身 success-rate vs updates 曲线为准**；论文 Table 2 数值仅"非可比参考"并列（三重不可比：VLM-init vs 机器人预训练、100k vs 8k updates、bf16+compile vs fp16）。
- **可选 VLM-init 臂**：`--policy.type=smolvla --policy.load_vlm_weights=true` 跑 8k updates——与 Table 2 **仍不可比**（VLM-init 的 action expert 随机初始化，8k 远未收敛），只能作为"与我们自己的 8k smolvla_base 臂对照"**隔离 init 因素**。仅配额富余时做。
- 主战场 LIBERO；`svla_so100_pickplace` 为可选辅助臂。改进方向 A（视觉增强）优先。

## 2. 术语、单位与统计（全文强制）

**单位**

| 术语 | 含义 | 换算 |
|---|---|---|
| micro-batch（mb） | `--steps` 计数单位；训练日志 `step_s`/`update_s` 的单位 | 主跑 `--steps=32000` @ bs16 |
| update | 一次权重更新 = mb × accum4 | 32000 mb = **8000 updates** |
| 时间锚点 | A100 20k updates ≈ 4h → T4 0.72–1.53 **s/mb** | = **2.88–6.12 s/update**；8k updates ≈ 6.4–13.6h |

- **M0 读数双校验**：①日志 `step_s` ×4 = s/update；②`s/update = 有效 batch 64 ÷ 日志 samples_per_s`（`samples_per_step = batch_size × dp_world_size` **不含 accum**，[lerobot_train.py:789,970](lerobot/src/lerobot/scripts/lerobot_train.py#L789)）。
- LR 调度按 mb 步进；任何 run 显式设 `--policy.scheduler_warmup_steps/--policy.scheduler_decay_steps` ≈ `--steps`。**resume 时不重建 preset**（[train.py:310](lerobot/src/lerobot/configs/train.py#L310)）——**降范围只对新 run 生效，禁止对已开始的 run 中途改 `--steps`**。

**评测两档（不得混用）**

- **快速档**（趋势）：训练内 `--env_eval_freq`（单位 mb），**必须显式 `--eval.n_episodes=10 --eval.batch_size=2 --eval.use_async_envs=false --env.init_states=true --env.hard_reset=true --env.observation_height=256 --env.observation_width=256`**（与协议档同条件）。耗时：100 eps → **≈300 次 batched-2 调用** + 100 次 hard reset ≈ **6–13 min/次**（M0 实测回填）。
- **协议档**（结论）：`lerobot-eval`，`n_action_steps=1` + 256×256 + `hard_reset=true` + `--eval.batch_size=10`（= 每任务 eps 数）+ 每任务 10 episodes。

**统计语义**：LIBERO 初始状态被 `init_state_id = episode_index` 钉死（[libero.py:175](lerobot/src/lerobot/envs/libero.py#L175)）——**换 `--seed` 只改策略噪声与 env RNG，不改初始条件**。因此：
- 多个 eval seed 是**相关重复**：**只报逐 seed 的 n=10 Wilson CI**，不得合并成 n=30；
- headline 协议档 **2 个 eval seed**（800 eps/臂），OOD 评测 1 个 eval seed，曲线评测 1 seed；
- **init-state 方差现有 CLI 不可测**（§12 已知限制）。训练 seed ≥2 是硬要求。

## 3. 锁定资源

| 项 | 值 |
|---|---|
| 代码 | `pip install "lerobot[smolvla,libero] @ git+https://github.com/huggingface/lerobot.git@146d8474932c8b566717eb2738ba459337ab71c4"`（系统环境 pip 直装，不用 venv） |
| 主数据集 | **`lerobot/libero`**（1693 eps / 40 任务，v3.0，1.9GB）+ `--dataset.revision=<M0 记录 sha>`；**不写 `--dataset.video_backend`** |
| LIBERO 资产 | `lerobot/libero-assets` + `~/.libero/config.yaml`（任何 `import libero` 之前；每会话重跑） |
| 模型 | `lerobot/smolvla_base`（450M；主臂初始化）。`--policy.compile_model=true` 可用（编译耗时 M0 实测后定）；`--accelerator.compile` 是占位符，禁用 |
| 扰动评测 | **自写 wrapper（主方案，§8）**；条件项 = libero_plus（6.4GB 资产 + root apt + 卸载 hf-libero，与标准 env 互斥） |

## 4. Kaggle 硬件、配额与 Secrets

- T4 16GB ×2；双卡需 bs8×accum4 或 bs16×accum2 维持 eff-64；FSDP 与 PEFT 互斥。M0 实测后定。
- **Secrets 注入（hub checkpoint 主机制的前提）**：Kaggle Secrets 不是环境变量，`!bash`/SSH shell 都拿不到。`!bash run_*.sh` 之前的 Python cell：
  ```python
  from kaggle_secrets import UserSecretsClient
  import os
  os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
  os.environ["HF_USER"] = "<你的Kaggle用户名>"
  # 同时写一份给 SSH shell 用（SSH 不继承 kernel 环境变量，见 KAGGLE_SSH §1.4）：
  from pathlib import Path
  p = Path("/tmp/hf_token"); p.write_text(os.environ["HF_TOKEN"]); p.chmod(0o600)
  # !env | grep -i hf_   # M0 当场确认子进程可见
  ```
  `push_checkpoint_to_hub` 走 `huggingface_hub.HfApi()`，只认 `HF_TOKEN` 环境变量或 `~/.cache/huggingface/token`。tailscale 的 `TS_AUTH_KEY` 同理注入（KAGGLE_SSH §1.3）。
- **跨会话续训（主机制）**：`--save_checkpoint_to_hub=true --policy.private=true --policy.repo_id=${HF_USER}/smolvla-${EXP_NAME}`。**`--policy.private=true` 必须（v6）**：`private=None` 默认建公开 repo（[train_utils.py:562](lerobot/src/lerobot/common/train_utils.py#L562)："public unless the organization's default is private"），checkpoint + 全部超参 + 数据 revision 都在里面。**repo_id 每 run 唯一**（`checkpoints/{目录名}/` 同 step 号互相覆盖）；**只有跨会话的主跑开此开关**，短 run 关掉，省 10.4GB/run 上传与每次 save 2–9 min 墙钟。
- **/kaggle/working 20GB 分账（v6 修正策略列）**：

  | 内容 | 大小 | 策略 |
  |---|---|---|
  | 主跑本地 ckpt（4 × ~2.6GB） | **~10.4GB 峰值** | **后台模式无交互 shell，无法即时清理**（save→push 全程无本地删除逻辑）——按峰值规划；若 M0 实测吃紧改 `--save_freq=16000`（2 个 ≈5.2GB） |
  | pip wheelhouse（见下） | 2–3GB | 持久 |
  | 插件包源码 lerobot_env_svla/ | <1MB | 持久（`pip install -e` 每会话重跑，§8） |
  | 评测输出/视频 | <1GB | recording 默认关，开时录完即转 |
  | tailscaled state/socket/日志、/tmp/hf_token | — | /tmp，不占 working |

  合计 ~14–15GB < 20GB，余量 ~5GB——任何 `--eval.recording=true` 前先 `du -sh` 自查。
  - **pip 缓存两选一（v6 修正机制）**：①只设 `PIP_CACHE_DIR=/kaggle/working/pipcache`（pip 自动复用下载缓存，省下载不省解析/构建）；②真离线秒装：先 `pip wheel -r reqs.txt -w /kaggle/working/wheelhouse`，再 `--no-index --find-links=/kaggle/working/wheelhouse`。**`PIP_CACHE_DIR` 不是 `--find-links` 能消费的扁平目录**。git+URL 需求缓存收益有限，M0 实测每会话安装墙钟后决定占不占这 2–3GB。
- 长命令一律 `.sh` + `!bash`；每个 .sh 头部 `export MUJOCO_GL=egl`。会话机制见 [KAGGLE_SSH.md](KAGGLE_SSH.md) §3。

## 5. 全项目预算与降范围规则（v6：补基线 seed#2 评测）

记 U = 实测 s/update（预期 2.88–6.12），E = 实测协议档 s/episode（batched）。

| # | 项 | 墙钟 | GPU 时 |
|---|---|---|---|
| 1 | M0 | 2–4h（含下载） | 1–2h |
| 2 | M1 sanity 2k updates | 1 会话（含安装 ~1h） | 2k×U = 1.6–3.4h |
| 3 | M1 主跑 8k updates（hub ckpt 开，含 ~10.4GB 上传 0.5–1.5h） | 1–2 会话（按 U） | 8k×U = 6.4–13.6h |
| 3b | 基线第二训练 seed 8k updates（M3 统计前提） | 1–2 会话（按 U） | 6.4–13.6h |
| 4 | M3 训练：筛选 2 臂 × 1 seed × 4k + 胜者 × 2 seeds × 8k | 2–3 会话 | **19.2–40.8h** |
| 4b | M3 评测：筛选 2 ×（spatial 100 + OOD 3 代表配置 × 100）= 800 eps；胜者 2 ×（协议 800 + OOD 1000）= 3600 eps → **4400 eps × E** | E=25 → ≤30.6h；E=10 → 12.2h | 同墙钟 |
| 5 | **基线 seed#1** 协议档评测：曲线 2 ckpt × spatial × 1 seed（200）+ 最终 4 套件 × 2 eval seeds（800）= **1000 eps × E** | ≤6.9h @E=25 | 同墙钟 |
| **5b** | **基线 seed#2 协议档 + OOD（v6 新增）：800 + 1000 = 1800 eps × E** | ≤12.5h @E=25 | 同墙钟 |
| 6 | **基线 seed#1** OOD 全矩阵：10 配置 × spatial 100 eps = **1000 eps × E** | ≤6.9h @E=25 | 同墙钟 |
| **核心合计** | | **墙钟 ≈ 62–140h** | **GPU ≈ 57–130h ≈ 1.9–4.3 周配额（30h/周）**；日历排期按 ×1.3 调试余量 ≈ **2.5–5.5 周**（可选臂另计） |

**M0 实测绑定降范围规则（单位：s/update；等效 s/mb = ÷4）**
- U ≤ 3.0：按表默认；主跑单会话。
- 3.0 < U ≤ 6.0：主跑/基线第二 seed 拆 2 会话（hub ckpt 续训）。
- **U > 6.0（v6：不再静默放弃 seed≥2）**：主跑降 5k、基线第二 seed 降 5k（**保留**）、筛选臂 3k × 1 seed、胜者 5k × 2 seeds → 训练 26k updates（U=6–8 时 43.3–57.8h）；评测按 E 规则收紧（E>25 → 协议档 1 eval seed）：4b→3600、5→600、5b→600、6→1000 = 5800 eps，E=25 → 40.3h；分支核心 ≈ **85–102h**。若 U、E 双高仍超配额，**唯一进一步降级 = 放弃基线第二 seed（训练 seed≥2 破例），结论显式降级为"单 seed 探索性、不做臂间显著性声明"并写进报告**——不允许静默。
- E ≤ 25 s/ep：headline 2 eval seeds；25 < E ≤ 60：协议档降 1 eval seed；E > 60：最终只跑 spatial+object 两套件。
- **对已开始的 run 不适用**（resume 不重算 cosine）。

## 6. M0 冒烟测试（会话 1；墙钟 2–4h）

顺序执行，失败即停：

0. **Secrets 注入 cell**（§4，含 /tmp/hf_token）+ `!env | grep -i hf_` 确认。
1. 安装锁 commit（记录每会话安装墙钟，决定 wheelhouse 是否值得）；`!nvidia-smi`。
2. libero 资产 cell（记录体积与下载墙钟）。
3. 记录 `lerobot/libero` revision sha、`codebase_version==v3.0`；查 `svla_so100_pickplace` 版本；打印 `get_benchmark_dict()`。
4. e2e 冒烟：`--steps=20 --policy.scheduler_warmup_steps=10 --policy.scheduler_decay_steps=20 --save_freq=0`，跑完清理 last ckpt；feature 冲突则加 `--policy.input_features=null --policy.output_features=null`。
5. **吞吐实测**：bs16×accum4 fp16 `--steps=100`；`step_s`×4 与 `64/samples_per_s` 双校验；同法测双卡与开增强；测 `--num_workers=2`；**套用 §5 规则**。
6. **eval 实测**：协议档 unbatched s/ep；batched async/sync 各一遍；快速档整评测单次耗时（预期 6–13 min）。smolvla_base 直评 SR 无意义，只看延迟。
7. checkpoint 实际大小 → 定 `save_freq`；`du -sh ~/.cache/huggingface; df -h` 确认缓存归属盘。

## 7. M1 基线复现（3–5 个会话）

`run_m1.sh`（v6：+private=true）：
```bash
#!/bin/bash
export MUJOCO_GL=egl
export HF_USER=<你的Kaggle用户名>          # 与 Secrets 注入 cell 一致；SSH 路径需先 source /tmp/hf_token（KAGGLE_SSH §1.4）
export EXP_NAME=m1_base_libero             # 每 run 唯一
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.private=true \
  --policy.repo_id=${HF_USER}/smolvla-${EXP_NAME} \
  --save_checkpoint_to_hub=true \
  --dataset.repo_id=lerobot/libero \
  --dataset.revision=<sha> \
  --batch_size=16 \
  --num_workers=2 \
  --accelerator.mixed_precision=fp16 \
  --accelerator.gradient_accumulation.steps=4 \
  --steps=32000 \
  --policy.scheduler_warmup_steps=2000 \
  --policy.scheduler_decay_steps=32000 \
  --save_freq=8000 \
  --seed=1000 \
  --env.type=libero --env.task=libero_spatial \
  --env.observation_height=256 --env.observation_width=256 \
  --env.init_states=true --env.hard_reset=true \
  --env_eval_freq=8000 \
  --eval.n_episodes=10 --eval.batch_size=2 --eval.use_async_envs=false \
  --output_dir=/kaggle/working/outputs/${EXP_NAME} \
  --job_name=${EXP_NAME}
```
- 续训：`--resume=true --config_path=${HF_USER}/smolvla-${EXP_NAME}`（参数化；不改 --steps）。
- sanity 与 M3 筛选 run：`--save_checkpoint_to_hub=false`（无需 private/repo_id）；主跑与基线第二 seed：开。

**协议档评测两个变体（都列出，按用途取用）**：
```bash
# 变体 A（曲线点，每 ckpt）：只跑 spatial、1 eval seed
lerobot-eval \
  --policy.path=<ckpt> --policy.n_action_steps=1 \
  --env.type=libero --env.task=libero_spatial \
  --env.observation_height=256 --env.observation_width=256 \
  --env.init_states=true --env.hard_reset=true --env.max_parallel_tasks=1 \
  --eval.batch_size=10 --eval.n_episodes=10 --eval.use_async_envs=<M0 结论> \
  --seed=0 \
  --output_dir=/kaggle/working/outputs/eval_${EXP_NAME}_curve_<step>

# 变体 B（headline，最终 ckpt）：4 套件、2 eval seeds（--seed=0 与 --seed=1 各跑一次）
lerobot-eval \
  --policy.path=<ckpt> --policy.n_action_steps=1 \
  --env.type=libero --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.observation_height=256 --env.observation_width=256 \
  --env.init_states=true --env.hard_reset=true --env.max_parallel_tasks=1 \
  --eval.batch_size=10 --eval.n_episodes=10 --eval.use_async_envs=<M0 结论> \
  --seed=<0/1> \
  --output_dir=/kaggle/working/outputs/eval_${EXP_NAME}_final_seed<0/1>
```

**产出**：逐 seed 协议档 SR vs updates 曲线 + 快速档曲线 + 吞吐/显存记录。

## 8. M2 泛化评测矩阵（wrapper 开发 0.5–1 会话 + 评测 1–2 会话）

**wrapper 注入路径（本地插件包，写死）**：在 /kaggle/working 建 `lerobot_env_svla/`，三条落地纪律：
1. **发行名 == 可导入模块名，且以 `lerobot_env_` 开头**：pyproject `[project] name = "lerobot_env_svla"` 且目录 `lerobot_env_svla/__init__.py`，注册装饰器在 `__init__.py` 顶层执行——扫描按发行名前缀匹配后 `import_module(发行名)`（[import_utils.py:274-281](lerobot/src/lerobot/utils/import_utils.py#L274-L281)）。
2. **导入失败是静默的**（`logging.exception` 吞掉后继续）：每次评测前确认日志 `Imported third-party plugin: lerobot_env_svla`，否则 `--env.type=libero_perturb` 只报"未知 choice"无根因（§12 检查项）。
3. **`pip install -e /kaggle/working/lerobot_env_svla` 每会话重跑**（editable 安装记录在 site-packages，不持久）。
   子类覆写 `get_env_processors()` 在 `LiberoProcessorStep()` **之后**追加扰动 step；`register_third_party_plugins()` 在 `lerobot_eval.py` main() 第一句执行 → `--env.type=libero_perturb` 可直接解析。

**同源算子（硬要求）**：直接实例化 `lerobot.transforms` 的 `Batched*` 类（普通 nn.Module）——训练端与评测端字面同一算子类。顺序：扰动放在 180° 翻转之后（颜色/噪声类不受影响，RandomShadow 仅几何镜像、无害，写入代码注释）。

| 轴 | 分布内 | 分布外 | 实现 |
|---|---|---|---|
| 套件内细分 | 4 套件协议档总表 | 无（zero-task 轴维持放弃） | `--env.task_ids` |
| 视觉/物理扰动 OOD | 标准渲染帧 | **筛选：3 代表配置（色温+噪声+遮挡）× spatial 100 eps × 1 eval seed；胜者：全 10 配置 × 100 eps** | wrapper（规模进 §5 item 4b） |
| so100 位置（可选） | 4/5 位置训练 | holdout 第 5 位置 | 独立微调臂 8k updates + 离线 MSE，标注无真机局限 |

失败回放：最终对比轮可选 `--eval.recording=true`（先 `du -sh` 自查 working 余量）。

## 9. M3 改进实验（核心）

**`--dataset.image_transforms.tfs` 赋值是整体替换默认字典**（[molmoact2.mdx:90](lerobot/docs/source/molmoact2.mdx#L90) 官方 idiom；点路径只对默认 6 键有效；不想要的变换不写即可）。

| 臂 | tfs JSON（整体替换） | 规模 |
|---|---|---|
| 基线 | M1 主跑 + 第二 seed（不重跑） | 8k × 2 seeds |
| A-光照/噪声 | `--dataset.image_transforms.tfs='{"planckian":{"type":"PlanckianJitter","kwargs":{"temperature":[4000,12000]}},"gamma":{"type":"GammaCorrection","kwargs":{"gamma":[0.6,1.7]}},"shadow":{"type":"RandomShadow","kwargs":{"opacity":[0.3,0.6]}},"patch":{"type":"GaussianPatchBrightness","kwargs":{"num_patches":[1,4],"sigma_range":[0.05,0.25],"factor_range":[0.4,1.6]}},"noise":{"type":"GaussianNoise","kwargs":{"std":[5.0,25.0]}}}'`（**5 项中 3 项为库默认**（Shadow/Patch/Noise），Planckian 与 Gamma **主动收窄**至 [4000,12000]/[0.6,1.7]，库默认 (3000,15000)/(0.5,2.0)；越界直接 ValueError：Planckian 限 3000–15000、Gamma 0<min≤max、Shadow 0≤min≤max≤1） | 先 4k × 1 seed 筛选，胜者 8k × 2 seeds |
| A-几何/遮挡 | `--dataset.image_transforms.tfs='{"affine":{"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}},"rot":{"type":"RandomRotation","kwargs":{"degrees":[-5.0,5.0]}},"dropout":{"type":"CoarseDropout","kwargs":{"max_holes":8,"max_height_frac":0.07,"max_width_frac":0.07}}}'`（**v6 补 RandomRotation 凑满 3 个候选**：原 2 候选时 `n_subset=min(2,3)=2` 每样本恒施 2 个变换，与光照臂的 3 个形成 50% 密度混淆——两臂现同为 3 候选 × 3 施加。CoarseDropout 参数是标量不是区间，强度分档靠改 max_holes/frac 两档配置） | 同上 |

- 所有 M3 run 的完整公共前缀（flag 写全，不许简写）：
  ```bash
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.tfs='<上表 JSON>' \
  --policy.scheduler_warmup_steps=<≈steps> --policy.scheduler_decay_steps=<=steps> \
  --policy.repo_id=${HF_USER}/smolvla-${EXP_NAME} --policy.private=true
  ```
- 假设：OOD（§8 wrapper 扰动轴）成功率↑，分布内不降。
- 判据：协议档 + **训练 seed ≥2**，逐 seed n=10 Wilson CI，区间重叠 = "未见差异"；同数据 revision、同 eval seeds；**两臂增强密度对齐（3 候选 × 3 施加）**，密度作为设计变量写进报告。
- M0 / 每 run 开跑前打印生效 transform 池复核 JSON 落地。

**备选 B（推理端，零训练）**：`num_steps` 10→5/2 在协议档下测成功率与延迟；`n_action_steps`>1 / chunk 重叠属吞吐变体，标注与论文协议不可比、单独报告（RTC 仅提一句）。

## 10. M4 整理（1–2 天）

- 目录：`outputs/{EXP_NAME}/` + `train_config.json` + metrics CSV + 评测 JSON（逐 seed + ci95）+ `run_*.sh`。
- **登记表**：EXP_NAME | hub repo_id（private） | init | updates | 数据 revision | 训练 seed | eval seeds | GPU 时 | 分布内 SR+CI（逐 seed） | OOD SR+CI | 结论。
- 并列展示表（不做可比性声明）：我们主臂 vs 论文 Table 2，逐列标差异。
- 失败分析 + 扰动矩阵交叉解读 + 实际成本（GPU 时/墙钟/会话数/Hub 占用）。

## 11. 风险与对策

| 风险 | 对策 |
|---|---|
| `import libero` 挂死 | 资产 cell 先行；重启后重跑 |
| 12h 会话被硬切 | hub ckpt 主机制（仅长 run 开）+ Secrets 注入 cell；短 run 单会话内完成 |
| Secrets 未注入（notebook 路径） | M0 第 0 步 `!env | grep -i hf_` 当场验证 |
| **Secrets 不进 SSH shell（跨文档）** | Python cell 写 `/tmp/hf_token`（chmod 600）→ .sh 头部 `export HF_TOKEN=$(cat /tmp/hf_token)`；或 SSH shell 手动 export（KAGGLE_SSH §1.4） |
| **hub repo 默认公开** | 所有开 `save_checkpoint_to_hub` 的 run 加 `--policy.private=true`（private=None 默认 public） |
| **后台模式无法清理本地 ckpt** | 按 10.4GB 峰值规划 working；吃紧改 `--save_freq=16000` |
| fp16 数值不稳 | 盯日志 `grad_scale`/`grad_norm`（fp16 下必有）→ 降 lr → 退 fp32（PEFT 是显存手段） |
| 显存不足 | 首选 PEFT/LoRA（与 FSDP 互斥）；fp32+降 bs 仅最后尝试 |
| 双卡有效 batch 翻倍 | bs8×accum4 或 bs16×accum2 |
| torchcodec 加载失败 | 不硬编码 backend，自动回退 pyav |
| 插件静默导入失败 | 每次评测前查日志 `Imported third-party plugin` |
| checkpoint 串台/覆盖 | repo_id 按 EXP_NAME 唯一；resume 不改 --steps |
| 增强配置静默失效 / 密度混淆 | tfs 整体替换 JSON + 开跑前打印生效池；两臂 3 候选 × 3 施加对齐 |
| libero_plus 不可行 | 主方案 wrapper 无新依赖 |
| 版本漂移 | 代码钉 commit；数据钉 revision；tailscale 二进制钉版本；进登记表 |

## 12. 开放项（M0 现场确认后回填）

- [ ] **s/update 双校验**：`step_s`×4 与 `64/samples_per_s` 互核
- [ ] s/update（单卡/双卡 × 开/关增强 × num_workers 2/4）→ 敲定 §5 规则分支
- [ ] 协议档 s/ep（unbatched / batched-async / batched-sync）+ 快速档单次耗时（预期 6–13 min）
- [ ] `lerobot/libero` revision sha；`svla_so100_pickplace` codebase_version
- [ ] `lerobot/libero-assets` 体积与下载墙钟；每会话 pip 安装墙钟（决定 wheelhouse 取舍）
- [ ] `get_benchmark_dict()` 4 套件覆盖完整性
- [ ] features 继承是否冲突（M0 第 4 步）
- [ ] checkpoint 实际大小 → 定 save_freq（默认 8000 mb / 吃紧 16000）；Hub 单次上传墙钟与总占用
- [ ] tfs JSON 落地复核：打印生效池（无 Identity 混入；两臂各 3 候选）
- [ ] 插件包导入日志确认：`Imported third-party plugin: lerobot_env_svla`
- [ ] Secrets 注入验证（notebook `!env` + SSH shell `echo` 双路径）
- [ ] 已知限制（不测，报告注明）：init-state 方差现有 CLI 不可测；eval seeds 只覆盖策略随机性
