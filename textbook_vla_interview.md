# SmolVLA × LIBERO 免费全链路复现 — VLA 岗位面试教科书

> 素材全部来自本项目真实运行数据。每个数字都能在 [GitHub 仓库](https://github.com/magickkdd/smolvla-libero-repro) 的日志与产物中溯源。

---

## 序章 · 3 分钟项目陈述（电梯稿，背下来）

> "我在零现金成本约束下，在免费的 Kaggle T4×2 上完整复现了 HuggingFace 的 SmolVLA（450M 参数机器人操作基础模型）在 LIBERO 仿真基准上的微调：8k updates 有效批量 64 训练 4.8 小时，LIBERO Spatial **10 任务 × 10 episodes 协议评测成功率 74%**（Wilson 95% CI [63.0%, 83.2%]），论文同协议数字是 90% 但用了 25 倍训练预算——考虑到我们只训练了 1.9 个 epoch、单套件任务，这个差距符合缩放规律。全链路包括：数据格式迁移、双卡 DDP 训练（比单卡提速 1.9 倍）、无人值守监控架构（ntfy 信标 + 前台进程 + 断点自动接力）、协议对齐的闭环评测（逐步重规划）。目前 16k updates 的阶梯点已训练完成待评测，后续计划是 32k 阶梯、LoRA 对照和多套件联合训练。"

**面试官心里的问题**（本教科书逐章回答）：这人是真跑过还是背的？→ 每个数字带日志级细节；遇到故障怎么处理？→ 第五章真实战例；懂不懂底层？→ 第一、二章。

---

## 第一章 · 项目架构、核心模块与代码结构

### 1.1 俯瞰：数据流一条线

```
lerobot/libero v3.0 (1.9GB, 1693 episodes, 273,465 帧)
   │  LeRobotDataset v3.0: parquet(状态/动作) + MP4(双相机视频) + meta/
   ▼
数据管线（每卡 2 worker, prefetch 4）
   │  torchcodec 解码 256×256 ×2 相机 │ 状态 MEAN_STD 归一化 │ 指令 tokenize 48 token
   ▼
SmolVLA 450M（冻结 VLM 骨干 ~350M + 可训练动作专家 ~100M）
   │  SmolVLM2 500M 截至第 16 层 → 视觉+语言特征
   │  动作专家(6 层 decoder, 交叉注意力读视觉特征) → flow matching 去噪 → 50 步动作块
   ▼
损失 = flow matching MSE（噪声速度场回归）→ AdamW(1e-4) + fp16 GradScaler + DDP all-reduce
   ▼
checkpoint（每 1000 步, pretrained_model/ + training_state/）→ ntfy 信标 → HF 私有仓库
   ▼
lerobot-eval（LIBERO Spatial 10 任务 × 10 ep, n_action_steps=1 逐步重规划, 280 步上限）
   ▼
eval_info.json → 成功率 74% [63.0, 83.2]
```

### 1.2 模型架构考点

| 模块 | 参数量 | 是否训练 | 面试要点 |
|---|---|---|---|
| SmolVLM2 骨干（视觉编码器 + 前 16 层 LLM） | ~350M | ❄️ 冻结 | 保留语言/视觉先验；小数据微调防灾难性遗忘；`freeze_vision_encoder` + `train_expert_only` |
| 动作专家（action expert） | ~100M | ✅ 全量 | 6 层 decoder，与骨干**交替交叉注意力**（非拼接）：视觉/语言 KV 来自骨干，query 来自动作 token |
| 总计 | 450M | 100M 可训练 | `num_learnable_params=99,880,992`（日志原值） |

**必考追问：为什么动作专家用交叉注意力而不是把动作 token 拼进骨干上下文？**
答：① 冻结骨干的 KV 只算一次可跨去噪步复用，推理省算力（SmolVLA 论文明确动机）；② 状态/动作分布与语言分布差异大，独立专家避免污染语言表征；③ 参数效率——只训 100M。

**必考追问：VLM 骨干为什么截到 16 层？**
答：SmolVLA 论文发现小型 VLA 只需骨干的前几层语义即可支撑操作任务（层数越深语言抽象越强但空间细节损失）；SmolVLM2-500M 全量 36 层，取 16 层是精度/速度的论文消融结论。

### 1.3 仓库结构（讲"工程演进"的素材）

```
smolvla-libero-repro/
├── kaggle_bootstrap.ipynb          v9  极简单卡版（基础设施参考实现）
├── kaggle_bootstrap_v10_speed.ipynb v10 单卡 bs32 对照（证明 T4 在 bs16 已算力饱和）
├── kaggle_bootstrap_v11_ddp.ipynb   v11 双卡 DDP 最终训练版（2.16 s/step, 1.9×）
├── kaggle_bootstrap_v12_eval.ipynb  v12 协议评测（8k 点：74%）
├── kaggle_bootstrap_v13_resume16k.ipynb v13→v19 接力续训（16k 点）
├── colab_eval16k.ipynb              v21 Colab 评测（按任务断点续测）
├── PLAN.md                          三阶段计划 + 差异归因表
├── results/M2_eval_libero_spatial.md 8k 评测完整报告
├── paper/SmolVLA_paper.md           论文笔记
└── logs/                            训练/评测日志归档
```

**面试叙事点**：notebook 的世代演进本身就是工程决策记录（ADR）——每个版本对应一次真实故障或实验假设，能讲出"为什么改"比"改了什么"值钱。

---

## 第二章 · 关键技术原理、算法流程与实现细节

### 2.1 Flow Matching 训练目标（核心算法，必须能写公式）

**一句话**：学习一个速度场 v_θ(x_t, t)，把高斯噪声 x_1 沿直线路径输运到动作分布 x_0。

```
训练:  x_t = (1−t)·x_0 + t·ε,  ε~N(0,I),  t~U[0,1]
       损失 = MSE(v_θ(x_t, t), ε − x_0)        # 回归条件速度场
推理:  x ← x + v_θ(x, t)·dt,  迭代 K=10 步（SmolVLA 默认 num_diffusion_steps=10）
```

**与扩散对比（必考）**：
| | DDPM | DDIM | Flow Matching |
|---|---|---|---|
| 路径 | 随机马尔可夫链 | 确定性反向 | 直线输运 |
| 采样步数 | 50-1000 | 10-50 | **10** |
| 训练目标 | 噪声 ε 预测 | x0/v 预测 | 速度场回归 |
| 机器人适用性 | 慢 | 中 | 快（实时控制刚需） |

**追问：为什么机器人策略偏爱 flow matching / 扩散类头而不用回归？**
答：演示数据天然多模态（同一个任务有多种合理动作轨迹），L2 回归会回归到轨迹均值（不在数据流形上，表现为抖动/犹豫）；生成式头建模完整条件分布。

### 2.2 动作块（Action Chunking）与分层推理

- 策略每次推理输出 **50 步动作块**（horizon=50），训练时随机取其中 `chunk_size` 段算损失
- **闭环执行有三种重规划粒度**（我们实测过两种）：
  1. `n_action_steps=50`：每 50 步推理一次，中间开环执行（快，~0.06s/环境步均摊）
  2. `n_action_steps=1`：**每环境步重新推理**（论文闭环评测协议，与我们复现一致）——推理开销 ×50，单集 ~2 分钟
- SmolVLA 论文的**异步推理堆叠（APS）**：开环执行旧块的同时 GPU 异步算新块，队列头插入——我们 M2 用同步逐步重规划（保守对齐协议），APS 是面试可讲的改进方向

### 2.3 视觉处理细节（论文级考点）

- **FFW（First-Frame-World）空间裁剪增强**：训练时把"裁剪"施加在视觉观测 + 动作轨迹上（等效于相机平移），一条数据变多种相机视角——小数据泛化的关键，SmolVLA 消融显示贡献显著
- 双相机：agentview（第三人称）+ 腕部相机，各 256×256 → patch embed 后共 64+ 视觉 token 量级
- **跨本体训练**：状态/动作维度用掩码 + per-robot 归一化，使单一策略兼容不同机器人——面试聊"泛化"必提

### 2.4 训练配置全账本（真实值，背下来）

| 项 | 值 | 日志字段 |
|---|---|---|
| 优化器 | AdamW lr=1e-4, betas=(0.9,0.95), wd=1e-10 | `optimizer` dump |
| 梯度裁剪 | max_norm=10（实测 grdn 0.9-3.5，远低阈值） | `grdn` |
| LR 调度 | warmup 500 → cosine → 2.5e-6（自动等比缩放 2000/32000→500/8000，比例 6.25% 不变） | `Auto-scaling LR scheduler` |
| 精度 | fp16 AMP + 动态 GradScaler（scale 65536 波动） | `scale` |
| 有效批量 | 32/卡 × 2 GPU × 1 accum = **64** | `Effective batch size` |
| 训练量 | 8000 updates × 64 = 512k 样本 ≈ 1.9 epoch | `epch:0.05→1.9` |
| 步时分解 | data 0.003s + prep 0.008s + updt 2.126s = **2.14s** | `step_s/updt_s/data_s` |

**追问：lr 为什么 1e-4 这么小？** 冻结骨干只训 100M 专家 + 大 eff-bs 64 + 预训练权重敏感 → 官方 preset 保守值。
**追问：wd=1e-10 约等于 0？** 是的，预训练微调惯例——weight decay 会把预训练权重往 0 拉，破坏先验。

### 2.5 checkpoint 结构与续训语义（本项目独有深坑，第五章展开）

```
checkpoints/008000/
├── pretrained_model/          # 推理用：config.json + model.safetensors
│   ├── policy_preprocessor.json / policy_postprocessor.json
│   ├── *_normalizer_processor.safetensors（数据集统计量）
│   └── tokenizer/（子目录！抢救时用 isfile 过滤会漏掉——血泪教训）
└── training_state/            # 严格续训用：optimizer_state.pt / rng_state /
                               # training_state.json（step/调度器/采样器偏移/scaler）
```

**三种"继续训练"的语义区分（我们真实踩过）**：
1. **严格 resume**（`--resume=true`）：全套状态恢复，LR 从衰减谷底继续 → 几乎不再学习
2. **热启动**（我们 16k 用）：只载权重，优化器动量/调度器重置 → 第二轮 cosine 重新升温衰减 = **标准延长训练配方**；严格 resume 反而学不动（谷底 lr 2.5e-6）
3. 从零重训：损失从 1.14 起步 vs 热启动 0.40 起步——日志一眼可辨

---

## 第三章 · 数据处理、训练与推理全链路

### 3.1 数据集工程（选型即考点）

| 候选 | 体积/格式 | 结论 |
|---|---|---|
| **lerobot/libero v3.0** ✅ | 1.9GB，LeRobotDataset v3.0（parquet+MP4+meta，revision `a1aaacb7` 锁定） | 唯一同时满足格式兼容 + 磁盘 + 官方复现文档 |
| physical-intelligence/libero | 疑似 v2.1 旧格式，需迁移脚本 | 格式风险 |
| HuggingFaceVLA/libero | 69.9GB PNG-in-parquet | Kaggle 物理装不下 |

**格式考点**：v3.0 = 数据(状态/动作 parquet) + 视频分离(MP4, 按需解码省空间) + meta/info.json(episode 索引/fps/任务)——对比 v2.1 的全联合 pickle，视频编解码可移植性和流式读取是核心改进。

### 3.2 评测协议（M2 的每一项都有出处）

```
lerobot-eval --env.type=libero --env.task=libero_spatial --env.task_ids='[N]'   # 单任务断点续测
  --env.observation_height=256 --env.observation_width=256     # 渲染分辨率对齐训练
  --env.init_states=true --env.hard_reset=true                 # 固定初始态，可复现
  --policy.n_action_steps=1                                    # 逐步重规划（论文协议！默认 50 不可比）
  --eval.n_episodes=10 --eval.batch_size=1                     # 每任务 10 集，闭环无并行
```

- 输出：`eval_info.json`（per_task successes 明细 + 视频路径）→ 上传 HF 归档
- **统计口径**：10 任务 × 10 = 100 episodes，74/100 → Wilson 95% CI **[63.0%, 83.2%]**（面试主动报区间，暴露"10 集测出 90%±60pp"的都不可信）

### 3.3 推理链路（延迟数字要能报）

逐步重规划时每环境步：观测构造 → VLM 前向（KV 缓存跨去噪步复用）→ 专家 10 步去噪 → 取动作[0] 执行 → ~0.4-0.5s/步（T4 fp32 batch1）。SmolVLA 论文同架构在消费 GPU 上 ~280ms（fp16 + APS）——我们评测为协议保真未开 AMP 推理，是已知可优化项。

---

## 第四章 · 主流 VLA 方案对比与项目亮点

### 4.1 横向对比表（面试白板题：把这张表画出来）

| 方案 | 总参数 | 骨干 | 动作头 | 动作表示 | 训练成本 | 推理延迟 |
|---|---|---|---|---|---|---|
| RT-2 (Google) | 55B | PaLI-X | 离散 token | 文本化 | 闭源 | ~3Hz |
| **OpenVLA** | 7B | Llama2+DINOv2/SigLIP | 离散 token (256 bins) | 单步 | 多卡 A100 | ~3-6Hz |
| **π0 (PI)** | 3.3B | PaliGemma | flow matching | 连续块 | A100 级 | ~50Hz 计算 |
| **GR00T-N1 (NVIDIA)** | 2B | Eagle/VLM | 扩散头+双系统 | 连续块 | 数据中心 | 中 |
| **RDT-1B** | 1.2B | T5+SigLIP | 扩散 (DDPM) | 连续块 | 高 | 慢 |
| **Octo** | 93M | 自研 transformer | 扩散头 | 连续块 | 低 | 快 |
| **SmolVLA ✅** | **450M** | SmolVLM2(截16层) | **flow matching** | 连续块(50) | **T4×2 可训** | **~280ms** |

**提炼三句话定位**：SmolVLA 的生态位 = ①参数最小但吸了社区级数据规模（-agibot 460k episodes 跨本体预训练）；②推理延迟为实时控制设计（ vs OpenVLA 的 3Hz）；③训练预算平民化（我们 T4×2 4.8h 的实证）。

### 4.2 差异归因表（74% vs 90%，面试最加分的部分）

| 归因项 | 论文 | 本复现 | 预期影响 |
|---|---|---|---|
| 训练步数 | 200k updates | **8k updates**（1/25） | 主要差距来源 |
| 训练数据 | 单套件亦可 | libero_spatial 单套件（同） | 单套件任务集中度抵消部分步数劣势 |
| 精度 | bf16 | fp16+GradScaler（T4 无 bf16） | 小 |
| 推理重规划 | 逐步（同） | 逐步（同） | 已对齐 |
| 评测协议 | Spatial 10×10+ | 完全一致 | 已对齐 |
| 随机性 | — | CI ±8-10pp | 74 vs 90 在 CI 外，差距真实但方向可解释 |

**标准答句**："差距主项是 1/25 的训练预算，这不是复现失败而是缩放曲线的一部分——我们正在用 16k/32k 阶梯点验证这条曲线。"（16k 已训练完成，评测排队中）

### 4.3 项目亮点（自我营销弹药）

1. **零成本全链路**：数据→训练→评测→产物归档全免费平台完成，任何一家公司内部集群都能无缝复刻方法
2. **1.9× DDP 提速的实证方法**：先用 bs32 单卡对照证明 T4 在 bs16 已饱和（完美线性 2.03s/mb），再上 torchrun DDP——"先证饱和再并行"的方法论
3. **无人值守实验架构**：前台进程 + traceback 直进日志（消灭静默失败）+ ntfy 信标（每 90 秒步级遥测）+ checkpoint 落盘即推 HF——配额死亡/容器回收零损失接力
4. **协议保真意识**：主动把训练内评测从 n_action_steps=50 改为 1，拒绝不可比数字

---

## 第五章 · 面试追问的底层细节与应对（含真实战例）

### 5.1 并行训练（本项目的实物证据链）

- **DDP**：每卡完整模型副本，`torchrun` 拉起 rank0/rank1，启动广播一次参数，之后每步只同步梯度。日志实证：`rank0/rank1`、`Effective batch size: 32 x 2 dp workers`
- **all-reduce & ring**：梯度 400MB/步（100M fp32），PCIe ~12GB/s ≈ 35ms ≈ 步时 1.6%——所以双卡 1.9× 加速成立。ring 算法每卡通信量 2(N−1)/N·Ψ 与卡数无关
- **选型决策树**（背这个）：模型状态 16Ψ bytes（fp16 参数2+梯度2+fp32 master 4+Adam m/v 8）单卡放得下？→DDP。放不下单算子放得下？→ZeRO-2/3（FSDP，+50% 通信）。单算子都爆？→TP（NVLink 域内）＋跨机 PP＋MoE EP＋长序列 CP
- **ZeRO 三级**：1=切优化器状态(8Ψ 大头)；2=+梯度；3=+参数（前向/反向临时 all-gather，通信 3Ψ = +50%）。FSDP=ZeRO-3 的 PyTorch 实现，lerobot `--parallelism.dp_shard`
- **本项目为什么不用 TP**：T4 走 PCIe 集合通信太慢 + 450M 单算子放得下——TP 解决的问题不存在
- **防混淆金句**："SmolVLA 的 action expert 是单个交叉注意力网络，不是 MoE，和专家并行无关"

### 5.2 数值精度

- fp16：5 位指数（下溢→梯度消失）+10 位尾数（舍入→更新被吞）。两个死法两个药：**GradScaler**（loss×scale 抬梯度，inf/NaN 则弃步并 ÷2）+ **fp32 master weights**（优化器永远在 fp32 上更新）
- bf16：8 位指数同 fp32（永不溢出，免 scaler），7 位尾数换来的。**T4 是 Turing 无 bf16**——我们被迫 fp16+scaler，也是与论文的已知偏离
- 日志实证：`scale: 32768→33587` 动态试探；梯度范数 `grdn` 在裁剪前已 unscale

### 5.3 训练语义三陷阱（展示严谨性）

1. **步数语义**：lerobot 的 steps 计的是每卡 micro-batch 步。跨配置对比必须锚定 eff-bs=bs×GPU×accum 与总样本 512k——我们三版配置（v9/v10/v11）账本等价
2. **调度器等比缩放**：steps=8000 时框架自动把 warmup 2000→500、decay 32000→8000（factor 0.25），LR-每样本轨迹与长跑一致
3. **严格 resume vs 热启动**：谷底 LR 续训学不到东西；延长训练的标准做法是重新升温衰减。我们 16k 用热启动并在归因表如实标注（优化器动量重置 + 新 seed 数据序）

### 5.4 真实工程战例（STAR 叙述库——比八股更值钱）

**战例 1：一次"成功"的假训练（静默失败）**
S：批处理模式训练显示 Successfully ran；T：实际训练从未发生；A：给 guard 加 150 秒硬存活检查 + 失败显式 raise；R：之后所有失败 5 分钟内定位。
根因链：遗留状态文件 + 目录冲突，掩盖在"成功"退出码下。**教训：退出码≠事实，关键路径必须有正反向断言。**

**战例 2：封号与申诉**
S：连续 36h 高负载 + 崩溃重试循环被自动系统判 Resource abuse 封号；T：恢复账号且不损失产物；A：诚实申诉（承认 retry/heartbeat 模式、给出 GitHub 证据链）+ 立即切换备用算力路线（AI Pro/Colab）；R：<1 天解封，项目零数据损失。
**教训：产物外置（HF/GitHub）让平台故障永不等于项目失败。**

**战例 3：配额 2× 隐藏消耗**
S：30h/周配额按预期 5h 应跑 8000 步，实际 2.4h 就死；A：读配额页 31:18/30 发现 **T4×2 会话按 2× 烧**；R：改单卡评测（P100 1×）+ 接力训练架构。
**教训：配额计量单位要实测验证，文档没写的会被账单写出来。**

**战例 4：接力 bug 链（v16→v19）**
路径深度过滤 `==3` 应为 `>=3`（HF 文件列表全是 4 段）→ 误判无备份退回 8k 重训；混合 6/7 位命名使 glob 失配 → 改精确文件清单逐个下载；校验差一层子目录 → 指向 `pretrained_model/`。四连修后机制闭环：任意命名/任意轮次都能正确接力。
**教训：对云 API 返回的路径结构不做假设；每次修复后用真实数据回归。**

**战例 5：评测卡死与断点续测设计**
逐步重规划 100 集 × 每步推理的 tqdm 洪流淹没了 notebook 显示通道（浏览器渲染反压），且评测无断点。方案：lerobot 源码确认 `--env.task_ids` 原生单任务支持 → 按任务粒度切片 + 每任务完成即上传 HF + 重跑自动跳过已完成。**教训：长任务设计断点粒度 = 最小可接受重做成本。**

### 5.5 高频追问快答（30 秒一题）

1. **loss 起步值说明什么**？热启动 0.40 vs 从零 1.14——权重是否加载的日志级证据
2. **为什么 batch=1 评测**？闭环逐步重规划的协议要求；batch 并行会破坏 per-episode 时序
3. **怎么判断训练是否学进去了**？loss 曲线 + grdn 稳定 + 定期 checkpoint 闭环评测
4. **数据增强为什么裁剪动作**？FFW：视觉裁剪 = 相机平移，动作同步平移保持一致，免费的数据增广
5. **warmup 比例怎么定**？官方 preset 6.25%（2000/32000），框架对短跑自动等比（500/8000）
6. **显存 9.7GiB 都在哪**？模型状态 7.2GB（16Ψ×450M）+ 激活；冻结骨干无优化器状态是大头节省
7. **如果给你 8×A100 会怎么改**？bf16 免 scaler、FSDP 或更大 bs、多套件联合、32k+ 步数、训练内周期闭环评测
8. **74% 怎么向非技术人汇报**？论文 25 倍预算下 90%，我们 4% 预算拿到 74%，缩放曲线上完全符合预期

---

## 第六章 · 复现与深化学习计划

### 6.1 已完成里程碑（产物索引）

| 里程碑 | 结果 | 产物 |
|---|---|---|
| M0 环境 | lerobot 0.6.2 + 资产 + 数据锁定 | bootstrap notebook |
| M1 训练 8k | 74% [63.0, 83.2] @100 episodes | checkpoint-008000 + eval_info.json + 100 视频 |
| M2 报告 | 差异归因表 | results/M2_eval_libero_spatial.md |
| M3 阶梯 | 16k 训练完成（待评测） | checkpoints-backup/0008000 |

### 6.2 M3 剩余实验（面试时说"正在进行"的部分）

1. **16k 评测**（排队中）：Curve 点 2，预期 80%±8
2. **32k 阶梯**：验证向 90% 收敛的斜率
3. **LoRA 对照**（`use_peft=true` 一行开关）：全量 vs PEFT 质量/效率
4. **四套件联合**（spatial/object/goal/long）：消除单套件归因
5. **libero_plus 扰动**：光照/相机/干扰物泛化轴（git fork + PYTHONPATH 注入）
6. **推理优化**：APS 异步堆叠 + fp16 推理 → 对齐论文 280ms

### 6.3 深化学习路线（论文 → 源码 → 实验）

1. 精读：SmolVLA 论文 §3（架构）§4（实验）→ 对照 `lerobot/policies/smolvla/` 源码（expert.py 交叉注意力、processor 管线）
2. 补齐基础：flow matching 原论文（Lipman 2023）、action chunking（ACT 论文）、LeRobotDataset 格式设计
3. 扩展阅读：π0（同用 flow matching 的 3B 先例）、OpenVLA（离散 token 对照面）、libero 基准论文
4. 动手深化：把 ntfy 信标换成 wandb；跑通 32k；写 LoRA 消融报告

### 6.4 面试演练清单（自测）

- [ ] 白板画 1.1 数据流图（3 分钟）
- [ ] 写 flow matching 训练/推理伪代码（2 分钟）
- [ ] 背 4.1 对比表 + 4.2 归因表
- [ ] 三个数字脱口而出：**74% [63.0, 83.2]、2.14 s/step、1.9 epoch/512k 样本**
- [ ] 讲一个战例（推荐 5.4-战例 1 或 4，带"教训"收尾）
- [ ] 回答"如果重做一遍会改什么"：一开始就用批处理+前台进程+产物外置，省 80% 运维时间

---

## 附录 A · 关键数字速查卡

| 数字 | 值 |
|---|---|
| 模型 | 450M（骨干 350M 冻结 + 专家 100M） |
| 训练 | 8k updates, eff-bs 64, 512k 样本, 1.9 epoch |
| 硬件/时长 | T4×2 DDP, 4h50m, 2.14 s/step, 双卡 100% |
| 精度 | fp16 + GradScaler(65536) |
| 8k 成功率 | **74%**（Wilson 95% CI 63.0-83.2, 100 eps） |
| 论文对照 | 90% @ 200k updates（25× 预算） |
| 16k | 训练完成（12000+4000 接力），评测排队 |
| 优化器 | AdamW 1e-4 (0.9,0.95) wd 1e-10 clip 10 |
| 数据 | libero_spatial, 273,465 帧 / 1693 episodes, 256×256 双相机 |
| 通信 | 梯度 400MB/步, ~35ms (1.6% 步时) |

## 附录 B · 反向问题（面试尾声你问面试官，展示视野）

1. "贵司的 VLA 训练管线用 torch 原生 DDP 还是 FSDP/DeepSpeed？策略网络和数据加载的瓶颈比大概是多少？"
2. "闭环评测的重规划频率和真机推理延迟预算怎么平衡？有没有用异步推理堆叠？"
3. "跨本体预训练数据的权重配比和灾难性遗忘有什么内部经验？"
