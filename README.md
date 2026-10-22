# SmolVLA × LIBERO — Zero-Cost Reproduction & Scaling Study

A full reproduction of HuggingFace [SmolVLA](https://huggingface.co/papers/2506.01844) (a 450M-parameter vision-language-action foundation model) fine-tuned on the [LIBERO](https://libero-project.github.io/) simulation benchmark — entirely on **free Kaggle T4×2** GPUs — covering the complete loop: **training → unattended monitoring → protocol-faithful evaluation → scaling experiments**.

## Results at a Glance

| Item | Value |
|---|---|
| Training | 8000 updates @ eff. batch 64 (512k samples ≈ 1.9 epochs), libero_spatial |
| Hardware | 2 × T4 (torchrun DDP), fp16 + GradScaler |
| Wall time | **4h 50m** (2.16 s/step, both GPUs at 100%) |
| Loss | 1.14 → ~0.5 (flow matching MSE) |
| Trainable params | 100M / 450M (frozen VLM backbone, official preset) |
| Success rate (8k) | **74%** (Wilson 95% CI [63.0%, 83.2%], 100 episodes / 10 tasks) — paper: 90% with 25× the training budget |
| Success rate (16k) | **80%** (Wilson 95% CI [71.0%, 86.6%], same protocol) — doubling training adds +6 pp; scaling curve in [results](results/M3_eval_16k.md) |
| **Model weights** | 🤖 [A2bao/smolvla-libero-spatial](https://huggingface.co/A2bao/smolvla-libero-spatial) (private repo, 8k/16k checkpoints + full training states) |

## Why This Stack

| Dimension | Choice | One-line rationale |
|---|---|---|
| Model | SmolVLA-base 450M | The only VLA tier that fits full fine-tuning on free T4s |
| Data | lerobot/libero v3.0 (1.9 GB) | Only option satisfying format compat + Kaggle disk + official docs simultaneously |
| Framework | lerobot 0.6.2 @ commit `146d8474` | Official implementation; train/eval/data in one pipeline |
| Parallelism | torchrun DDP ×2, bs32/GPU | 450M fits per-GPU → DDP beats FSDP; 1.9× speedup over single GPU |
| Precision | fp16 + GradScaler | T4 (Turing) has no bf16 — the only 2× fast path |
| Fine-tuning | Frozen VLM backbone + full 100M expert | Official preset; 9.7/15 GiB VRAM headroom |
| Execution | Kaggle batch mode + foreground training + auto-resume | No idle timers; silent failures eliminated by construction |
| Monitoring | ntfy.sh HTTP beacons (every 10 min) | Zero accounts, bidirectionally reachable, 5/5 reliability in production |

Full decision records (alternatives considered + fallback plans) in [PLAN.md](PLAN.md).

## Reproduce

1. Create a Kaggle Notebook (GPU T4 ×2), import [kaggle_bootstrap_v11_ddp.ipynb](kaggle_bootstrap_v11_ddp.ipynb)
2. **Save Version → Save & Run All** (batch mode; closing the browser changes nothing)
3. Health markers appear in order: `OK-setup` → `Effective batch size: 32 x 2 dp workers` → tqdm counts to 8000 → `DONE`
4. Checkpoints every 2000 steps (optimizer/scheduler/RNG full state); crashes auto-resume

Training entry point (equivalent expansion):

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

For the 16k ladder point: [kaggle_bootstrap_v13_resume16k.ipynb](kaggle_bootstrap_v13_resume16k.ipynb) warm-starts from the latest checkpoint on the Hub and continues +8k updates; it auto-resumes from the newest backup if the session dies.

## Repository Layout

```
├── kaggle_bootstrap.ipynb               # v9 minimal single-GPU (infra reference)
├── kaggle_bootstrap_v10_speed.ipynb     # v10 single-GPU bs32 A/B (proves T4 saturates at bs16)
├── kaggle_bootstrap_v11_ddp.ipynb       # v11 2×T4 DDP training (the 8k run)
├── kaggle_bootstrap_v12_eval.ipynb      # v12 protocol eval (8k → 74%)
├── kaggle_bootstrap_v13_resume16k.ipynb # v13→v19 quota-resilient continuation (16k)
├── kaggle_bootstrap_v20_eval16k.ipynb   # v20 protocol eval for the 16k checkpoint
├── colab_eval16k.ipynb                  # v21 Colab eval, per-task resumable
├── results/                             # eval reports + aggregate JSON
│   ├── M2_eval_libero_spatial.md        #   8k: 74%, per-task breakdown
│   ├── M3_eval_16k.md                   #   16k: 80%, paired-flip analysis
│   └── aggregate_16k.json
├── PLAN.md                              # 3-phase plan + attribution tables
├── KAGGLE_SSH.md                        # SSH/tunnel lessons (deprecated, archived)
├── paper/SmolVLA_paper.md               # paper notes
└── logs/                                # archived training/eval logs
```

## Unattended Monitoring Architecture

Training runs as a **foreground process** of the notebook cell — any crash puts its traceback directly into the batch log, eliminating silent failures by construction. An independent heartbeat process pushes step/GPU/disk telemetry to a private ntfy.sh topic every 10 minutes, letting a remote agent take over monitoring and diagnosis. Every 1000–2000 steps a full-state checkpoint lands (optimizer/scheduler/GradScaler/RNG/sampler offset) and is pushed to the Hub in the background — session death costs at most one interval, and continuation warm-starts from the newest backup automatically.

## License

MIT
