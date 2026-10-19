# Kaggle Notebook SSH 连接与实例开关机调研 v3

> 调研日期：2026-10-17/18 · **对应 PLAN.md v6**（57–130h GPU 口径；PLAN 修订后同步复核本文 §4）
> 证据分级：P0=官方文档/公告，P1=权威二手，P2=可验证社区教程/代码，P3=论坛帖/单源经验（论坛回复一律 P3）
> 可信度标注：[CONFIRMED] / [MAJORITY] / [DISPUTED] / [SINGLE-SOURCE] / [推断] / [待核实]

## 0. 结论速览

1. **SSH 主方案 = 官方容器指南路径**：userspace tailscaled + `tailscale up --ssh`，本地 `ssh <user>@kaggle-notebook.<tailnet>.ts.net`（MagicDNS，v3 改：IP 每会话会变）。无 openssh-server、无密码。
2. **与 ModelScope+Tailscale 共存：可以**（免费档用户设备无上限，P0）。
3. **开关机**：GPU 会话 12h 上限（PLAN 主跑 8k updates ≈ 6.4–13.6h，**上界必被强切**，长任务一律后台模式 + hub checkpoint）；空闲回收 20 min（另一官方页 60 min，[DISPUTED] 取保守值）；周配额 30h；交互 GPU 并发数随时间变过（1 或 4，[DISPUTED]）；Persistence 档位 [待核实]。

---

## 1. SSH 连接 Kaggle Notebook

### 1.1 方案对比

| 方案 | 原理 | 优点 | 缺点 | 证据 |
|---|---|---|---|---|
| **A. Tailscale（主方案）** | userspace tailscaled + `--ssh`，MagicDNS 直连 | 官方容器指南原路径；无密码无 sshd；与 ModelScope 同构；**MagicDNS 名跨会话稳定** | 需 userspace 模式；出网封 UDP 退 DERP 中继（仍端到端加密，延迟差，M0 netcheck） | P0 [容器 SSH 指南](https://tailscale.com/learn/ssh-into-docker-container) + [userspace 文档](https://tailscale.com/docs/concepts/userspace-networking) |
| B. Tailscale + OpenSSH（备选） | 自装 sshd | 需要标准 sshd 行为（SFTP/端口转发工具链）时用 | host key、/run/sshd、密码管理 4 个失败点，见 §1.3 备选 | P0+P2 |
| C. ngrok + OpenSSH | TCP 隧道转 22 | P2 教程最全 | 每次会话改 HostName/Port；流量过第三方 | P2 [教程](https://hoang-quoc-trung.github.io/remote-ssh-kaggle-vscode/) |
| D. Zrok / cloudflared | 控制台/CF Tunnel | 免费 | 社区小 / 需 CF 账号 | P2 [kaggle-zrok](https://www.kaggle.com/code/kayak0/kaggle-zrok) |
| E. 官方 Terminal | 编辑器左下角终端图标 | 零配置 | 仅网页编辑器内，非 SSH | P3（论坛 staff 回复） |
| F. JupyterSSH | 经 Jupyter server 流式转发 | 不依赖外部隧道 | 单一项目 [SINGLE-SOURCE] | P3 [JuypterSSH](https://github.com/yaelliethy/JuypterSSH) |

### 1.2 前置条件

- 手机验证（Internet/GPU 前提）[CONFIRMED]。
- Settings：Accelerator=GPU T4×2；Internet=ON；**Persistence 档位 [待核实]**（公告 355440/问答 355889 存在但正文 JS 抓不到；M0 打开 Settings 记录实际档位）。
- 政策：无官方明文禁止 [MAJORITY]；"封号"传言 [SINGLE-SOURCE]。自保：仅个人开发、凭据不进源码/output、不做公共代理。

### 1.3 主方案操作步骤（Tailscale `--ssh`）

**凭据纪律**：`tailscaled.state` 含节点私钥，**绝不放 /kaggle/working**（Files-only 持久化会存成 output 制品）。state/socket/日志一律 `/tmp`；每会话一次性短过期 auth-key 重新 up。**代价（v3 明示）：每会话是新设备、100.x IP 每次都变**——管理台会积累旧设备（用完即删），本地连接用 MagicDNS 名而不是 IP（见 §1.4）。

**Cell 1（Python，注入 Secrets；必须写进 notebook 源码以支持后台模式）**：
```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["TS_AUTH_KEY"] = UserSecretsClient().get_secret("TS_AUTH_KEY")   # 一次性短过期 key
os.environ["HF_TOKEN"]   = UserSecretsClient().get_secret("HF_TOKEN")       # PLAN §4 训练用
from pathlib import Path
p = Path("/tmp/hf_token"); p.write_text(os.environ["HF_TOKEN"]); p.chmod(0o600)
# %%bash 子进程继承 kernel 的 os.environ ✓；SSH shell 不继承，用 /tmp/hf_token（§1.4）
```

**Cell 2（bash，每会话重跑）**：
```bash
%%bash
# 1) tailscale 静态二进制，钉版本（<TS_VER> 由 M0 记录当时 stable 版号后回填）
curl -fsSL -o /tmp/tailscale.tgz https://pkgs.tailscale.com/stable/tailscale_<TS_VER>_amd64.tgz
tar -xzf /tmp/tailscale.tgz -C /tmp
TS=$(echo /tmp/tailscale_*/tailscale); TSD=$(dirname "$TS")/tailscaled
# 2) userspace 模式：nohup + 全重定向 + </dev/null（%%bash 否则挂死）；全部落 /tmp
nohup $TSD --tun=userspace-networking \
  --state=/tmp/tailscaled.state --socket=/tmp/tailscaled.sock \
  >/tmp/tailscaled.log 2>&1 </dev/null &
sleep 3
# 3) auth-key 分支（后台 Save & Run All 只能走这里；浏览器登录 URL 分支后台必败）
$TS --socket=/tmp/tailscaled.sock up --hostname=kaggle-notebook --ssh --auth-key=$TS_AUTH_KEY
$TS --socket=/tmp/tailscaled.sock ip -4    # 记录本次 IP（MagicDNS 为主，IP 为备）
$TS --socket=/tmp/tailscaled.sock netcheck # M0：直连 or DERP 中继，回填开放项
```

- `tailscale up --ssh` 后本地直接 `ssh <tailscale用户名>@kaggle-notebook.<tailnet>.ts.net`——不装 openssh-server、不设密码（官方指南原路径）。
- **设备管理**：/tmp state + 一次性 key ⇒ 每会话一台新设备；**同 hostname 设备并存会让 MagicDNS 名解析歧义——每次开新会话前先在管理台删掉旧 kaggle-notebook**（[推断]：MagicDNS 对同名设备的解析行为未查到官方明文，删除旧设备保证唯一性是稳妥做法）。
- 备选分支（需标准 sshd 时）：`apt-get -y install openssh-server && ssh-keygen -A && mkdir -p /run/sshd`，密码从 Secrets 注入或仅公钥，`nohup /usr/sbin/sshd -D >/tmp/sshd.log 2>&1 </dev/null &`。

### 1.4 本地电脑侧与已知坑

```
# ~/.ssh/config（需 tailnet 开启 MagicDNS；管理台 DNS 页一键开）
Host kaggle-notebook
    HostName kaggle-notebook.<tailnet>.ts.net
    User <tailscale用户名>
    ServerAliveInterval 60
Host modelscope-notebook
    HostName modelscope-notebook.<tailnet>.ts.net
    User youruser
```
- MagicDNS 名跨会话稳定（--hostname 固定 + 删旧设备保唯一）；若 MagicDNS 不可用，退回每会话更新 HostName=新 IP。
- **SSH shell 不继承 kernel 的 os.environ**（GPU 变量与 HF_TOKEN 都一样）：训练脚本若从 SSH 里跑，先 `export HF_TOKEN=$(cat /tmp/hf_token)`（Cell 1 已写好该文件，600 权限、非持久）。
- GPU 环境变量真实值 M0 实测回填（`which nvidia-smi; echo $LD_LIBRARY_PATH; python -c "import torch;print(torch.version.cuda)"`）。
- **venv 与 20GB 二选一（对齐 PLAN §4 分账表）**：torch venv 8–15GB 与主跑 ckpt 峰值 ~10.4GB 放不下同一个 working。推荐不持久 venv；pip 缓存两选一：①只设 `PIP_CACHE_DIR=/kaggle/working/pipcache`（自动复用下载缓存，省下载不省解析/构建）；②真离线秒装先 `pip wheel -r reqs.txt -w /kaggle/working/wheelhouse` 再 `--no-index --find-links=/kaggle/working/wheelhouse`。**PIP_CACHE_DIR 不是 --find-links 能消费的扁平目录**。git+URL 需求缓存收益有限，M0 实测每会话安装墙钟后决定占不占 2–3GB（PLAN §3 本就是系统 pip 直装，无需 venv）。
- `/kaggle/input` 只读（私有数据源 ~107GB）；`/kaggle/working` ~20GB。
- 交互会话空闲回收时 SSH 一起断（§3.2）。

---

## 2. 与 ModelScope+Tailscale 共存

**可以同时连**（tailnet 设计用法）：免费档用户设备无上限（P0 pricing）；两台 notebook 各拿 100.x IP / 各持 MagicDNS 名，同时在线互不影响；同账号多设备登录正常。注意：auth-key 一次性短过期、走 Secrets（TS_AUTH_KEY、HF_TOKEN 都要绑到对应 notebook）；notebook 用普通（非 ephemeral）key 不烧 1000 min/月 ephemeral 额度；过期残留设备管理台手动删。

---

## 3. Notebook 实例开关机与配额

### 3.1 硬限制 [P0 docs/notebooks + 公告 302908]

| 项 | 值 |
|---|---|
| 会话最长时长 | CPU/GPU 12h；TPU 9h（数字历史上调过，公告 302908） |
| /kaggle/working | 20GB 自动保存 |
| scratchpad | 临时盘会话即弃；**HF 缓存归属属 [推断]**，M0 `du -sh ~/.cache/huggingface; df -h` 实测回填 |
| GPU 周配额 | 30h/周（重置每周，"有时更高"），交互+后台都计 |
| 并发限制 | 交互 GPU 数：报错实证 1（product-feedback/160728）与 4（product-feedback/100200）并存 → **[DISPUTED]**；同一 notebook 可开多个后台 batch 会话 [P0] |

### 3.2 空闲回收

- 20 min（docs/notebooks）vs 60 min（efficient-gpu-usage）[DISPUTED]，按保守 20 min。
- 正在运行的 cell 期间不回收（官方论坛原话）。**推论（v2 定稿）：长命令放一个 cell 里跑即 running；超 12h 只能 "Save & Run All" 后台模式（不受空闲回收影响）。** 隧道流量不是 kernel 活动，挂着不跑会被回收 [推断]。
- 后台模式两前提：**HF write token 已绑定该 notebook 的 Secrets**（batch 下 Secret 须事先绑定）；启动 cell 只能用 auth-key 分支（Cell 1 必须在 notebook 源码里）。链路：后台跑 + `--save_checkpoint_to_hub=true --policy.private=true` + 被切后 `--resume`（PLAN §4/§7）。

### 3.3 开机检查单

1. 手机验证 ✓。
2. Settings：Accelerator=GPU T4×2；Internet=ON；Persistence=实际档位 [待核实]（若有 Variables and Files 档可省每会话 git 源码安装 5–10 min）。
3. 环境默认全新：安装代码脚本化在头部；pip 缓存策略见 §1.4。
4. **Secrets 已绑定：TS_AUTH_KEY（必须）、HF_TOKEN（后台训练必须）**。
5. 跑 Cell 1（注入）+ Cell 2（tailscale up + netcheck）。
6. 后台模式：交互改动先存进 notebook 源码；SSH 训练先 `export HF_TOKEN=$(cat /tmp/hf_token)`。

### 3.4 关机检查单

1. **主动 Stop**（右上角或 Active Events），别只关标签——白烧 20–60 min 配额。
2. Stop 前产物已进 /kaggle/working 或已推 HF Hub（**private repo**）。
3. `du -sh /kaggle/working` 自查 20GB（超限 → output 保存失败/截断）。
4. 管理台删旧 kaggle-notebook 设备；记录本周 GPU 用量对照 PLAN §5。

### 4. 与 PLAN.md 的对齐（v3 更新）

- **单次会话**：主跑 8k updates ≈ 6.4–13.6h——12h 上限只容下区间下界，**主跑/基线第二 seed 一律后台模式 + hub checkpoint**；交互会话留给 M0 实测与短任务（sanity 1.6–3.4h、M3 筛选 3–4k updates）。
- **周配额日历**：30h/周 ≈ 每周 ≤2.5 个满额 GPU 会话。PLAN v6 核心合计 **57–130h GPU ≈ 1.9–4.3 周配额**，其中 M3 训练+评测（19.2–40.8h + 12.2–30.6h）合计就吃掉 1–2.4 周；U>6/E>25 降档后分支核心 ≈85–102h——**默认区间下没有整次失败重跑余量，重跑靠降档规则腾空间**。
- **20GB 分账**：主跑 ckpt 峰值 ~10.4GB（后台无法即时清理）+ wheelhouse 2–3GB + 插件源码 + 评测输出 ≈14–15GB，余量 ~5GB；tailscaled state 与 /tmp/hf_token 都在 /tmp。
- 版本对应：**本文 v3 对应 PLAN.md v6**（57–130h，含基线 seed#2 评测 item 5b）。

---

## 5. 开放项（M0 回填）

- [ ] Persistence 实际档位（355440/355889）
- [ ] 交互 GPU 并发上限当前值；后台 batch 是否占用交互名额（决定"后台训练+交互评测"并行）
- [ ] `tailscale netcheck`：直连 or DERP
- [ ] GPU 环境变量真实值（PATH/LD_LIBRARY_PATH/CUDA）
- [ ] HF 缓存归属盘与配额（du + df）
- [ ] 快速档/协议档评测单次耗时（回填 PLAN §5 的 E）
- [ ] hub checkpoint 单次上传墙钟；每会话 pip 安装墙钟（wheelhouse 取舍）
- [ ] tailscale stable 版本号（钉进 Cell 2 的 `<TS_VER>`）
- [ ] MagicDNS 开启确认 + 同名设备解析行为验证（删旧设备策略有效）

## 6. 来源清单

P0：
- Kaggle Notebooks 官方文档（12h/9h、20GB、20min idle、Save & Run All）：kaggle.com/docs/notebooks
- Kaggle Efficient GPU Usage（60min idle、30h/周）：kaggle.com/docs/efficient-gpu-usage
- Kaggle 会话时长上调公告：kaggle.com/discussions/product-feedback/302908
- Kaggle GPU 配额公告：kaggle.com/general/108481
- Kaggle Persistence 公告/问答 [SINGLE-SOURCE 待核实]：product-feedback/355440、questions-and-answers/355889
- Tailscale Userspace Networking：tailscale.com/docs/concepts/userspace-networking
- **Tailscale 容器 SSH 指南（--ssh 主方案出处）**：tailscale.com/learn/ssh-into-docker-container
- Tailscale Pricing / SSH 文档：tailscale.com/pricing、kb/1193

P2：
- Remote-SSH Kaggle via VS Code（ngrok 流程）：hoang-quoc-trung.github.io/remote-ssh-kaggle-vscode
- Kaggle-zrok：kaggle.com/code/kayak0/kaggle-zrok
- ModelScope notebook OpenSSH+Tailscale 社区方案：modelscope.cn/learn/435314
- Kaggle Secrets 使用（UserSecretsClient）：kaggle.com/docs（Secrets 章节）

P3（论坛/单源，谨慎采信）：
- "SSH 封号"传言：product-feedback/293168
- 交互 GPU 并发 1：product-feedback/160728；并发 4：product-feedback/100200 → 合并 [DISPUTED]
- "30h 可拉到 ~42h"（与官方累计口径冲突，不依赖）
- JuypterSSH：github.com/yaelliethy/JuypterSSH
