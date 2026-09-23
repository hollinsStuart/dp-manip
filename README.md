# dp-manip

基于 ManiSkill 的专家轨迹生成 + Diffusion Policy 训练项目。本仓库（Mac）是**唯一权威源**，代码、配置、文档都在这里编辑，再同步到 ubuntu 与 wsl；本文件是入口，负责说明设备分工、数据流和常用命令。

- 当前状态：[STATUS.md](./STATUS.md)
- 未完成事项：[TODO.md](./TODO.md)
- 课程要求与进度对照：[docs/requirements.md](./docs/requirements.md)
- 面向编码代理的操作约束：[AGENT.md](./AGENT.md)
- 多设备工作流计划：[PLAN.md](./PLAN.md)
- Day 1 原始记录：[docs/history.md](./docs/history.md)

---

## 一、设备分工

| 设备               | SSH 访问        | 项目路径                                      | 硬件                                 | 角色                              | 关键限制                                                            |
| ------------------ | --------------- | --------------------------------------------- | ------------------------------------ | --------------------------------- | ------------------------------------------------------------------- |
| MacBook（本机）    | 本地            | `/Users/hollins/Documents/Coding/dp-manip`    | M3 Max / 36 GB                       | 权威仓库、编辑、文档、编排、数据中转、分析 | 装不了 mplib（`libclang==11.0.1` 无 macOS ARM64 wheel），不生成数据 |
| ubuntu             | `ssh ubuntu`    | `~/Coding/dp-manip`                           | i7-8700K / 16 GB / GTX 1080 Ti 11 GB | 专家轨迹生成 + state 重放         | GTX 1080 Ti（sm_61）与当前 PyTorch CUDA 构建不兼容，**不能训练**    |
| wsl                | `ssh wsl`       | `~/projects/dp-manip`                         | WSL2 / RTX 4090 / 驱动 591.86        | DP 训练与评估                     | 尚未安装 ManiSkill（只有训练/校验环境）                             |

> SSH 别名定义在 `~/.ssh/config`：`ubuntu` = 10.0.0.200，`wsl` = 10.0.0.248。另有 `ubuntu-frp` 走公网 frp，一般只在局域网不可达时使用。

**一句话原则：ubuntu 造数据，wsl 训模型，Mac 做编排和权威仓库；不要把训练放到 ubuntu，不要在 Mac 上折腾 mplib。**

---

## 二、仓库结构

```text
dp-manip/
  README.md AGENT.md STATUS.md TODO.md PLAN.md
  docs/history.md                           # Day 1 原始记录（原 in.txt）
  pyproject.toml uv.lock .python-version    # wsl 训练环境；只有 wsl 可以据此 uv sync
  scripts/                                  # 训练 / 校验脚本（在 wsl 运行）
  configs/                                  # 任务级 dataset / training / evaluation 配置
  run_cpu.py patches/ environment/ mplib-probe-overrides.txt   # ubuntu 专家环境
  manifests/                                # 数据 SHA-256 清单（入库）
  demos-*/ data/                            # 数据，gitignore，rsync 传输
  results/ checkpoints/ logs/               # 训练产出，gitignore，从 wsl 回传
```

`.venv/` 在三端各自独立、互不相同，全部 gitignore。Mac 本地 `.venv` 是 9.22 装的 ManiSkill 仿真环境（无 mplib），ubuntu 的是专家环境，wsl 的是训练环境。

---

## 三、同步方式

所有 git 与 rsync 操作都由 **Mac 发起**，远端从不主动连接 Mac（细节见 [PLAN.md](./PLAN.md)）：

- **代码 / 配置 / 文档**：Mac 提交 → `git push ubuntu main` / `git push wsl main`。远端仓库设置了 `receive.denyCurrentBranch=updateInstead`，远端有未提交改动时 push 会被拒绝。
- **wsl 上的临时修改**：在 wsl 本地提交 → Mac `git fetch wsl && git merge --ff-only wsl/main`。
- **数据**：rsync 传输，传完用 `manifests/*.sha256` 校验。
- **训练产出**：从 wsl rsync `results/ checkpoints/ logs/` 回 Mac，不带 `--delete`。

日常用 `scripts/sync.sh`（仅在 Mac 执行）：

```bash
scripts/sync.sh status         # 三端 HEAD、工作区、数据清单
scripts/sync.sh push           # Mac 提交后推到 ubuntu / wsl
scripts/sync.sh fetch          # 取回 wsl 上的提交（仅 fast-forward）
scripts/sync.sh data           # ubuntu demos-*/ → Mac，Mac data/ → wsl，并校验
scripts/sync.sh manifest       # 新数据产生后重新生成清单，再提交
scripts/sync.sh pull-results   # wsl 训练产出 → Mac
```

---

## 四、数据流

```text
ubuntu                                          wsl
──────                                          ───
run_cpu.py
  └─ 专家轨迹 (motionplanning)  ─── rsync ──▶  data/pickcube/*.h5|json
replay_trajectory --obs-mode state
  └─ state 观测重放              ─── rsync ──▶  data/pickcube/state/*.h5|json
                                                       │
                                                       ▼
                                                 DP dataset → 训练 → 评估
                                                       │
Mac：权威仓库，发起所有同步；数据经 Mac 中转  ◀── rsync ─┘ results / checkpoints / logs
```

原始专家文件只有动作，`obs` 组为空；可训练数据是 `obs_mode: state` 的重放版本。两者都要传递。

---

## 五、各设备环境

### ubuntu（专家数据源）

| 项目          | 值                                                                 |
| ------------- | ------------------------------------------------------------------ |
| 系统/硬件     | Ubuntu Server 26.04，i7-8700K，16 GB RAM，GTX 1080 Ti 11 GB         |
| NVIDIA 驱动   | 580.178.04                                                          |
| Python        | 3.11.15，虚拟环境 `~/Coding/dp-manip/.venv`                         |
| 核心包        | `mani-skill==3.0.1`、`mplib==0.2.1`、`sapien==3.0.3`、`numpy==1.26.4` |
| 依赖冻结      | `environment/ubuntu-expert-freeze.txt`                             |
| MPlib 覆盖    | `mplib-probe-overrides.txt`（`mplib==0.2.1`）                       |
| 适配补丁      | `patches/mani_skill_mplib_0_2_1.patch`（`set_base_pose` / `plan_screw`） |
| 入口          | `run_cpu.py`（CPU 物理 + CPU 渲染）                                 |

运行方式：`sim_backend="physx_cpu"` + `render_backend="cpu"`，并通过 `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` 指定 Vulkan 驱动。

> ⚠️ 仓库根目录的 `pyproject.toml` / `uv.lock` 是 wsl 的训练环境。**ubuntu 上禁止 `uv sync`**：uv 会把 `.venv` 精确同步成那份 lock，卸掉 mani-skill 与 mplib。ubuntu 环境只按 freeze 文件与补丁重建。

> 旧环境 `~/Coding/dp-manip-old-backup`（含 `.venv-mplib-probe`）和 `~/Coding/dp-manip-clean-venv-backup` 仅作备份，不要在其上继续工作，也不要删除，除非 TODO E 确认。

### wsl（训练节点）

| 项目        | 值                                                                |
| ----------- | ----------------------------------------------------------------- |
| 系统/硬件   | WSL2 Ubuntu 24.04.5 LTS，RTX 4090（计算能力 8.9）                  |
| Python      | 3.11.15，虚拟环境 `~/projects/dp-manip/.venv`，由 `uv 0.12.18` 管理 |
| 核心依赖    | PyTorch `2.14.0+cu130`、NumPy `1.26.4`、h5py `3.16.0`             |
| GPU 驱动    | 591.86（`nvidia-smi` 报 CUDA 13.1，PyTorch 运行时 CUDA 13.0）      |
| 尚未安装    | ManiSkill、torchvision、Diffusers、Hydra 等（评估环境待补）        |
| 校验脚本    | `scripts/verify_cuda.py`、`scripts/smoke_train_cuda.py`           |
| 数据脚本    | `scripts/inspect_dataset.py`、`scripts/validate_replay.py`、`scripts/check_temporal_windows.py` |

复现环境（wsl 项目根）：

```bash
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_CACHE_DIR="$PWD/.uv-cache"
uv venv --python 3.11.15
uv sync --frozen
```

venv 无 pip 模块，用 `uv pip ... --python .venv/bin/python` 操作。

### MacBook（本机）

M3 Max / 36 GB。本地 `.venv`（Python 3.11，ManiSkill + Vulkan/MoltenVK）可做仿真与可视化，但 mplib 装不上，所以只承担编辑、文档、数据中转和分析。**Mac 上同样不要 `uv sync`**，否则会把本地 ManiSkill 环境替换成 wsl 的训练依赖。

---

## 六、数据资产（PickCube，10 条，全部成功）

| 名称                     | 内容                                        | 维度                                                        |
| ------------------------ | ------------------------------------------- | ----------------------------------------------------------- |
| 原始专家轨迹             | `pickcube_batch10.h5` + `.json`             | `actions float32 (T, 8)`；`obs` 为空                        |
| state 重放               | `pickcube_batch10.state.pd_joint_pos.physx_cpu.{h5,json}` | `obs float32 (T+1, 42)`；`actions float32 (T, 8)` |

- 动作长度：74、74、50、86、76、88、71、74、49、84（共 726 步）。
- 控制模式 `pd_joint_pos`，仿真/渲染后端均为 CPU，seed 0–9。
- H5 顶层为 `traj_0`…`traj_9`，每条一个组；JSON `episodes[].episode_id` 是边界映射。逐步 `terminated`/`truncated` 可能提前变真，切分只能用组 + JSON ID。
- 三端 SHA-256 一致，清单见 `manifests/ubuntu-demos.sha256`、`manifests/wsl-data.sha256`；各文件哈希也列在 [STATUS.md](./STATUS.md)。

ubuntu 路径：`~/Coding/dp-manip/demos-batch/PickCube-v1/motionplanning/`
wsl 路径：`data/pickcube/`（原始）与 `data/pickcube/state/`（重放）
Mac：同时保存两种布局（`demos-*/` 与 `data/`），用作中转和备份。

---

## 七、常用命令

### ubuntu：生成专家轨迹

```bash
cd ~/Coding/dp-manip
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
.venv/bin/python run_cpu.py --env-id PickCube-v1 \
  --sim-backend physx_cpu --only-count-success -n 10 \
  --traj-name pickcube_batch10 --record-dir demos-batch
```

### ubuntu：重放为 state 观测

```bash
cd ~/Coding/dp-manip
.venv/bin/python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos-batch/PickCube-v1/motionplanning/pickcube_batch10.h5 \
  --obs-mode state --save-traj --use-env-states \
  --max-retry 0 --num-envs 1 --verbose
```

### Mac：数据中转（ubuntu → Mac → wsl）

```bash
rsync -a ubuntu:Coding/dp-manip/demos-batch ./
mkdir -p data/pickcube/state
cp demos-batch/PickCube-v1/motionplanning/pickcube_batch10.{h5,json} data/pickcube/
cp demos-batch/PickCube-v1/motionplanning/pickcube_batch10.state.pd_joint_pos.physx_cpu.{h5,json} data/pickcube/state/
rsync -a data/ wsl:projects/dp-manip/data/
shasum -a 256 -c manifests/wsl-data.sha256
```

### Mac：重新生成清单（新数据产生后）

```bash
ssh ubuntu 'cd ~/Coding/dp-manip && find demos-* -type f | LC_ALL=C sort | xargs sha256sum' > manifests/ubuntu-demos.sha256
ssh wsl 'cd ~/projects/dp-manip && find data -type f | LC_ALL=C sort | xargs sha256sum' > manifests/wsl-data.sha256
```

### wsl：环境与数据校验

```bash
cd ~/projects/dp-manip
.venv/bin/python scripts/verify_cuda.py
.venv/bin/python scripts/smoke_train_cuda.py --amp
.venv/bin/python scripts/inspect_dataset.py            # 默认检查 state 文件；也可传 path/to/file.h5 [--json path/to/file.json]
.venv/bin/python scripts/check_temporal_windows.py
.venv/bin/python scripts/validate_replay.py \
  data/pickcube/pickcube_batch10.h5 \
  data/pickcube/state/pickcube_batch10.state.pd_joint_pos.physx_cpu.h5
```

`inspect_dataset.py` 会打印元数据、每条轨迹的形状与类型、数值统计、布尔计数和对齐警告；`check_temporal_windows.py` 用历史长度 2、动作 horizon 8 验证时间窗口不跨 episode。

---

## 八、当前状态与下一步

- ubuntu：PickCube 环境、专家生成、state 重放、回放链路全部通过；10/10 成功。
- wsl：M0 训练节点验证完成（CUDA、FP32/AMP 训练、数据校验全通过）；正式训练尚未开始。
- Mac：9.23 建为权威 git 仓库，ubuntu / wsl 已挂接，`scripts/sync.sh` 可用。

下一步（详见 [TODO.md](./TODO.md)）：完成多设备工作流 → 确定控制模式（`pd_joint_pos` 8 维 vs `pd_ee_delta_pos` 4 维）与观测/动作 schema → 写 dataset adapter 与评估环境 → 跑通「专家轨迹 → 数据集 → DP 训练 → 评估」→ 扩展到六个任务。
