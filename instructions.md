# DP-Manip 正式训练操作说明

本文面向负责集群正式训练的队员。每位队员只负责一个 ManiSkill 任务；训练、续跑、日志、
checkpoint 和评估都使用本仓库的统一入口。不要修改实验配置，也不要使用旧的 Job Array 脚本。

## 1. 先确认自己负责哪个任务

“Task 1”是团队分工编号，不是程序参数。传给程序的 `TASK` 必须是下表中的英文名称：

| 团队编号 | 任务 | `TASK` 参数 |
| --- | --- | --- |
| Task 1 | PickCube | `pickcube` |
| Task 2 | StackCube | `stackcube` |
| Task 3 | PushCube | `pushcube` |
| Task 4 | PullCube | `pullcube` |
| Task 5 | PegInsertionSide | `peginsertionside` |
| Task 6 | PlugCharger | `plugcharger` |

例如，负责 Task 1 的队员应执行：

```bash
export TASK=pickcube
```

不要写 `TASK=1`、`TASK="Task 1"` 或 Slurm 数组下标。

## 2. 版本、数据路径和个人输出目录

开始前向负责人取得以下信息：

1. 仓库地址和批准训练的 commit；
2. `DATA_ROOT`：只读的 RGB 数据集根目录；
3. `RUN_ROOT`：自己账号下、计算节点可访问的个人正式输出目录。

每位队员使用自己的 `RUN_ROOT`，不同队员不需要共享同一个目录。运行名自带 task 名称，最后
可以安全汇总；但同一队员负责的 data-size、backbone 和 evaluation 必须始终使用同一个
`RUN_ROOT`，这样 N=100 UNet 才会复用，续训和评估也才能找到原来的 checkpoint。

本集群没有 `/scratch`。推荐每人把自己的任务写到个人 home 下的独立目录，例如
`$HOME/dp-runs-pickcube`。按当前 checkpoint 大小，一个任务的 26 个唯一 run 预计约占
50–60 GB；100 GB home 通常能容纳一个任务，但不适合一个人同时承担两个任务。训练期间必须
持续检查个人配额，空间不足时先停止提交并联系负责人，不能通过删除正在使用的 checkpoint
临时腾空间。

首次准备仓库：

```bash
git clone --branch refactor/dp-manip-rgb-cluster <repo-url> dp-manip
cd dp-manip
git checkout <负责人提供的 commit>
./setup.sh
```

每次登录后，在仓库根目录设置本次运行参数。下面以 Task 1 为例：

```bash
cd ~/7606C/dp-manip

export TASK=pickcube
export DATA_ROOT=<负责人提供的数据集根目录>
export RUN_ROOT="$HOME/dp-runs-${TASK}"

test -d "$DATA_ROOT"
mkdir -p "$RUN_ROOT"
df -h "$HOME"
```

`DATA_ROOT` 下的数据由 `configs/tasks/<task>.toml` 选择。例如 PickCube 会读取：

```text
$DATA_ROOT/train/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5
$DATA_ROOT/val/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5
```

文件名中的 `state` 是现有导出命名；正式 trainer 实际读取 HDF5 中的 `obs_rgb/rgb` 和
`obs_rgb/state`。不要移动、重命名或修改数据文件。

## 3. 提交前检查

先确认代码版本正确、工作区没有个人修改：

```bash
git status --short --branch
git rev-parse HEAD
```

然后只检查自己负责的任务：

```bash
.venv/bin/python scripts/inspect_dataset.py \
  --data-root "$DATA_ROOT" \
  --config "configs/tasks/${TASK}.toml"

.venv/bin/python scripts/check_experiment.py \
  --experiment data_size --task "$TASK" --data-root "$DATA_ROOT"

.venv/bin/python scripts/check_experiment.py \
  --experiment backbone --task "$TASK" --data-root "$DATA_ROOT"
```

三条命令必须成功。数据检查应输出 `PASS`，两个实验检查应分别输出：

```text
Gate B ok: data_size matrix is controlled across 16 cells
Gate B ok: backbone matrix is controlled across 15 cells
```

提交前还应查看只读计划。它不会启动训练：

```bash
.venv/bin/python scripts/sweep.py plan \
  --experiment configs/experiments/data_size.toml \
  --task "$TASK" --data-root "$DATA_ROOT" --output-root "$RUN_ROOT"
```

状态含义：

- `pending`：尚未完成，将进入队列；
- `pending (resume.pt)`：之前中断，将从恢复点继续；
- `completed`：已经完成，正式提交时会跳过；
- `conflict`：目录中是另一份配置，必须停止并联系负责人，不能删除或覆盖。

## 4. 第一步：训练 data-size 实验

每个任务的 data-size 实验共有 16 个 run：

- N=25：seed 1–3；
- N=50：seed 1–3；
- N=100：seed 1–5；
- N=200：seed 1–5。

一个 Slurm 作业会申请两张 GPU，内部两个 worker 动态领取这 16 个 run。不要计算数组下标，
不要提交 `train_array.sbatch`，也不要自行添加 `--set`；正式训练必须保持 100k steps。

```bash
sbatch --open-mode=append \
  --export=ALL,TASK="$TASK",EXPERIMENT=configs/experiments/data_size.toml,DATA_ROOT="$DATA_ROOT",RUN_ROOT="$RUN_ROOT" \
  slurm/train_dual_gpu.sbatch
```

命令会输出：

```text
Submitted batch job <JOBID>
```

把 `<JOBID>` 记录到团队表格或实验记录中。当前 QOS 对每个用户只允许一个 submitted job，
所以该作业结束前不要再提交 backbone 或 evaluation 作业。

## 5. 如何查看训练进度和 loss

查看队列：

```bash
squeue -u "$USER" -o "%.18i %.34j %.2t %.10M %R"
```

仓库根目录中的顶层 Slurm 日志记录调度过程：

```bash
tail -f "slurm-dp-rgb-train-dual-<JOBID>.out"
```

它会显示哪个 worker 开始、完成或跳过了哪个 run，例如：

```text
[worker 0] starting pickcube_rgb_unet_n25_s1
[worker 1] starting pickcube_rgb_unet_n25_s2
[worker 0] completed pickcube_rgb_unet_n25_s1 ...
completed: 16 skipped: 0 failed: 0 interrupted: 0
```

每个 run 有自己的详细训练日志：

```bash
ls "$RUN_ROOT/logs/${TASK}_rgb_"*.log
tail -f "$RUN_ROOT/logs/pickcube_rgb_unet_n25_s1.log"
```

正式配置每 100 optimizer steps 输出并记录一次 loss，例如：

```text
[010000/100000] loss=0.12345 lr=... elapsed=...
[010000] fixed validation denoising loss=0.23456
```

机器可读曲线保存在每个 run 的 `metrics.jsonl`：

```bash
tail -n 10 "$RUN_ROOT/pickcube_rgb_unet_n25_s1/metrics.jsonl"
```

每行是一个 JSON 对象，常见字段为：

- `step`：optimizer step；
- `train_loss`：该记录点的训练去噪 loss；
- `lr`：学习率；
- `elapsed_s`：本次启动后的累计训练时间；
- `val_loss`：固定验证集去噪 loss，仅在验证步骤出现。

验证 loss 在 10k、30k、60k 和最终 100k steps 记录。`metrics.jsonl` 不是每一步一行，
而是每 100 steps 一行；中断续训时继续追加到同一个文件。

## 6. 每个 run 的输出在哪里

运行名格式为：

```text
<task>_rgb_<backbone>_n<N>_s<seed>
```

例如：

```text
pickcube_rgb_unet_n25_s1
pickcube_rgb_transformer_n100_s3
```

目录结构：

```text
$RUN_ROOT/
├── logs/
│   ├── <run-name>.log                 # 单个训练 run 的 stdout/stderr
│   └── eval/<run-name>.log            # 单个闭环评估的 stdout/stderr
└── <run-name>/
    ├── run.json                       # 完整配置、数据 fingerprint、seed、Git 版本
    ├── metrics.jsonl                  # 训练/验证 loss 曲线
    ├── summary.json                   # final step、最终 loss、耗时、GPU 峰值
    ├── checkpoints/
    │   ├── step_010000.pt
    │   ├── step_030000.pt
    │   ├── step_060000.pt
    │   ├── final.pt                   # 正式结果使用这个 checkpoint
    │   └── resume.pt                  # 中断续训使用，不作为正式评估结果
    └── eval/
        └── test_final.json            # final.pt 的 test 闭环评估结果
```

`test_final.json` 包含逐 episode 结果以及汇总指标，例如 `success_once`、
`success_at_end`、`episode_len`、`return` 和推理耗时。

## 7. 中断、自动续训和重新提交

Slurm 到达 walltime 或发生预抢占时会提前发送 USR1。trainer 会在当前 optimizer step 完成后
写入 `resume.pt`，作业自动 requeue，并恢复 model、optimizer、scheduler、EMA、scaler 和 RNG
状态。正常情况下不需要人工处理。

如确实需要手动测试或优雅中断，只能使用：

```bash
scancel --signal=USR1 --batch "<JOBID>"
```

不要使用普通的 `scancel <JOBID>`；它会直接取消作业，不保证写完 checkpoint，也不会走自动
requeue 流程。

如果自动 requeue 失败，可以原样重新执行同一条 `sbatch` 命令：已有 `final.pt` 的 run 会被
跳过，有 `resume.pt` 的 run 会自动续训。不要手工删除 checkpoint。

## 8. data-size 完成后的验收

作业从 `squeue` 消失后，检查顶层日志最后一行。必须满足：

```text
failed: 0 interrupted: 0
```

再次运行计划和实际配置对账：

```bash
.venv/bin/python scripts/sweep.py plan \
  --experiment configs/experiments/data_size.toml \
  --task "$TASK" --data-root "$DATA_ROOT" --output-root "$RUN_ROOT"

.venv/bin/python scripts/check_experiment.py \
  --experiment data_size --task "$TASK" \
  --data-root "$DATA_ROOT" --run-root "$RUN_ROOT"
```

期望计划显示 16 个 completed，Gate B 显示 16 cells ok。若存在 pending、conflict、failed，
不要开始 backbone，先把顶层日志和对应的 `$RUN_ROOT/logs/<run-name>.log` 发给负责人。

## 9. 第二步：训练 backbone 实验

data-size 验收通过后，使用完全相同的 `DATA_ROOT` 和 `RUN_ROOT`：

```bash
sbatch --open-mode=append \
  --export=ALL,TASK="$TASK",EXPERIMENT=configs/experiments/backbone.toml,DATA_ROOT="$DATA_ROOT",RUN_ROOT="$RUN_ROOT" \
  slurm/train_dual_gpu.sbatch
```

backbone 实验声明 15 个 run：UNet、Transformer、MLP 各 5 个 seed。UNet 的 N=100 run 与
data-size 实验完全相同，因此正常情况下会直接 `skipped: 5`，只新增 10 个 Transformer/MLP
run。若换了 `RUN_ROOT`，这 5 个 UNet 会被错误地重新训练。

完成后检查：

```bash
.venv/bin/python scripts/sweep.py plan \
  --experiment configs/experiments/backbone.toml \
  --task "$TASK" --data-root "$DATA_ROOT" --output-root "$RUN_ROOT"

.venv/bin/python scripts/check_experiment.py \
  --experiment backbone --task "$TASK" \
  --data-root "$DATA_ROOT" --run-root "$RUN_ROOT"
```

期望计划显示 15 个 completed，Gate B 显示 15 cells ok。

## 10. 第三步：闭环评估

训练和计划检查全部通过后，评估 data-size 的 `final.pt`：

```bash
sbatch --open-mode=append \
  --export=ALL,TASK="$TASK",EXPERIMENT=configs/experiments/data_size.toml,RUN_ROOT="$RUN_ROOT",CHECKPOINT=final.pt,SPLIT=test,NUM_ENVS=4 \
  slurm/eval_dual_gpu.sbatch
```

该作业完成后，再评估 backbone：

```bash
sbatch --open-mode=append \
  --export=ALL,TASK="$TASK",EXPERIMENT=configs/experiments/backbone.toml,RUN_ROOT="$RUN_ROOT",CHECKPOINT=final.pt,SPLIT=test,NUM_ENVS=4 \
  slurm/eval_dual_gpu.sbatch
```

评估顶层日志位于：

```text
slurm-dp-rgb-eval-dual-<JOBID>.out
```

每个 run 的评估日志和结果分别位于：

```text
$RUN_ROOT/logs/eval/<run-name>.log
$RUN_ROOT/<run-name>/eval/test_final.json
```

顶层汇总必须为 `failed: 0 interrupted: 0`。缺少 checkpoint 时，评估会明确打印
`missing checkpoint` 并以失败结束，不要忽略或手工伪造结果文件。

## 11. 常见错误和禁止事项

- 不要提交 `slurm/train_array.sbatch` 或 `slurm/eval_array.sbatch`。
- 不要把团队编号当成 `TASK` 参数；例如 Task 1 使用 `TASK=pickcube`。
- 不要在正式训练命令中加入 smoke 使用的 `--set train.total_iters=...`。
- 不要让同一用户同时提交训练与评估作业；当前 QOS 只允许一个 submitted job。
- 不要把同一个 `TASK` 分给两名队员，也不要把同一个任务拆到多个 `RUN_ROOT`。
- 不要更改 `configs/baseline.toml`、`configs/tasks/` 或 `configs/experiments/`。
- 不要删除、移动或覆盖别人的 run、checkpoint、日志和数据文件。
- 不要使用普通 `scancel <JOBID>` 期待自动续训。
- 不要因为 validation loss 更低而选择中间 checkpoint；正式结果只使用 `final.pt`。
- 不要把 `physx_cuda` 的评估结果混入本项目的 `physx_cpu` 结果。

## 12. 将个人结果汇总到中央目录

每位队员完成 data-size、backbone、两次 evaluation 和第 8–10 节的检查后，才能开始汇总。
汇总过程中不得再向个人 `RUN_ROOT` 提交任何写入作业。先确认：

```bash
squeue -u "$USER"
du -sh "$RUN_ROOT"
df -h "$HOME"
```

`squeue` 中不得再有本任务的训练或评估作业。完整保留个人 `RUN_ROOT`，不要提前删除
`resume.pt`、中间 checkpoint 或日志。

负责人需要准备一个能容纳所有任务的中央目录。保留全部 checkpoint 时，六个任务合计可能需要
约 300 GB；如果空间不足，应由负责人统一制定保留策略，而且必须在全部评估和核验完成后执行。

不要直接覆盖中央正式目录。先把自己的结果复制到以账号和任务命名的暂存目录：

```bash
export CENTRAL_ROOT=<负责人提供的中央汇总目录>
export STAGING_ROOT="$CENTRAL_ROOT/incoming/${USER}-${TASK}"

mkdir -p "$STAGING_ROOT"
rsync -a --checksum "$RUN_ROOT/" "$STAGING_ROOT/"

mkdir -p "$STAGING_ROOT/slurm-logs"
cp slurm-dp-rgb-train-dual-<DATA_SIZE_JOBID>.out "$STAGING_ROOT/slurm-logs/"
cp slurm-dp-rgb-train-dual-<BACKBONE_JOBID>.out "$STAGING_ROOT/slurm-logs/"
cp slurm-dp-rgb-eval-dual-<DATA_SIZE_EVAL_JOBID>.out "$STAGING_ROOT/slurm-logs/"
cp slurm-dp-rgb-eval-dual-<BACKBONE_EVAL_JOBID>.out "$STAGING_ROOT/slurm-logs/"
```

如果中央目录不是共享挂载，负责人应提供等价的远程 `rsync` 目标；不要自行使用聊天软件压缩传输
大型 checkpoint。

复制后在暂存目录重新检查实际结果。`data.root` 属于运行时路径，因此不同队员的数据绝对路径
可以不同，不影响实验矩阵对账：

```bash
.venv/bin/python scripts/check_experiment.py \
  --experiment data_size --task "$TASK" --run-root "$STAGING_ROOT"

.venv/bin/python scripts/check_experiment.py \
  --experiment backbone --task "$TASK" --run-root "$STAGING_ROOT"

find "$STAGING_ROOT" -name summary.json -type f | wc -l
find "$STAGING_ROOT" -path '*/checkpoints/final.pt' -type f | wc -l
find "$STAGING_ROOT" -path '*/eval/test_final.json' -type f | wc -l
```

两个 Gate B 检查都必须成功，三个文件计数正常情况下都应为 26。负责人确认暂存目录完整后，
再把六个互不重叠的 task 合并到最终结果目录。由于 run 和日志文件名都含 task 名称，正确分工下
不会重名；若汇总工具报告同名或覆盖，立即停止，先检查是否有任务被重复分配。

只有负责人明确确认中央副本完整后，队员才可以清理个人 `RUN_ROOT`。个人目录删除后无法依靠
Git 恢复 checkpoint。

## 13. 完成后交付给负责人

每位队员至少提供：

1. 自己负责的 `TASK`；
2. 使用的 Git commit、`DATA_ROOT`、个人 `RUN_ROOT` 和中央 `STAGING_ROOT`；
3. data-size、backbone、两次 evaluation 的 Slurm job ID；
4. data-size 和 backbone 的最终队列汇总；
5. 两次 `check_experiment.py --run-root` 输出；
6. 所有 run 的 `summary.json`、`final.pt` 和 `eval/test_final.json` 是否齐全；
7. 中央暂存目录的三个文件计数和 Gate B 输出；
8. 任何 requeue、failed、conflict 或人工操作的说明。

遇到不确定情况时先停止提交并保留日志，不要通过改配置、删目录或重跑到“看起来成功”的方式
处理问题。
