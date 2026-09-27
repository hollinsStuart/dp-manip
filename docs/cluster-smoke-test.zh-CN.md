# 集群 Smoke Test 检查表

在提交 100k-step 正式 sweep 之前，用约 200 optimizer steps 的小作业验证单作业双 GPU 队列、
GPU 隔离、动态补位、抢占续训、三种 backbone 和结果对账。每一步都可直接复制执行。

用到的 smoke grid：`configs/experiments/smoke.toml`（`policy.backbone` = unet / transformer /
mlp，各 1 个训练 seed，共 3 个 run）；降低训练预算用普通 `--set` 覆盖传入，不写进任何正式
实验定义。

## 0. 前置检查

在登录节点执行（`gpu2gate1`）：

```bash
cd ~/7606C/dp-manip
./setup.sh                       # 若 .venv 尚未建立
# 本集群没有 /scratch，数据与输出都放在 $HOME；按实际部署调整这两个根目录。
export DATA_ROOT=$HOME/maniskill-demogen/data/dataset
export SMOKE_ROOT=$HOME/dp-smoke
rm -rf "$SMOKE_ROOT"             # 每次都从干净的 smoke root 开始

.venv/bin/python scripts/inspect_dataset.py --data-root "$DATA_ROOT" \
  --config configs/tasks/peginsertionside.toml

.venv/bin/python scripts/check_experiment.py --experiment smoke --task peginsertionside \
  --set train.total_iters=200 --set 'train.validation_steps=[200]' \
  --set 'train.checkpoint_steps=[200]'

bash -n slurm/train_dual_gpu.sbatch
bash -n slurm/eval_dual_gpu.sbatch
```

期望：数据检查 `PASS`；Gate B 输出 `smoke matrix is controlled across 3 cells`；两条 `bash -n`
无输出。

## Round 1：单作业、双 GPU 并发、动态补位

```bash
sbatch --export=ALL,TASK=peginsertionside,EXPERIMENT=configs/experiments/smoke.toml,DATA_ROOT="$DATA_ROOT",RUN_ROOT="$SMOKE_ROOT" \
  slurm/train_dual_gpu.sbatch \
  --set train.total_iters=200 --set 'train.validation_steps=[200]' \
  --set 'train.checkpoint_steps=[200]'
# 记下输出里的 Submitted batch job <JOBID>
export JOBID=<JOBID>
```

**1a. 队列里只有 1 个作业，且不是数组。**

```bash
squeue -u "$USER" -o "%.18i %.34j %.2t %.10M %R"
```

期望：只有一行 dp-rgb-train-dual 作业，`JOBID` 没有 `_1`/`_2` 之类的数组后缀。

**1b. 两个 worker 同时训练。**

```bash
tail -f "slurm-dp-rgb-train-dual-${JOBID}.out"
```

期望（3 个 run、2 个 worker）：

```text
[worker 0] starting peginsertionside_rgb_unet_n100_s1
[worker 1] starting peginsertionside_rgb_transformer_n100_s1
...
completed: 3 skipped: 0 failed: 0 interrupted: 0
```

第三个 run 会在某个 worker 完成上一个 run 后立即开始：日志里先出现一条 `completed`，紧接着
同一个 worker 出现新的 `starting`（动态补位，不是成对等待）。

**1c. 每个 trainer 只看到一张 GPU**（在作业仍运行时，另开一个登录会话）。

```bash
srun --jobid="$JOBID" --overlap bash -c '
  for pid in $(pgrep -f "train_dp.py"); do
    args=$(tr "\0" " " < /proc/$pid/cmdline)
    case "$args" in *".venv/bin/python"*) ;; *) continue ;; esac
    echo "pid=$pid $(tr "\0" "\n" < /proc/$pid/environ | grep ^CUDA_VISIBLE_DEVICES=)"
  done'

srun --jobid="$JOBID" --overlap nvidia-smi \
  --query-compute-apps=pid,used_memory --format=csv
```

期望：python 进程（两个 trainer 及其 DataLoader worker）都只带单值
`CUDA_VISIBLE_DEVICES=0` 或 `=1`，不出现 `0,1`；`nvidia-smi` 只显示两个 trainer pid
各自占用一张卡。（`pgrep -f train_dp.py` 也会匹配 DataLoader worker 和检查用 shell 自身，
上面的过滤器只保留 python 进程。）

**1d. 训练产物。**

```bash
for backbone in unet transformer mlp; do
  run="$SMOKE_ROOT/peginsertionside_rgb_${backbone}_n100_s1"
  ls "$run/checkpoints/final.pt" "$run/checkpoints/resume.pt" "$run/summary.json"
  cat "$run/run.json" | .venv/bin/python -c 'import json,sys; d=json.load(sys.stdin)["config"]; print(d["train"]["total_iters"], d["policy"]["backbone"])'
done
```

期望：3 个 run 都有 `final.pt` / `resume.pt` / `summary.json`，且 `run.json` 记录
`total_iters=200` 与对应 backbone。

## Round 2：抢占、requeue 与续训

先清空 smoke root，重新提交与 Round 1 相同的命令，等两个 trainer 都进入训练循环后故意中断：

```bash
rm -rf "$SMOKE_ROOT"
sbatch --export=ALL,TASK=peginsertionside,EXPERIMENT=configs/experiments/smoke.toml,DATA_ROOT="$DATA_ROOT",RUN_ROOT="$SMOKE_ROOT" \
  slurm/train_dual_gpu.sbatch \
  --set train.total_iters=200 --set 'train.validation_steps=[200]' \
  --set 'train.checkpoint_steps=[200]'
export JOBID=<新的 JOBID>

# 等到两个 trainer 都进入训练循环（各自的 logs/<run>.log 出现 [000001/...]）后再发信号。
# 不要在数据集加载阶段就发：那时 trainer 还没安装 USR1 handler，会被默认动作杀死并
# 记为 failed（该 run 下次分配会从头重训，但日志不符合本步骤期望）。
# 普通 scancel "$JOBID" 会直接取消作业，不会给 trainer 写 checkpoint 的机会：
scancel --signal=USR1 --batch "$JOBID"
```

**2a. trainer 写完 checkpoint 才退出。**

```bash
ls -l "$SMOKE_ROOT"/*/checkpoints/resume.pt
tail -40 "slurm-dp-rgb-train-dual-${JOBID}.out"
```

期望：运行中的两个 run 都写出新的 `resume.pt`；顶层日志出现
`received signal ...; no new runs will start` 和
`[worker N] interrupted <run> exit=75 (checkpoint saved; requeue to continue)`；
未被领取的 run 记为 `interrupted`。作业以 75 退出并被 requeue（见 `2b`）。

**2b. 重新提交后跳过已完成、续训中断。**

```bash
squeue -u "$USER"           # requeue 后作业重新出现；若已被取消则重跑 Round 1 的 sbatch 命令
tail -f "slurm-dp-rgb-train-dual-${JOBID}.out"
```

期望：下一次分配的计划阶段把已有 `final.pt` 的 run 记为 `skipped`，把有 `resume.pt` 的 run
继续训练（命令仍是 `--resume auto`），汇总形如：

```text
completed: 3 skipped: 1 failed: 0 interrupted: 0
```

具体数字取决于中断前有几个 run 已经完成；无论多少，不应出现 `failed`，也不应重新训练已有
`final.pt` 的 run（对照 `run.json` 的 `started` 时间与 metrics 行数）。

## Round 3：三种 backbone 回归与结果对账

Round 1/2 让 3 个 run 全部完成后：

```bash
.venv/bin/python scripts/check_experiment.py --experiment smoke --task peginsertionside \
  --run-root "$SMOKE_ROOT" \
  --set train.total_iters=200 --set 'train.validation_steps=[200]' \
  --set 'train.checkpoint_steps=[200]'
```

期望：`3 cells ok`，`--run-root` 用 `run.json` 与同一组 `--set` 对账，无 drift、无非零退出。

可选的快速闭环评估（`SPLIT=train` 用 smoke spec 声明的 1 个共享训练 seed，避免 test split
的 100 个 episode）：

```bash
sbatch --export=ALL,TASK=peginsertionside,EXPERIMENT=configs/experiments/smoke.toml,RUN_ROOT="$SMOKE_ROOT",SPLIT=train,NUM_ENVS=2 \
  slurm/eval_dual_gpu.sbatch

for backbone in unet transformer mlp; do
  ls "$SMOKE_ROOT/peginsertionside_rgb_${backbone}_n100_s1/eval/train_final.json"
done
```

## 清理

```bash
rm -rf "$SMOKE_ROOT"
```

## 通过标准

- [ ] `squeue` 只有 1 个作业，ID 无数组后缀。
- [ ] 顶层日志同时出现 `[worker 0] starting` 与 `[worker 1] starting`。
- [ ] `/proc/<pid>/environ` 显示两个 trainer 各只看到一张 GPU（单值 `0` / `1`）。
- [ ] 第三个 run 在先完成的 worker 上立即开始（动态补位）。
- [ ] `scancel --signal=USR1 --batch` 后两个 trainer 都写出 `resume.pt`，作业 requeue。
- [ ] 再次运行时 completed run 记为 `skipped`、中断 run 续训，无重复训练。
- [ ] unet / transformer / mlp 都有 `final.pt`、`resume.pt`、`summary.json`。
- [ ] `check_experiment.py --run-root` 输出 `3 cells ok`。
- [ ] （可选）`SPLIT=train` 评估产出 `eval/train_final.json`。

全部通过后才用 `slurm/train_dual_gpu.sbatch` 提交正式 100k-step sweep；正式提交不带
`--set` 预算覆盖，`RUN_ROOT` 使用同一个目录（本集群用 `$HOME/dp-runs`，集群没有
`/scratch`）并在 data-size 与 backbone 之间保持一致。
