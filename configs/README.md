# Layered RGB experiment configs

正式入口按以下顺序解析，并将完整结果保存进 checkpoint：

```text
baseline.toml
  + tasks/<task>.toml
  + experiment override
  + CLI runtime override
```

`baseline.toml` 是仿真/渲染、vision、policy、train、EMA、diffusion、evaluation
和通用 data 默认值的唯一权威来源。`tasks/*.toml` 只保存环境、控制模式、回合长度
和数据路径。根目录的 `*_rgb.toml` 仅为旧命令提供跳转，不含第二份 baseline 参数。
`config.load()` 会直接拒绝 task 文件里出现的任何其他 section 或 key，避免任务配置悄悄覆盖 baseline。

正式 data-size grid 定义在 `experiments/data_size.toml`，其中声明
`variable = "data.num_demos"`、实验 values 和各 value 的 replicate seeds。core 只要求
`data.num_demos` 为正整数。条件式 N=400 follow-up 单独放在
`experiments/data_size_optional400.toml`，不属于正式 grid。
两个 spec 的 `[diagnostics] train_eval_episodes = 25` 决定 `sweep.py eval --split train`
只评估按 seed 升序的前 25 个训练 seed（所有嵌套子集共有），使过拟合诊断在不同 N 间可比；
直接调用 `eval_dp.py --split train` 默认评估该 checkpoint 的全部训练 seed。

轨道 B（模型结构）定义在 `experiments/backbone.toml`：`variable = "policy.backbone"`、
`values = ["unet", "transformer", "mlp"]`，三个 arm 都使用训练种子 1–5。结构参数统一由
`baseline.toml` 的 `policy.transformer_*` / `policy.mlp_*` 解析，因此三个 arm 的 resolved
config 只在 `policy.backbone` 上不同。N_B 默认使用 baseline 的 `data.num_demos = 100`；
六任务基线跑完后，UNet 在 100 条时成功率均值低于 0.10 的难任务按 `docs/final-plan.md` §6
改用 200 条（运行时另加 `--num-demos 200`，unet arm 直接复用轨道 A 的 N=200 格子）。
`[diagnostics] train_eval_episodes = 25` 与 data-size grid 相同，用于在共有的前 25 个训练
seed 上做过拟合诊断。

`experiments/smoke.toml` 是集群 smoke 用的缩小网格（`policy.backbone` 三个 arm × seed 1），
训练预算通过 `--set train.total_iters=...` 等运行时覆盖传入，不写进正式实验定义；
步骤见 `docs/cluster-smoke-test.zh-CN.md`。

`data.num_demos=N` 固定选择按 `episode_seed` 升序排序后的前 N 条示范，与 HDF5 导出
顺序和 `episode_id` 无关，因此 `25 ⊂ 50 ⊂ 100 ⊂ 200` 对任何导出结果都成立；
`episode_id` 与 `episode_seed` 在同一 split 内必须唯一，重复会直接报错。
`run.json` 的 `data_selection` 记录实际选中的 `demo_seeds`，
`tests/test_data_nesting.py` 与 `scripts/inspect_dataset.py` 负责校验该不变量。

例如，解析 PickCube 的 N=50 数据量实验：

```bash
python scripts/train_dp.py \
  --config configs/tasks/pickcube.toml \
  --experiment configs/experiments/data_size.toml \
  --experiment-value 50
```

`backbone.toml` 的 value 是字符串，统一入口直接按 value 选择：

```bash
python scripts/run_experiment.py \
  --task pickcube --experiment backbone --value transformer --seed 1
```

`--value` 对 data_size 同样适用（`--experiment data_size --value 50`），字符串形式会匹配
spec 里声明的整数；`scripts/train_dp.py --experiment-value` 也接受同样的字符串。

`--set SECTION.KEY=VALUE`、`--seed`、`--num-demos` 是最后应用的运行时覆盖。
它们不能覆盖所选实验的变量本身（例如 data-size 实验里的 `--num-demos`），否则直接报错；
实验变量只能通过 `--value` / `--experiment-value` 选择。

新解析的配置里每个值都必须来自 `baseline.toml`（代码不提供默认值）。读取已记录的
checkpoint、`resume.pt`、`run.json` 时用 `config.from_recorded`：对后来才新增的字段
（如 `policy.backbone`、backbone 结构参数、`train.betas`）补上这些 run 当时实际使用的值。
集群上可用 `--data-root` 覆盖数据根目录。临时 smoke 可用
`--set train.total_iters=...`；正式实验仍使用 baseline 的固定训练预算。

新实验可加 `--set data.preload=true`：训练开始前把所选 episode 的 RGB 一次性解码进内存
（每条 demo 约 15 MB，train 和 val 都会加载），DataLoader worker 不再逐窗口解压 gzip。
默认 `false`，即原来的逐窗口懒读。读出的样本逐位相同，所以它和 `data.root` 一样属于运行时
字段，不进 `control_hash`，也不算 drift；但同一个 run 续跑（resume）时要保持同样的取值。

提交正式实验前，用 `scripts/check_experiment.py` 自动做 Gate B 检查：它解析 spec 声明的
所有 `(value, seed)` cell，只允许声明的实验变量、replicate seed 和运行时 `data.root` / `data.preload` 不同；
backbone 实验另外允许 `policy.unet_*` / `policy.transformer_*` / `policy.mlp_*` 结构参数不同
（计划中的 `policy.backbone.*`），data-size 等其他实验里这些结构参数也必须一致。其余差异以
per-key 矩阵报错并返回非零状态；同时按 Phase 13/§19 输出每个任务的 `control_hash`（同一矩阵的
所有 cell 必须相同）。

加 `--run-root` 时改为检查每个 cell 实际训练用的配置（`<run-root>/<run name>/run.json`）：
各 cell 之间互相比较，并与当前声明比较，所以能发现提交时的 `--set`/`--num-demos` 或
在旧版 `baseline.toml` 下训练的 run；还没有 `run.json` 的 cell 单独列出。汇总结果前跑一次。

```bash
python scripts/check_experiment.py --experiment data_size
python scripts/check_experiment.py --experiment backbone
python scripts/check_experiment.py --experiment backbone --run-root /scratch/$USER/dp-runs
```

每个 run 的 `run.json` 也会记录 Phase 14（§19）的 `experiment_context`（name / variable /
value / seed / `control_hash`，与 checker 共用 `dp_manip.invariants` 的同一套规则）、
`git` revision 和数据集 `fingerprint`，所以 `--run-root` 的对账和事后审计不需要重新推导实验网格。
