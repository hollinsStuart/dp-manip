# Slurm 单作业双 GPU 启动器重构计划

## 1. 这项任务要做什么

这项任务要把当前依赖 Slurm Job Array 的实验启动方式，改造成：

```text
只向 Slurm 提交 1 个作业
        │
        ├── GPU 0：一次运行一个独立训练任务
        └── GPU 1：一次运行一个独立训练任务
```

两张 GPU 共用一个待运行队列。某张 GPU 完成当前训练后，立即从队列中领取下一个任务，不需要等待另一张 GPU。这样既能满足当前账户“最多只能提交 1 个 Slurm 作业”的 QOS 限制，又能让 2 张 GPU 尽量持续工作。

本次工作只重构集群调度和启动流程，不改变模型结构、训练超参数或科学实验设计。

---

## 2. 背景与已经确认的集群限制

### 2.1 每个用户最多只能提交一个作业

在用户没有其他排队或运行中作业时：

```bash
sbatch --array=0-1%2 ./test_gpu.sbatch
```

仍会因为以下限制而失败：

```text
QOSMaxSubmitJobPerUserLimit
Batch job submission failed: Job violates accounting/QOS policy
```

但只提交一个数组元素能够成功。因此，当前账户/QOS 应按以下约束设计：

```text
每个用户最多有 1 个已提交 Slurm 作业
```

这意味着即使把数组并发量设为 `%2` 或 `%4`，多元素 Job Array 仍然不可用。

### 2.2 单个作业可以申请两张 GPU

以下资源申请已经实测成功：

```bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
```

作业内能同时看到两张 RTX 4080。申请 3 张 GPU 不可用，因此当前方案按以下上限设计：

```text
单个作业最多实际使用 2 张 GPU
```

除非以后集群配置发生变化，否则新的生产启动器固定以“两张 GPU、八个 CPU、64 GB 内存”为目标资源。

---

## 3. 为什么必须重构

当前训练入口 `slurm/train_array.sbatch` 的结构是：

```text
Slurm Array
├── 数组任务 0 → 1 张 GPU → 1 个训练
├── 数组任务 1 → 1 张 GPU → 1 个训练
├── 数组任务 2 → 1 张 GPU → 1 个训练
└── 数组任务 3 → 1 张 GPU → 1 个训练
```

Slurm 会把每个数组元素视为一个已提交作业，因此该设计与当前 QOS 不兼容。

新的结构应为：

```text
单个 Slurm 作业
│
├── GPU 0 worker
│   ├── 运行 A
│   ├── 运行 C
│   └── ……
│
└── GPU 1 worker
    ├── 运行 B
    ├── 运行 D
    └── ……
```

Slurm 只看到一个作业，但作业内最多同时存在两个彼此独立的训练进程，每个训练进程只能看到一张 GPU。

---

## 4. 必须保持不变的实验设计

### 4.1 数据量实验

每个 ManiSkill task 的实验组合为：

| 数据量 | seed 数量 | 运行数 |
|---:|---:|---:|
| N=25 | 3 | 3 |
| N=50 | 3 | 3 |
| N=100 | 5 | 5 |
| N=200 | 5 | 5 |
| 合计 |  | 16 |

### 4.2 Backbone 实验

| Backbone | seed 数量 | 运行数 |
|---|---:|---:|
| UNet | 5 | 5 |
| Transformer | 5 | 5 |
| MLP | 5 | 5 |

其中 Backbone 实验的 5 个 UNet 运行与数据量实验中的 N=100 UNet 完全相同，必须复用已有结果，不得重复训练。因此每个 task 真正新增的 Backbone 训练只有：

```text
Transformer 5 次 + MLP 5 次 = 10 次
```

每个 task 实际训练总量约为：

```text
数据量实验 16 次 + 新增 Backbone 实验 10 次 = 26 次
```

PegInsertionSide 的 100k step 基准约为每次 3.4–3.7 小时，一个 task 约需 90–92 GPU 小时。若两张 GPU 持续工作，理想墙钟时间约为 46 小时。因此启动器必须支持作业到时后重新提交，并安全地继续未完成工作。

---

## 5. 重构原则

### 5.1 实验配置只有一个事实来源

不得在新调度器中手写或复制以下映射：

- task；
- experiment；
- value；
- seed；
- `num_demos`；
- 输出目录；
- 配置覆盖项。

现有 `configs/tasks/*.toml`、`configs/experiments/*.toml` 和 `scripts/sweep.py` 已经负责生成实验网格及解析运行名称。新调度器应直接复用这些逻辑，必要时把它们重构成更清晰的可复用函数，但不能建立第二套实验定义。

### 5.2 保留现有运行身份和目录语义

继续使用当前配置解析和 `default_run_name()` 生成运行目录。不能随意修改 `RUN_ROOT`、实验身份或输出命名。

这样可以保证：

```text
data_size / N=100 / UNet
              ↓ 同一份 resolved config 和运行目录
backbone / UNet
```

Backbone 实验发现已有 `final.pt` 后会直接复用，而不是重新训练。

### 5.3 复用现有完成和续训语义

当前训练器已经提供：

- 同一配置存在 `checkpoints/final.pt`：认为运行完成并跳过；
- 存在 `checkpoints/resume.pt` 且使用 `--resume auto`：从已有状态续训；
- 完成文件与当前配置不一致：不能错误跳过。

新调度器只负责判断哪些运行需要入队，并调用现有训练入口。不得引入另一种不兼容的 `.done` 文件或完成状态格式。

### 5.4 单个训练进程只能看到一张 GPU

两个 worker 启动子进程时分别设置：

```text
worker 0 → CUDA_VISIBLE_DEVICES=0
worker 1 → CUDA_VISIBLE_DEVICES=1
```

每个子进程内部仍使用普通的 `cuda` 设备，但只能发现自己被分配的那张物理 GPU。不能让两个训练器同时抢占两张卡。

### 5.5 使用动态队列，不做静态配对

不能把运行固定写成 `(A, B)`、`(C, D)` 这样的批次。正确行为是共享队列：

```text
待运行：A B C D E

GPU 0 → A
GPU 1 → B

B 先完成：GPU 1 立即领取 C
A 后完成：GPU 0 立即领取 D
```

这样 Transformer、MLP、UNet 运行时间不同时，也不会让较早空闲的 GPU 等待。

---

## 6. 分批执行与验收安排

### 6.1 协作方式

整个重构拆成 9 个小任务。每次只执行一个任务，完成后暂停，由验收方检查代码、测试和行为；当前任务通过后再开始下一个任务。

每轮遵循以下规则：

1. 不提前实现后续任务；
2. 不顺手修改模型或实验超参数；
3. 保留当前分支中与本任务无关的已有改动；
4. 新行为必须有自动化测试；
5. 旧入口必须继续工作，除非当前任务明确要求替换；
6. 执行者完成后只需说明“任务 N 已完成”，验收方会直接检查共享工作区中的 diff 并运行测试；
7. 验收结果只有三种：通过、需要小修、退回重新设计；
8. 未通过验收时，不进入下一任务。

推荐每个任务通过验收后单独提交一次 Git commit，便于回退和定位问题；但是否提交由执行者决定，验收本身不要求必须 commit。

### 6.2 任务依赖关系

```text
任务 1：稳定运行清单接口
  ↓
任务 2：完成状态与待运行计划
  ↓
任务 3：通用双 worker 动态队列
  ↓
任务 4：接入真实训练与 GPU 隔离
  ↓
任务 5：中断、checkpoint 与重新提交
  ↓
任务 6：生产 Slurm 脚本与 CPU 配置
  ↓
任务 7：评估流程迁移
  ↓
任务 8：完整本地回归与文档
  ↓
任务 9：集群 smoke test 和最终验收
```

任务 1–8 是代码与本地测试阶段；任务 9 必须在 HKU 集群上执行。

### 6.3 任务 1：稳定运行清单接口，并增加 task 过滤

#### 本次要做

1. 整理 `scripts/sweep.py` 中现有 `Run`、`runs()`、配置解析和运行命名逻辑，使后续调度器可以直接复用；
2. 为 sweep CLI 增加语义化的 `--task peginsertionside` 过滤；
3. 未提供 `--task` 时保持当前全任务行为不变；
4. 保持现有 `show`、`train --index`、`eval --index` 入口可用；
5. 增加测试，证明过滤前后的运行集合和顺序正确；
6. 增加测试，证明数据量实验的 N=100 UNet 与 Backbone 实验的 UNet 仍解析到相同运行名称和目录身份。

#### 本次不要做

- 不实现双 worker；
- 不创建新的 Slurm 脚本；
- 不设置 `CUDA_VISIBLE_DEVICES`；
- 不修改 checkpoint/续训逻辑；
- 不修改 DataLoader worker 数量；
- 不迁移评估数组；
- 不改变任何 TOML 实验定义。

#### 验收标准

- [ ] `data_size.toml` 在不指定 task 时仍生成 96 个运行；
- [ ] `backbone.toml` 在不指定 task 时仍生成 90 个运行；
- [ ] `--task peginsertionside` 对 data-size 生成 16 个运行；
- [ ] `--task peginsertionside` 对 backbone 生成 15 个运行；
- [ ] value、seed、run name 与修改前完全一致；
- [ ] N=100 UNet 与 Backbone UNet 的 5 个 seed 分别具有相同 run name；
- [ ] 不带 `--task` 的旧命令行为不变；
- [ ] 非法 task 会给出清楚错误并返回非零；
- [ ] `tests/test_sweep.py` 及相关配置测试通过；
- [ ] diff 中没有任务 2–9 的提前实现。

#### 验收时会执行

```bash
python scripts/sweep.py show --experiment configs/experiments/data_size.toml
python scripts/sweep.py show --experiment configs/experiments/data_size.toml --task peginsertionside
python scripts/sweep.py show --experiment configs/experiments/backbone.toml --task peginsertionside
python -m unittest tests.test_sweep tests.test_config
```

验收方还会检查 diff，确认运行顺序、默认行为和实验身份没有漂移。

### 6.4 任务 2：建立完成状态检查与待运行计划

#### 本次要做

1. 基于任务 1 的运行清单，为每个运行生成 `completed` 或 `pending` 状态；
2. 复用训练器当前对 `final.pt` 与配置一致性的判断，不能只因为文件存在就错误跳过；
3. 保留 `resume.pt` 对应运行为 pending，由现有 `--resume auto` 负责续训；
4. 提供不启动训练的 `plan` 或 `--dry-run` 输出；
5. 输出总数、completed/skipped 数和 pending 数；
6. 用临时运行目录测试完成、缺失、可续训、配置不匹配四种情况。

#### 本次不要做

- 不启动并发进程；
- 不绑定 GPU；
- 不修改 Slurm 文件；
- 不发明新的 `.done` 文件。

#### 验收标准

- [ ] 完成状态与现有 trainer 语义一致；
- [ ] 配置匹配的 `final.pt` 被列为 skipped；
- [ ] `resume.pt` 存在但没有 `final.pt` 时仍列为 pending；
- [ ] 配置不匹配的 `final.pt` 不会被静默跳过；
- [ ] dry-run 不会启动训练或改写运行目录；
- [ ] 输出可清楚看到每个运行及汇总。

### 6.5 任务 3：实现可测试的双 worker 动态队列

#### 本次要做

1. 实现最多两个 worker 的共享 pending queue；
2. 使用短小的假命令测试，不接真实训练和 GPU；
3. 证明先完成的 worker 会立即领取下一个任务；
4. 为每个运行保存独立 stdout/stderr 日志；
5. 记录 run ID、worker ID、开始、完成、失败和退出码；
6. 一个任务失败后继续独立剩余任务；
7. 最终打印 completed/skipped/failed 汇总；
8. 任一必需任务失败时最终返回非零。

#### 本次不要做

- 不申请 GPU；
- 不运行真实 200-step 训练；
- 不处理 Slurm requeue；
- 不修改科学配置。

#### 验收标准

- [ ] 并发进程数永远不超过 2；
- [ ] 动态补位有效，不是静态成对等待；
- [ ] 每个运行有独立日志；
- [ ] 失败不会丢失，也不会被汇报为整体成功；
- [ ] 队列为空和只有一个任务时行为正确；
- [ ] 调度器测试稳定且不依赖真实 GPU。

### 6.6 任务 4：接入真实训练命令和单 GPU 隔离

#### 本次要做

1. 让队列消费现有 sweep 生成的训练命令；
2. worker 0 子进程设置 `CUDA_VISIBLE_DEVICES=0`；
3. worker 1 子进程设置 `CUDA_VISIBLE_DEVICES=1`；
4. 子进程内部继续使用现有 `--device cuda` 语义；
5. 保持 `DATA_ROOT`、`RUN_ROOT`、experiment、seed 和 run name 传递正确；
6. 提供可注入假训练命令的测试，验证两个子进程各自只看到一个逻辑 GPU；
7. 验证现有单运行入口不受影响。

#### 本次不要做

- 不加入 Slurm 资源头；
- 不完成 requeue；
- 不迁移 eval；
- 不进行 100k-step 训练。

#### 验收标准

- [ ] 两个 worker 的 GPU 环境互相隔离；
- [ ] 训练命令仍来自唯一的 sweep/config 逻辑；
- [ ] 参数中没有手写 task/value/seed 映射；
- [ ] `RUN_ROOT` 和默认运行命名保持不变；
- [ ] 单进程旧入口的测试继续通过。

### 6.7 任务 5：处理中断、checkpoint 与重新提交

#### 本次要做

1. 调度器捕获 `SIGUSR1`、`SIGTERM` 等相关信号；
2. 收到信号后停止领取新任务；
3. 向两个正在运行的 trainer 转发信号；
4. 等待子进程完成现有 `resume.pt` 写入流程；
5. 正确聚合 trainer 的 checkpoint/requeue 退出状态；
6. 增加信号测试，验证不会在 checkpoint 写完前退出；
7. 模拟第二次启动，验证 completed 被跳过、interrupted 保持 pending 并使用 `--resume auto`。

#### 本次不要做

- 不改变 trainer 的 checkpoint 格式；
- 不实现不安全的新型部分恢复；
- 不掩盖普通训练失败。

#### 验收标准

- [ ] 信号到达后不再调度新运行；
- [ ] 两个活动子进程都收到信号；
- [ ] 调度器等待子进程退出并保留退出状态；
- [ ] 第二次启动不会重跑已完成运行；
- [ ] 普通失败与“已保存 checkpoint、等待 requeue”能够区分。

### 6.8 任务 6：增加生产 Slurm 入口并调整 CPU worker 配置

#### 本次要做

1. 新增 `slurm/train_dual_gpu.sbatch`；
2. 固定申请 2 GPU、8 CPU、64 GB RAM；
3. 不使用 `#SBATCH --array`；
4. 接入任务 5 的信号与 requeue 语义；
5. 验证并传递 `TASK`、`EXPERIMENT`、`DATA_ROOT`、`RUN_ROOT`、`PYTHON`；
6. 将双 trainer 场景下每个 trainer 的 DataLoader 默认值设为 3，并继续允许配置覆盖；
7. 更新或增加 Slurm 脚本的静态测试。

#### 验收标准

- [ ] `sbatch` 文件中没有数组声明；
- [ ] 资源精确为 2 GPU、8 CPU、64 GB RAM；
- [ ] 用户不需要计算数组 index；
- [ ] 两个 trainer 不会各启动 8 个 DataLoader worker；
- [ ] 原有单运行配置仍可覆盖 `num_workers`；
- [ ] 脚本从仓库根目录和正常 `SLURM_SUBMIT_DIR` 场景均能解析路径。

### 6.9 任务 7：将评估迁移到同一单作业队列

#### 本次要做

1. 复用任务 3–4 的动态 worker 和 GPU 隔离；
2. 复用现有 eval 命令构造逻辑；
3. 保留 `CHECKPOINT`、`SPLIT`、`NUM_ENVS` 等参数；
4. 为每个评估运行保存独立日志；
5. 评估失败也要进入最终失败汇总；
6. 新增无 Job Array 的评估 Slurm 入口，或将现有文件明确改造成单作业入口。

#### 验收标准

- [ ] 完整工作流不再要求提交多元素 eval array；
- [ ] train/eval 共享运行清单和调度核心，没有两套队列；
- [ ] checkpoint 路径、split 与 episode 规则保持原样；
- [ ] 评估缺少 checkpoint 时给出明确失败。

### 6.10 任务 8：本地全量回归与用户文档

#### 本次要做

1. 运行与 sweep、config、checkpoint、resume、三种 backbone、实验漂移检查有关的全部测试；
2. 运行仓库可在当前环境中执行的完整测试集；
3. 更新 README，给出训练、评估、续跑、日志和故障查看命令；
4. 说明同一 `RUN_ROOT` 对 N=100 UNet 复用的重要性；
5. 准备集群 smoke 配置和逐项检查表；
6. 检查 diff，确认没有科学参数漂移。

#### 验收标准

- [ ] 相关自动化测试全部通过；
- [ ] `check_experiment.py` 对声明配置检查通过；
- [ ] README 中只有可在当前 QOS 下工作的推荐命令；
- [ ] 旧入口的兼容状态被明确记录；
- [ ] 集群 smoke test 步骤可直接照着执行。

### 6.11 任务 9：集群 smoke test 与最终验收

#### 本次要做

1. 使用 200 iterations 配置提交单个 Slurm 作业；
2. 证明 Slurm 队列中只有 1 个作业；
3. 证明作业内有两个独立 trainer；
4. 用日志或 `nvidia-smi` 证明每个 trainer 只看到一张 GPU；
5. 证明先完成的 GPU 会立即领取下一项；
6. 故意中断并重新提交；
7. 证明 completed 被跳过，interrupted 能续训；
8. 分别跑通 UNet、Transformer、MLP 的训练、验证与 checkpoint 保存；
9. 运行 `check_experiment.py --run-root ...` 检查实际结果；
10. 检查每个运行日志和最终汇总。

#### 最终验收材料

- Slurm job ID 和 `squeue` 证据；
- 顶层 scheduler 日志；
- 两个同时运行任务的独立日志；
- GPU 可见性证据；
- 中断前后两次启动的日志；
- completed/skipped/failed 汇总；
- 三种 Backbone 的 smoke checkpoint；
- `check_experiment.py` 输出。

任务 9 通过后，才允许启动完整 100k-step sweep。

### 6.12 第一次执行范围

第一次只执行“任务 1：稳定运行清单接口，并增加 task 过滤”。完成任务 1 后立即暂停，不要开始写 worker、GPU 调度器或新的 Slurm 脚本。

执行者完成后告知验收方“任务 1 已完成”。验收方会按 6.3 节检查工作区 diff、运行指定测试，并给出通过或修改意见。

---

## 7. 总体技术改动说明

### 阶段一：抽取并复用运行清单逻辑

以 `scripts/sweep.py` 的 `Run`、`runs()`、配置解析和命令构造逻辑为基础，提供可复用的运行清单能力：

1. 根据 experiment 配置生成全部声明的 `(task, value, seed)`；
2. 支持按一个语义化 task 过滤，例如 `peginsertionside`；
3. 为每个运行解析 canonical config 和默认运行目录；
4. 继续使用现有命令构造逻辑启动训练或评估；
5. 让旧的单 index/单运行入口继续可用，避免破坏现有本地测试与调试流程。

用户不再需要查找和计算 `64-79`、`60-74` 等全局数组下标。

### 阶段二：实现单作业双 worker 调度器

增加一个轻量级队列调度入口。它负责：

1. 读取目标 task 和 experiment；
2. 生成 canonical 运行清单；
3. 检查现有结果，把已完成运行记为 `skipped`；
4. 把未完成或缺失运行加入 pending queue；
5. 启动两个 worker，分别绑定 GPU 0 和 GPU 1；
6. 任意 worker 完成后，立即领取下一项；
7. 收集每个子进程的退出码；
8. 所有任务结束后输出汇总；
9. 只要有必需运行失败，调度器最终返回非零退出码。

建议让调度器本身不导入重量级训练栈，只负责生成命令、管理子进程和状态。每个训练仍通过现有统一 trainer 执行。

### 阶段三：实现安全的中断与重新提交

完整 task 很可能无法在一次允许的 walltime 内结束。调度器应处理 Slurm 的提前通知信号：

1. 收到 `SIGUSR1` 或终止信号后停止领取新任务；
2. 将信号转发给当前两个训练子进程；
3. 等待训练器按现有逻辑写出 `resume.pt`；
4. 记录仍未完成的任务；
5. 让 Slurm 脚本执行现有 requeue 流程，或安全退出以便用户重新提交；
6. 下次启动时跳过已有 `final.pt` 的运行，并让中断运行通过 `resume.pt` 继续。

必须避免调度器在子进程还没有写完 checkpoint 时过早退出。

### 阶段四：新增生产 Slurm 启动脚本

新增类似 `slurm/train_dual_gpu.sbatch` 的入口，资源头使用：

```bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
```

具体 walltime 根据队列政策设置，不假设一次 allocation 能覆盖约 46 小时的全部运行。

训练并发不得再使用：

```bash
#SBATCH --array=...
```

顶层脚本验证 `DATA_ROOT`，保留 `RUN_ROOT`、`PYTHON`、信号转发及 requeue 行为，并调用新的队列调度入口。

### 阶段五：调整 DataLoader worker 配置

双 GPU 作业只有 8 个 CPU，同时运行两个独立 trainer。每个 trainer 不能再启动 8 个 DataLoader worker。

仓库当前已经通过 `train.num_workers` 配置 DataLoader，当前 baseline 值为 4。此次改动应继续使用该配置项，并把集群双训练器的合理默认值设为每个 trainer 3 个 worker；若后续基准测试发现 2 更稳定，再调整配置，而不是在训练代码中硬编码。

目标资源分配约为：

```text
调度器与主进程开销
GPU 0 trainer + 3 DataLoader workers
GPU 1 trainer + 3 DataLoader workers
```

单运行入口仍应允许通过配置覆盖为 0、2、3、4 等值。

### 阶段六：为每个运行建立独立日志

两个 trainer 的标准输出不能混写到同一个 Slurm 日志。每次运行应写入共享存储中的独立日志，例如：

```text
RUN_ROOT/
├── <run-name>/
│   └── ……
└── logs/
    ├── <run-name>.log
    └── <run-name>.log
```

日志必须存放在从 `gpu2gate1` 可见的 home、project 或 `RUN_ROOT` 文件系统中，不能只写到计算节点本地 `/tmp`。

顶层 Slurm 输出只保留调度事件，例如：

```text
Detected GPUs: 0,1
Pending runs: 14

[GPU0] starting peginsertionside/data_size/N25/seed1
[GPU1] starting peginsertionside/data_size/N25/seed2
[GPU1] completed ... exit=0
[GPU1] starting ...seed3
[GPU0] failed ... exit=1
```

### 阶段七：明确失败处理

默认采用“继续独立任务，最后统一失败”的策略：

1. 某个运行失败后，记录 run ID、GPU 编号、日志路径和退出码；
2. 另一张 GPU 上正在运行的任务不受影响；
3. 空闲 worker 继续处理其余 pending 任务；
4. 最后输出汇总，例如：

```text
completed: 14
skipped: 10
failed: 2
```

5. 如果 `failed > 0`，顶层进程返回非零退出码，Slurm 作业不能被静默标记为成功。

### 阶段八：同步处理评估流程

当前 `slurm/eval_array.sbatch` 同样依赖多元素 Job Array，因此也受到相同 QOS 限制。为了保持“一条命令可运行”的工作流，应将训练和评估共享的队列管理抽象复用到评估：

- 训练模式由每个 worker 构造训练命令；
- 评估模式由每个 worker 构造评估命令；
- 两者都使用同一套 task/experiment 运行清单、GPU 绑定、独立日志、动态领取和失败汇总；
- 评估继续保留 `CHECKPOINT`、`SPLIT`、`NUM_ENVS` 等现有参数。

如果实现阶段不能在同一改动中完成评估迁移，必须把它列为明确的后续任务，并在 README 中注明现有评估命令仍受 QOS 限制，不能继续宣称完整流程只需一条可用命令。

### 阶段九：更新使用文档

最终用户工作流应采用语义化参数。例如，运行单个 task 的数据量实验：

```bash
sbatch \
  --export=ALL,TASK=peginsertionside,EXPERIMENT=configs/experiments/data_size.toml,DATA_ROOT="$DATA_ROOT",RUN_ROOT="$RUN_ROOT" \
  slurm/train_dual_gpu.sbatch
```

运行 Backbone 实验时只切换 `EXPERIMENT`。用户无需手动计算数组范围。

文档还应说明：

- `RUN_ROOT` 必须在数据量实验与 Backbone 实验之间保持一致；
- 可重复提交同一命令；
- 已完成运行会跳过；
- 中断运行会按现有 checkpoint 语义续训；
- 每个运行的日志位置；
- 如何查看最终 completed/skipped/failed 汇总。

---

## 8. 预计涉及的文件

具体文件名可在实现时小幅调整，但职责应保持清晰。

| 文件 | 计划改动 |
|---|---|
| `scripts/sweep.py` | 抽取/扩展运行清单、task 过滤和命令构造逻辑，同时保留旧单 index 行为 |
| 新的队列调度脚本 | 管理两个 GPU worker、pending queue、信号、日志和结果汇总 |
| `slurm/train_dual_gpu.sbatch` | 单作业申请 2 GPU、8 CPU、64 GB RAM，并调用队列调度器 |
| `slurm/eval_array.sbatch` 或新的评估脚本 | 移除不可用的多元素数组工作流，接入单作业 worker 模型 |
| `configs/baseline.toml` 或集群专用覆盖 | 将双训练器场景下的 `train.num_workers` 合理设为 3，保持可配置 |
| `tests/test_sweep.py` | 验证实验清单与旧逻辑一致、task 过滤正确、UNet 复用不变 |
| 新的调度器测试 | 验证动态领取、并发上限、GPU 隔离、跳过、失败和信号处理 |
| `README.md` / Slurm 文档 | 更新用户提交、续跑、日志和评估说明 |

---

## 9. 测试与验证计划

### 9.1 本地自动化测试

至少覆盖：

1. 新旧逻辑生成的 `(task, value, seed, run name)` 完全一致；
2. 选择一个 task 时只生成该 task 的运行；
3. 数据量实验 N=100 UNet 与 Backbone UNet 解析到同一运行目录；
4. 已有且配置一致的 `final.pt` 被跳过；
5. 缺少 `final.pt` 的运行进入队列；
6. 每个 worker 的子进程环境只暴露一张 GPU；
7. 任一运行结束后，对应 worker 会领取下一个运行；
8. 同时运行的 trainer 永远不超过 2 个；
9. 单个运行失败时失败信息被保留，最终退出码非零；
10. 每个运行写入独立日志；
11. 原有单运行和单 index 脚本继续工作；
12. `check_experiment.py` 仍能验证生成结果，没有 config drift。

### 9.2 集群 smoke test

不要一开始运行 100k steps。先使用约以下规模的 smoke 配置：

```text
total_iters = 200
validation_steps = [200]
checkpoint_steps = [200]
```

分三轮验证：

#### 第一轮：双卡并发

- Slurm 中只出现 1 个作业；
- 作业内同时存在 2 个独立 trainer；
- 两个 trainer 分别只看到 1 张 GPU；
- `nvidia-smi` 与日志中的 GPU 分配一致；
- 任一短任务先结束后，空闲 GPU 自动领取下一项。

#### 第二轮：中断与续跑

- 故意中断作业；
- 确认运行中的 trainer 写出可用 `resume.pt`；
- 重新提交同一条命令；
- 已完成运行被跳过；
- 未完成运行续训或按既有规则重跑；
- 没有重复覆盖已完成结果。

#### 第三轮：三种 Backbone 回归

- UNet、Transformer、MLP 都能完成训练；
- 都能完成验证；
- 都能保存 checkpoint；
- `check_experiment.py` 检查通过。

完成以上验证后，才启动完整的 100k-step 实验。

---

## 10. 最终验收标准

- [ ] 训练只需提交 1 个 Slurm 作业。
- [ ] 训练并发不再依赖 Slurm Job Array。
- [ ] 作业申请 2 张 GPU、8 个 CPU、64 GB RAM。
- [ ] 任意时刻最多运行 2 个 trainer。
- [ ] worker 0 的 trainer 只能看到 1 张 GPU。
- [ ] worker 1 的 trainer 只能看到 1 张 GPU。
- [ ] 某个 trainer 完成后，对应 GPU 自动领取下一个 pending run。
- [ ] 现有 experiment config 仍是实验定义的唯一事实来源。
- [ ] seed、实验组合和科学超参数没有改变。
- [ ] 已完成运行会自动跳过。
- [ ] walltime 到期或中断后，重新提交能够安全继续剩余工作。
- [ ] 数据量实验 N=100 UNet 结果仍可被 Backbone 实验复用。
- [ ] `RUN_ROOT` 语义和现有运行命名保持不变。
- [ ] 每个运行拥有独立日志。
- [ ] 顶层日志清楚记录 GPU 分配、开始、完成与失败事件。
- [ ] 双 trainer 场景下不会让每个 trainer 各启动 8 个 DataLoader worker。
- [ ] 任一必需运行失败时，顶层作业最终返回非零状态。
- [ ] 现有单运行脚本仍然可用。
- [ ] UNet、Transformer、MLP smoke test 仍然通过。
- [ ] `check_experiment.py` 仍能验证结果且无配置漂移。
- [ ] 评估流程已迁移，或其剩余数组限制已被明确记录为后续任务。

---

## 11. 明确不做的事情

本次重构不修改：

- 模型架构；
- optimizer；
- learning rate；
- batch size；
- diffusion 超参数；
- seed 定义；
- dataset split；
- 实验的科学设计；
- 输出指标定义。

唯一允许触及训练配置的项目是 DataLoader worker 数量，而且必须保持独立、可配置，不得借此改变其他训练行为。

---

## 12. 完成后的效果

重构完成后，用户只需按 task 和 experiment 提交一次作业。这个作业内部会：

```text
解析现有实验配置
    ↓
找出已完成、可续训和未开始的运行
    ↓
跳过已完成运行
    ↓
将其余运行放入共享队列
    ↓
两张 GPU 动态领取任务
    ↓
分别记录日志和退出状态
    ↓
输出 completed / skipped / failed 汇总
```

因此，这项改造解决的是“当前 Slurm QOS 不允许数组多作业”的工程问题，同时保留现有实验定义、checkpoint、结果目录及 N=100 UNet 复用关系。
