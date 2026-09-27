# 7606C Unified Experiment Pipeline Refactor Plan

## 1. 目标

当前仓库同时存在：

- `dp-manip`
- `VariDP`

两套相对独立的 Diffusion Policy 实现。

本次 refactor 的最终目标不是单纯整理目录，而是把它们逐步收敛成：

```text
One Shared RGB Diffusion Policy Pipeline
                +
Multiple Experiment Configurations
```

最终所有正式实验都必须经过同一套：

```text
RGB Dataset
    ↓
Observation Encoder
    ↓
Observation Features
    ↓
Diffusion Policy
    ↓
Configurable Noise-Prediction Backbone
    ├── UNet
    ├── Transformer
    └── MLP
    ↓
Trainer
    ↓
Evaluator
```

实验之间只能修改明确声明的 experimental variable。

例如：

```text
data_size:
    num_demos = 25 / 50 / 100 / 200

backbone:
    UNet / Transformer / MLP

failure_case:
    failure data construction
    failure loss
    auxiliary head
```

禁止每个 experiment 拥有自己的 trainer / dataset / evaluation pipeline。

---

# 2. 当前基线

工作分支：

```text
refactor/dp-manip-rgb-cluster
```

当前 `dp-manip` 已完成大部分 RGB migration，可以作为未来 canonical implementation 的基础。

当前已经具备：

```text
RGB HDF5 dataset
lazy loading
train/validation split
RGB preprocessing
visual encoder
proprioception conditioning
Conditional UNet
DDPM
EMA
fixed-step training
checkpoint / resume
cluster / Slurm support
closed-loop evaluator
```

当前 RGB baseline 的重要参数已经基本统一，例如：

```text
obs horizon        = 2
prediction horizon = 16
action horizon     = 8

UNet dims          = [256, 512, 1024]
diffusion embed    = 256
kernel size        = 5
groups             = 8

diffusion steps    = 100

training steps     = 100000
batch size         = 64
learning rate      = 1e-4
```

因此：

> 不重新实现 RGB pipeline。

后续工作应该以当前 `dp-manip` RGB pipeline 为 canonical core 候选进行抽象。

---

# 3. 核心原则

整个重构期间必须遵守以下 invariants。

## Invariant 1 — One Trainer

所有正式实验必须进入同一个 Trainer。

禁止：

```text
data_size_trainer.py
backbone_trainer.py
failure_case_trainer.py
```

允许：

```text
trainer.py
```

实验差异只能来自 config。

---

## Invariant 2 — One RGB Observation Pipeline

所有正式实验必须使用完全相同的：

```text
RGB loading
RGB preprocessing
augmentation
normalization
visual encoder
proprioception handling
```

特别是在 backbone comparison 中：

```text
UNet
Transformer
MLP
```

必须使用同一个 visual encoder。

否则实验变量不再只是 backbone。

---

## Invariant 3 — Explicit Experimental Delta

一次实验必须可以表示为：

```text
resolved_config
=
canonical_baseline
+ task_config
+ experiment_override
+ replicate_seed
```

例如 backbone experiment：

```text
baseline
+ PickCube
+ policy.backbone=transformer
+ seed=1
```

除了：

```text
policy.backbone
seed
runtime metadata
```

其他 scientific parameters 必须相同。

---

## Invariant 4 — Refactor Must Not Change Hyperparameters

refactor 不要求训练行为在数值上保持不变（RGB pipeline 本身就是新实现）。

但禁止在 architecture refactor 中顺便：

```text
换 optimizer
换 scheduler
换 EMA
换 diffusion scheduler
换 horizon
换 normalization
换 dataset split
```

如果某个 Phase 确实需要改其中之一，必须单独说明原因，并直接修改 `baseline.toml`，
不能在代码里另起一份。

---

## Invariant 5 — No Big-Bang Rewrite

迁移必须 incremental。

任何阶段结束后都应该存在至少一条可运行 pipeline。

---

# 3.5 统一阶段验收标准

整个重构最终只确认两件事：

```text
1. 能跑
2. 超参统一
```

RGB pipeline 是从 state training 迁移来的新实现，没有需要逐位保持的已验证结果。
因此所有 Phase 的验收都放宽为下面两道 gate。各 Phase 的 Acceptance 只补充该阶段
特有的最小检查；与本节冲突时以本节为准。

## Gate A — 能跑

- 现有 unit tests 全部通过。
- 用最小 smoke 配置走通：

```text
train → save checkpoint → load checkpoint → evaluate
```

- 只要求不报错、loss 为有限值。不比较 loss 数值、checkpoint hash 或成功率。
- 只改 config / 文档 / 测试的 Phase：unit tests 通过、config 能正常解析即可。

smoke 命令：

```bash
python scripts/train_dp.py \
    --config configs/tasks/pickcube.toml \
    --experiment configs/regression/pickcube_phase0.toml \
    --data-root "$DATA_ROOT" \
    --device cpu \
    --exp smoke_<phase>

python scripts/eval_dp.py runs/smoke_<phase>/checkpoints/final.pt \
    --split val --episodes 1 --num-envs 1 --device cpu --render-backend cpu
```

## Gate B — 超参统一

- `configs/baseline.toml` 里的超参没有被改动，除非该 Phase 明确以此为目标并写明原因。
- 训练和评估用到的参数全部来自 resolved config
  （baseline → task → experiment → CLI），代码里不另写一份硬编码超参。
- 同一 experiment 的不同 arm 之间，resolved config 只在声明的实验变量、
  replicate seed 和 runtime 字段上不同。
- 不出现第二套 dataset / observation encoder / trainer / evaluator。

## 不作为验收要求

以下内容可以做，但不阻塞 Phase 完成，统一留到 M17 清理：

```text
新旧实现数值等价 / bitwise 一致
与 Phase 0 冻结的 loss 曲线或 checkpoint hash 对比
续训随机轨迹与连续训练完全一致
命名、注释、文档中残留的旧术语
防御性校验、报错信息、边界情况测试
```

## 正式实验前的一次性检查

提交正式集群实验之前，额外做一次：

- 在 GPU 上跑一遍 Gate A（batch 64 短 smoke，记录峰值显存）。
- 对整个 experiment matrix 做一次 Gate B 的 config drift 检查。

---

# 4. Target Architecture

最终推荐结构：

```text
7606C/
│
├── dp_policy/
│   │
│   ├── config/
│   │   ├── loader.py
│   │   ├── schema.py
│   │   └── validation.py
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   ├── rgb.py
│   │   ├── normalization.py
│   │   └── transforms.py
│   │
│   ├── models/
│   │   ├── observation_encoder.py
│   │   ├── policy.py
│   │   │
│   │   └── backbones/
│   │       ├── base.py
│   │       ├── unet.py
│   │       ├── transformer.py
│   │       └── mlp.py
│   │
│   ├── diffusion/
│   │   ├── scheduler.py
│   │   └── sampling.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── ema.py
│   │   └── checkpoint.py
│   │
│   └── evaluation/
│       └── evaluator.py
│
├── configs/
│   ├── baseline.toml
│   │
│   ├── tasks/
│   │   ├── pickcube.toml
│   │   ├── pushcube.toml
│   │   ├── stackcube.toml
│   │   ├── peginsertionside.toml
│   │   ├── plugcharger.toml
│   │   └── ...
│   │
│   └── experiments/
│       ├── data_size.toml
│       ├── backbone.toml
│       └── failure_case.toml
│
├── scripts/
│   ├── run_experiment.py
│   └── sweep.py
│
├── tests/
│
└── legacy/
```

注意：

> 不要求第一阶段立刻移动成这个目录。

这是 target state，不是第一步。

---

# 5. Phase 0 — Freeze Current RGB Baseline

## 目标

在进一步抽象前，先记录当前 `refactor/dp-manip-rgb-cluster` 的行为。

避免以后不知道 refactor 是否改变了实验。

## 工作

记录：

```text
dataset schema
normalization behavior
RGB preprocessing
augmentation
visual encoder
UNet architecture
optimizer
scheduler
EMA
DDPM scheduler
training steps
batch size
horizons
evaluation seeds
checkpoint format
```

选择一个很小的 regression configuration，例如：

```text
PickCube
10–25 demos
small training budget
fixed seed
```

保存：

```text
resolved config
initial loss
short training loss curve
checkpoint
evaluation output
```

这不是追求性能。

目的是建立：

```text
RGB baseline regression reference
```

## Acceptance

必须能执行：

```text
train
→ save checkpoint
→ load checkpoint
→ evaluate
```

记录下来的 loss 曲线和 checkpoint hash 只作参考，不是后续 Phase 的回归目标。

---

# 6. Phase 1 — Config Single Source of Truth

这是下一步最高优先级工作。

当前每个 task TOML 中重复：

```text
vision
policy
train
eval
```

这会产生 config drift。

## 目标结构

```text
configs/
├── baseline.toml
└── tasks/
```

### baseline.toml

保存所有 experiment 默认保持不变的参数：

```toml
[vision]
...

[policy]
...

[train]
...

[ema]
...

[diffusion]
...

[eval]
...
```

### task config

只包含任务天然不同的参数：

```toml
[task]
env_id = "PickCube-v1"
control_mode = "..."
max_episode_steps = ...

[data]
train_path = "..."
val_path = "..."
```

## Config resolution

实现：

```text
baseline
    ↓ merge
task
    ↓ merge
experiment override
    ↓ merge
CLI runtime override
```

得到：

```text
resolved_config
```

训练和 evaluation 都只读取 resolved config。

## Acceptance

通用标准见 §3.5（Gate A 能跑 + Gate B 超参统一）。本阶段额外检查：

对于：

```text
PickCube
StackCube
```

如果不考虑 task-specific 字段：

```text
vision
policy
train
eval
```

必须来自同一个 baseline source。

---

# 7. Phase 2 — Remove Data-Size Experiment Knowledge From Core

当前 core 仍然知道：

```text
25 / 50 / 100 / 200 / 400
```

这些不是 Diffusion Policy 的约束。

这是 data-size experiment 的实验设计。

## 修改

当前类似：

```python
num_demos in {25, 50, 100, 200, 400}
```

改成 core 只验证：

```python
num_demos > 0
```

实验 grid：

```text
25
50
100
200
```

移入：

```text
configs/experiments/data_size.toml
```

或对应 experiment specification。

## Sweep

现有 data-size sweep 不能继续充当通用 sweep。

改成类似：

```text
experiments/data_size
```

定义：

```text
variable = data.num_demos
values = [25, 50, 100, 200]
```

## Acceptance

通用标准见 §3.5（Gate A 能跑 + Gate B 超参统一）。本阶段额外检查：

Core 不应该出现：

```text
25
50
100
200
400
```

这些具体实验值。

---

# 8. Phase 3 — Clean RGB / Proprio Terminology

当前 HDF5 schema 里可能存在：

```text
obs_rgb/state
```

但其实际语义是：

```text
non-privileged proprioception
```

为了避免未来误认为 policy 仍然依赖 simulator state：

内部统一改名：

```text
state        → proprio
state_mean   → proprio_mean
state_std    → proprio_std
```

HDF5 adapter 负责：

```text
obs_rgb/state
      ↓
proprio
```

正式 policy 内禁止继续叫 privileged `state`。

## Acceptance

通用标准见 §3.5（Gate A 能跑 + Gate B 超参统一）。本阶段额外检查：

模型和 trainer 的接口（batch key、函数参数）使用：

```text
rgb
proprio
```

HDF5 adapter 内部、注释和文档中残留的 `state` 不阻塞。

---

# 9. Phase 4 — Strengthen Data-Size Experiment Invariants

data-size 实验要求：

```text
25 ⊂ 50 ⊂ 100 ⊂ 200
```

因此 dataset subset 必须 deterministic。

不要依赖：

```text
episode_id 恰好等于 seed 顺序
```

## 推荐

显式记录：

```text
episode seed
```

并：

```python
episodes = sorted(episodes, key=lambda x: x.seed)
episodes = episodes[:num_demos]
```

最好同时 assertion：

```text
selected seeds
```

写进 run metadata。

例如：

```json
{
  "num_demos": 50,
  "demo_seeds": [...]
}
```

## Acceptance

通用标准见 §3.5（Gate A 能跑 + Gate B 超参统一）。本阶段额外检查：

对同一 dataset：

```text
N=25 subset of N=50
N=50 subset of N=100
N=100 subset of N=200
```

自动 test 验证。

---

# 10. Phase 5 — Improve Resume Reproducibility

当前 Slurm resume 已能恢复：

```text
model
optimizer
scheduler
EMA
scaler
step
```

建议进一步保存 RNG state：

```text
Python RNG
NumPy RNG
torch CPU RNG
torch CUDA RNG
```

恢复 checkpoint 时同时恢复。

## 目标

尽可能保证：

```text
continuous run
```

与：

```text
run
→ preempt
→ resume
```

具有一致 stochastic trajectory。

即使无法做到 bitwise identical，也应该最大程度减少：

```text
结果依赖被 Slurm 抢占次数
```

## Acceptance

通用标准见 §3.5（Gate A 能跑 + Gate B 超参统一）。本阶段额外检查：

被抢占后能从 `resume.pt` 继续训练直到完成。随机轨迹与连续训练一致属于尽力目标，
不作为验收要求。

---

# 11. Phase 6 — Extract ObservationEncoder Boundary

这是未来接 VariDP 的关键步骤。

当前 observation encoding 不应该直接返回 flatten 后的：

```text
(B, To × D)
```

新的统一接口：

```python
obs_features = observation_encoder(rgb, proprio)
```

输出固定为：

```text
(B, To, Dobs)
```

例如：

```text
B = batch
To = observation horizon
Dobs = RGB feature + proprio feature
```

## 非常重要

ObservationEncoder 只负责：

```text
RGB
+
proprio
        ↓
shared feature sequence
```

它不应该知道：

```text
UNet
Transformer
MLP
```

## Acceptance

通用标准见 §3.5（Gate A 能跑 + Gate B 超参统一）。本阶段额外检查：

ObservationEncoder 输出 shape 为 `(B, To, Dobs)`。

---

# 12. Phase 7 — Introduce Backbone Interface

定义统一 Noise Predictor abstraction。

概念上：

```python
class NoisePredictor:
    def forward(
        noisy_actions,
        timestep,
        obs_features
    ):
        ...
```

其中：

```text
noisy_actions:
    (B, Tp, action_dim)

obs_features:
    (B, To, obs_dim)
```

不同 backbone 自己决定如何消费 observation features。

## UNet Adapter

```text
(B, To, D)
    ↓ flatten
(B, To×D)
    ↓
Conditional UNet
```

## MLP Adapter

可以类似：

```text
(B, To, D)
    ↓ flatten
global condition
```

## Transformer Adapter

保留：

```text
(B, To, D)
```

作为 condition tokens。

因此：

```text
              Shared Observation Encoder
                        │
                  (B, To, D)
                        │
          ┌─────────────┼─────────────┐
          │             │             │
       flatten       sequence      flatten
          │             │             │
        UNet        Transformer       MLP
```

## Acceptance

通用标准见 §3.5（Gate A 能跑 + Gate B 超参统一）。本阶段额外检查：

`backbone = "unet"` 通过 Gate A。

切换 backbone 不允许创建不同：

```text
dataset
visual encoder
trainer
scheduler
EMA
evaluation pipeline
```

---

# 13. Phase 8 — Preserve Current UNet As Reference

不要在抽象 backbone interface 时顺便替换 UNet。

首先把当前 canonical UNet 包装成：

```text
UNetBackbone
```

`DiffusionPolicy(backbone="unet")` 必须继续使用当前 UNet 结构和 baseline 超参：

```text
unet_dims          = [256, 512, 1024]
diffusion embed    = 256
kernel size        = 5
groups             = 8
```

不要求新旧实现数值等价：RGB pipeline 本身是从 state training 迁移来的新实现，
没有需要逐位保持的已验证结果。

## Acceptance

```text
backbone = "unet"
    ↓
train → save checkpoint → load checkpoint → evaluate
```

能跑通，且 resolved config 中的 UNet 与训练超参仍全部来自 `baseline.toml`。

---

# 14. Phase 9 — Start VariDP Migration

到这里才开始真正修改 / 拆解 VariDP。

VariDP 的定位：

```text
donor implementation
```

而不是第二套 pipeline。

## 应迁移

主要迁移：

```text
UNet abstraction
Transformer backbone
MLP backbone
```

特别关注：

```text
VariDP/dp/backbones.py
```

## 不迁移

禁止直接搬：

```text
VariDP trainer
VariDP dataset
VariDP scheduler
VariDP EMA
VariDP evaluator
VariDP diffusion training loop
```

这些都应该被 canonical pipeline 替代。

---

# 15. Phase 10 — Adapt Transformer and MLP

分别创建：

```text
TransformerBackbone
MLPBackbone
```

要求它们实现与 UNet 相同的 NoisePredictor contract。

## Backbone Experiment Hard Constraints

以下必须完全相同：

```text
dataset
training subset
RGB preprocessing
augmentation
visual encoder
proprio handling
optimizer
LR
batch size
training steps
EMA
diffusion scheduler
diffusion steps
horizons
evaluation seeds
```

唯一实验变量：

```text
policy.backbone
```

允许 backbone 自身拥有结构参数：

```text
Transformer:
    layers
    heads
    hidden dim

MLP:
    layers
    hidden dim

UNet:
    channel dims
```

但这些属于对应 architecture definition。

## Acceptance

通用标准见 §3.5（Gate A 能跑 + Gate B 超参统一）。本阶段额外检查：

- UNet / Transformer / MLP 各自通过 Backbone Smoke Test（§20）并走通 Gate A。
- 三个 backbone 的 resolved config 满足 Gate B，只有 `policy.backbone` 及其结构参数不同。
- 不要求 Transformer / MLP 达到任何性能指标。

---

# 16. Phase 11 — Canonical Experiment Definitions

建立：

```text
configs/experiments/
```

## data_size.toml

```toml
[experiment]
name = "data_size"
variable = "data.num_demos"
values = [25, 50, 100, 200]
```

## backbone.toml

```toml
[experiment]
name = "backbone"
variable = "policy.backbone"
values = ["unet", "transformer", "mlp"]
```

以后：

## failure_case.toml

可能：

```toml
[experiment]
name = "failure_case"
```

具体变量根据最终实验设计：

```text
failure ratio
failure loss weight
auxiliary success head
```

但仍然不能产生第三套 trainer。

---

# 17. Phase 12 — Unified Entry Point

最终建立：

```text
scripts/run_experiment.py
```

统一入口。

## Data-size

```bash
python scripts/run_experiment.py \
    --task pickcube \
    --experiment data_size \
    --value 50 \
    --seed 0
```

## Backbone

```bash
python scripts/run_experiment.py \
    --task pickcube \
    --experiment backbone \
    --value transformer \
    --seed 0
```

内部统一：

```python
cfg = resolve_config(
    baseline,
    task,
    experiment,
    value,
    seed,
)

dataset = build_dataset(cfg)

policy = build_policy(cfg)

trainer = Trainer(
    cfg,
    dataset,
    policy,
)

trainer.train()
```

绝对不能出现：

```python
if experiment == "data_size":
    run_dp_manip()

elif experiment == "backbone":
    run_varidp()
```

这是假统一入口。

---

# 18. Phase 13 — Experiment Invariant Checker

这是 Gate B 的自动化版本。实现一个最简单的 resolved config 比较即可。

对于一次 experiment matrix，比较所有 resolved configs。

例如 backbone experiment：

```text
UNet config
Transformer config
MLP config
```

允许差异：

```text
policy.backbone
policy.backbone.*
run.seed
runtime.*
```

其余差异直接报错。

例如：

```text
ERROR: Unexpected experiment config drift

train.batch_size:
    unet        = 64
    transformer = 128
```

这样可以防止实验长期维护后悄悄失去 controlled comparison。

---

# 19. Run Metadata

每次实验保存：

```text
resolved config
experiment name
experimental variable
experimental value
seed

git commit
git branch
dataset path
dataset fingerprint
selected demo seeds

hostname
GPU
start time
training duration
```

推荐额外生成：

```text
control_hash
```

它由所有非实验变量生成。

同一 experiment matrix：

```text
control_hash
```

必须相同。

---

# 20. Testing Strategy

必须的 tests（对应 §3.5 两道 gate）：

```text
Config Test
Experiment Drift Test
Checkpoint Test
Backbone Smoke Tests
```

其余（Dataset Nesting、Observation Encoder shape、UNet Smoke）有则更好，不阻塞。

## Config Test

验证：

```text
baseline + task + override
```

merge 正确。

---

## Experiment Drift Test

验证：

```text
backbone experiment
```

只改变 backbone。

---

## Dataset Nesting Test

验证：

```text
25 ⊂ 50 ⊂ 100 ⊂ 200
```

---

## Observation Encoder Test

输入：

```text
RGB + proprio
```

输出：

```text
(B, To, D)
```

shape 正确。

---

## UNet Smoke Test

验证：

```text
UNetBackbone
```

可以完成 train / save / load / evaluate，且使用 baseline 中的 UNet 超参。

---

## Checkpoint Test

验证：

```text
train
save
load
resume
evaluate
```

---

## Backbone Smoke Tests

每个：

```text
UNet
Transformer
MLP
```

执行：

```text
one forward
one backward
one optimizer step
one sampling pass
```

---

# 21. Legacy Strategy

旧代码暂时不要删。

尤其 state-based implementation 可以作为：

```text
debug / regression reference
```

但不得继续参与正式实验。

建议阶段性：

```text
legacy/state/
legacy/varidp/
```

最终等：

```text
RGB UNet baseline
Transformer
MLP
```

全部验证后再删除重复实现。

或者至少保留 Git tag：

```text
pre-unified-pipeline
```

避免为了保留历史代码而污染新架构。

---

# 22. Commit Strategy

不要一个巨大 commit。

建议拆成：

```text
1. refactor(config): add canonical baseline layering

2. refactor(experiment):
   remove data-size assumptions from core

3. refactor(data):
   rename state conditioning to proprio

4. fix(data):
   enforce deterministic nested subsets

5. fix(checkpoint):
   persist RNG state

6. refactor(model):
   extract observation encoder

7. refactor(model):
   introduce noise predictor interface

8. refactor(model):
   wrap existing UNet as backbone

9. feat(backbone):
   migrate transformer from VariDP

10. feat(backbone):
    migrate MLP from VariDP

11. feat(experiment):
    add backbone experiment config

12. feat(cli):
    add unified experiment runner

13. test:
    add config and experiment invariants

14. chore:
    mark legacy implementations
```

每一步都应该能单独 review。

---

# 23. Worker / Supervisor 执行方式

不建议一次给 worker：

```text
“统一整个仓库”
```

应该拆成连续 task。

每个 task 包含：

```text
Goal
Scope
Non-goals
Files expected to change
Behavior invariants
Acceptance tests
```

Supervisor 每阶段只检查 §3.5 的两道 gate：

```text
能跑吗？（Gate A）
超参统一吗？（Gate B：baseline 未被改动、没有第二套 pipeline、没有硬编码超参）
```

其他问题记到 M17，不阻塞当前 Phase。

如果 worker 已完成 95%，最后只是少量：

```text
rename
small validation
minor integration
```

Supervisor 可以直接修掉，不需要为了剩余 5% 再发完整 task。

---

# 24. 推荐实际执行顺序

当前从：

```text
refactor/dp-manip-rgb-cluster
```

继续。

执行顺序：

```text
M1
Freeze RGB baseline

M2
Baseline config + task config

M3
Remove data-size knowledge from core

M4
state → proprio terminology

M5
Nested subset invariant

M6
RNG checkpoint improvement

M7
Extract ObservationEncoder

M8
Extract NoisePredictor interface

M9
Wrap existing UNet

M10
Smoke validation (runs end to end; hyperparameters unchanged)

M11
Migrate Transformer from VariDP

M12
Migrate MLP from VariDP

M13
Create backbone experiment

M14
Create unified run_experiment entrypoint

M15
Add invariant checker

M16
Deprecate duplicated VariDP pipeline

M17
Final cleanup (directory, naming, docs, deferred fixes)
```

---

# 25. 本轮不要做的事情

当前阶段明确禁止：

```text
不要重写整个 dp-manip

不要把整个 VariDP merge 进 dp-manip

不要重新实现 trainer

不要重新设计 diffusion scheduler

不要换视觉 encoder

不要同时修改 baseline hyperparameters

不要马上删除 state implementation

不要马上进行大规模目录 rename

不要在 backbone migration 前实现统一 CLI

不要让每个 experiment 自己维护一份 config
```

---

# 26. Definition of Done

整个 refactor 完成时，需要满足：

### Architecture

```text
只有一套正式 RGB Dataset
只有一个 Observation Encoder
只有一个 Diffusion Policy wrapper
只有一个 Trainer
只有一个 Evaluator
只有一个 checkpoint format
```

### Backbone

支持：

```text
UNet
Transformer
MLP
```

通过同一个 interface。

### Config

存在唯一：

```text
canonical baseline
```

任务只提供 task-specific differences。

### Experiments

可以运行：

```text
data_size
backbone
failure_case（以后）
```

而不复制 pipeline。

### CLI

例如：

```bash
python scripts/run_experiment.py \
    --task pickcube \
    --experiment backbone \
    --value transformer \
    --seed 0
```

### Reproducibility

每次 run 尽量保存（不阻塞完成）：

```text
resolved config
git revision
dataset identity
selected demos
seed
```

### Scientific Validity

同一 experiment 中：

> 除实验变量、replicate seed 和 runtime metadata 外，其余 experimental conditions 完全一致。

---

# 27. 最终架构判断

这次 refactor 的正确方向不是：

```text
dp-manip
+
VariDP
→
一个更大的仓库
```

而是：

```text
dp-manip RGB pipeline
          ↓
extract canonical core
          ↑
VariDP backbone assets
```

最终：

```text
                    Shared RGB DP
                         │
           ┌─────────────┼─────────────┐
           │             │             │
      Data Size       Backbone     Failure Case
           │             │             │
     N demos       U / T / MLP     failure vars
```

`dp-manip` 和 `VariDP` 最终都不再代表 implementation。

它们只代表项目历史上两条实验线的来源。

新的代码结构中：

> implementation belongs to the shared pipeline; experiments belong to configuration.
