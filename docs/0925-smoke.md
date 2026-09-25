# 运行记录：9.25 — RGB 数据链路冒烟验证（六任务 × VariDP）

**日期**：2026-09-25 ｜ **代码**：`e463d00` → 本文件所在提交 ｜ **机器**：ubuntu（生成）、Mac（导出）、wsl（训练与评估）｜ 计划：[final-plan.md](./final-plan.md)

## 一、结论

- **链路通了**：6 个任务都完成了「专家生成 → 转换到计划的控制模式（rgb 与 state 各一次）→ 修正第 0 帧 → 导出 → VariDP 原版 `train/train.py` 训练 → `train/eval.py` 评估」。VariDP 的代码一行未改，数据用 `--h5` 直接读入。
- **数据与评估环境对得上**：导出文件中每条示范的第 0 帧，与 wsl 上一个普通 `gym.make` 环境（加 `FlattenRGBDObservationWrapper`）按同一种子 reset 出来的观测完全一致（state 差 ≤ 6e-8），图像平均每个像素值只差 0.2–0.4（ubuntu 用 NVIDIA 渲染、wsl 用 lavapipe，属于正常的渲染器差别）。
- **只训练 1000 步**，PullCube 已成功 5/8、PushCube 1/8，说明观测与动作是对齐的。其余任务为 0，这在预期之内，本轮不衡量性能。
- 发现 ManiSkill 3.0.1 的 **4 个会污染数据的问题**，已在生成与导出脚本里规避或修正（§四）；另有 **4 项需要修订 final-plan**（§五）。
- 7606-train-template 这一轮没有做（用户决定先只做 VariDP）。

## 二、生成数据的代码（全部在仓库里）

| 步骤 | 机器 | 脚本 | 作用 |
| --- | --- | --- | --- |
| 专家生成 | ubuntu | `run_cpu.py`（新增 `--start-seed`） | 运动规划，`pd_joint_pos`，obs none；验证示范从种子 4000 开始 |
| 转换 | ubuntu | `mani_skill.trajectory.replay_trajectory` | 从原始轨迹 `--use-first-env-state -c <mode>`，`-o rgb --shader minimal` 与 `-o state` 各转一次 |
| 第 0 帧修正 | ubuntu | `scripts/first_frame_obs.py` | 在未推进过物理的新环境里按种子 reset，核对状态与原始轨迹第 0 帧一致，写 `*.first_obs.h5` 旁路文件 |
| 以上三步串联 | ubuntu | `scripts/smoke/gen_data.sh` | 每任务训练 N 条（种子 0 起）、验证 M 条（种子 4000 起） |
| 质量统计 | 任意 | `scripts/smoke/replay_stats.py` | 专家成功率、转换成功率、被丢弃的种子、中途重置、长度、相机数（final-plan §2.3） |
| 导出 | Mac | `scripts/export_demos.py`（`scripts/smoke/export_data.sh` 批量） | 按种子配对 rgb/state，校验一致，丢弃中途重置的示范，替换第 0 帧，按种子重排为 `traj_0..`，回读校验 |
| 预览 | Mac | `scripts/smoke/preview_rgb.py` | 前 5 条示范的首帧与末帧拼图 |
| 对齐检查 | wsl | `scripts/smoke/check_rgb_obs.py` | 导出文件首帧与评估环境 reset 观测对比 |
| VariDP 环境 | wsl | `scripts/smoke/setup_teammates.sh`（`REPOS=varidp`） | 克隆并固定 `28afb87`，按其文档 `uv sync` |
| VariDP 训练评估 | wsl | `scripts/smoke/run_varidp.sh`、`scripts/smoke/varidp_eval.py` | 见 §三.4 |

导出文件的命名、目录和 JSON 与官方示范一致（§三.5），每个 split 一棵目录树：

```
<根目录>/{train,val}/<Env>/motionplanning/trajectory.state.<mode>.physx_cpu.h5
<根目录>/{train,val}/<Env>/motionplanning/trajectory.state.<mode>.physx_cpu.json              只有官方字段
<根目录>/{train,val}/<Env>/motionplanning/trajectory.state.<mode>.physx_cpu.export_info.json  本次导出的信息
<根目录>/train/<Env>/motionplanning/sample.png                                                预览
```

HDF5 内容（`scripts/export_demos.py` 文件头有完整说明）：

```
traj_i/obs            (T+1, D)        float32  obs_mode=state 的扁平向量（含物体位姿，仅供 state 训练）
traj_i/obs_rgb/rgb    (T+1, 128,128,3C) uint8  C 路相机按通道拼接
traj_i/obs_rgb/state  (T+1, P)        float32  obs_mode=rgb 的 agent + extra（非特权）
traj_i/actions, success, terminated, truncated, env_states
```

`obs_rgb/*` 与评估时 `FlattenRGBDObservationWrapper(env, rgb=True, depth=False)` 的 `obs["rgb"]`、`obs["state"]` 逐键对应。

## 三、执行记录

### 1. 阶段 0（Mac）

`e463d00`：`export_demos.py` 取代 `export_stanford_dp.py` 与 `sdp_maniskill/`；`run_cpu.py` 加 `--start-seed`；`scripts/smoke/` 初版。先在 Mac 上用伪造的 fixture 测过导出、预览和 template 插件生成。

### 2. 阶段 1（ubuntu，第二次运行 `OUT=demos-smoke0925b`，约 6.5 分钟）

第一次运行（`demos-smoke0925/`）的 state 用的是 `--use-env-states` 重放 rgb 文件，数据有问题（§四.1），`60d0aca` 改为从原始轨迹独立转换后重跑。两次运行的 24 个原始轨迹和 rgb 文件中 22 个逐字节相同（差别只在 PlugCharger 两条中途重置示范的随机占位动作），生成是确定性的。

`scripts/smoke/replay_stats.py demos-smoke0925b`：

| 任务 | 相机 | 训练集：专家 / 转换 / 可用 | 验证集：专家 / 转换 / 可用 | 长度（中位） |
| --- | --- | --- | --- | --- |
| PickCube | 1 | 10/10 · 10/10 · 10 | 5/5 · 5/5 · 5 | 49–88（74） |
| StackCube | 2 | 10/10 · 10/10 · 10 | 5/5 · 5/5 · 5 | 94–120（109） |
| PushCube | 1 | 10/11 · 10/10 · 10 | 5/5 · 5/5 · 5 | 62–77（68） |
| PullCube | 1 | 10/10 · 10/10 · 10 | 5/5 · 5/5 · 5 | 63–79（72） |
| PegInsertionSide | 2 | 10/13 · 8/10 · 8 | 5/5 · 3/5 · 3 | 123–190（153） |
| PlugCharger | 2 | 10/13 · 7/10 · 6（种子 3 中途重置） | 5/7 · 4/5 · 3（种子 4006 中途重置） | 155–234（167，不含两条重置示范的 457、412） |

「专家」为成功条数 / 尝试的种子数；「转换」为转到 `pd_ee_delta_pos(e)` 成功的条数；「可用」再去掉中途重置的示范。24 个旁路文件中，reset 出来的状态与原始轨迹第 0 帧全部完全一致。

### 3. 阶段 2（Mac 导出，约 1 分钟）

`SRC=demos-smoke0925b DST=data/smoke0925b scripts/smoke/export_data.sh`，12 个文件全部通过：rgb 与 state 两次转换的 `env_states` 差为 0；PlugCharger 种子 3、4006 被丢弃；第 0 帧修正实际改动的只有 PickCube 13 条示范的 `is_grasped`（1 → 0），其余是 6e-8 级浮点误差和腕部相机少数像素 1 个灰度级的变化。与修正前的导出（`data/smoke0925/`）逐项比对：只有第 0 行不同。

| 任务 | 训练 / 验证 | 动作 | `obs` | `obs_rgb/state` | 图像 |
| --- | --- | --- | --- | --- | --- |
| PickCube | 10 / 5 | 4 | 42 | 29 | 128×128×3 |
| StackCube | 10 / 5 | 4 | 48 | 25 | 128×128×6 |
| PushCube | 10 / 5 | 4 | 35 | 25 | 128×128×3 |
| PullCube | 10 / 5 | 4 | 35 | 28 | 128×128×3 |
| PegInsertionSide | 8 / 3 | 7 | 43 | 25 | 128×128×6 |
| PlugCharger | 6 / 3 | 7 | 46 | 25 | 128×128×6 |

预览图（现为 `data/smoke0925c/train/<Env>/motionplanning/sample.png`）内容正常。PickCube 画面中看不到目标点（ManiSkill 对相机隐藏了 `goal_site`），所以 `obs_rgb/state` 中的 `goal_pos` 是必需的。

### 4. 阶段 3–4（wsl）

**环境**：`REPOS=varidp scripts/smoke/setup_teammates.sh`，约 3 分钟。VariDP `28afb87`，Python 3.12.3，torch 2.14.0+cu126（RTX 4090 可用），numpy 2.5.3，h5py 3.16.0，gymnasium 1.3.0，mani-skill 3.0.1，sapien 3.0.3；`.venv` 7.2 GB。VariDP 仓库没有提交 `uv.lock`，本次现场解析了 105 个包。我们自己的 `.venv` 与锁文件未被改动（安装前后时间戳和校验和核对）。

**训练与评估**：`VARIDP_BACKEND=cpu scripts/smoke/run_varidp.sh`，19:37:51–19:45:14（CST），16 步全部 OK。数据为修正前的 `data/smoke0925/`，与 `data/smoke0925b/` 只差第 0 帧，不影响「流程能跑通」这一结论。

| 任务 | 网络（参数量） | 训练 1000 步，batch 256 | 评估：CPU 仿真，8 回合（种子 2000–2007） |
| --- | --- | --- | --- |
| PickCube | mlp 0.353M / unet 66.4M / transformer 8.97M | 8 s / 50 s / 28 s | 0/8 · 0/8 · 0/8（100 步） |
| StackCube | unet 66.6M | 41 s | 0/8（200 步） |
| PushCube | unet 66.2M | 48 s | 1/8 |
| PullCube | unet 66.2M | 47 s | **5/8** |
| PegInsertionSide | unet 66.5M | 39 s | 0/8（300 步） |
| PlugCharger | unet 66.5M | 45 s | 0/8（200 步） |

**对齐检查**：`check_rgb_obs.py --render-backend cpu`。修正后的数据：12 个文件各前 5 条（共 54 条）加 PickCube 全部 15 条，第 0 帧的 `obs` 和 `obs_rgb/state` 与 reset 观测全部一致（差 ≤ 6e-8）。修正前的数据中 PickCube 有 4 条差恰为 1.0，就是 `is_grasped`。

### 5. 改为官方的文件名和 JSON，并与官方数据对比

队友下载并转换过官方 PickCube 示范（`.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5`，1000 条，已验证能跑通）。与之对比：

- **HDF5 结构完全相同**：`traj_N/{obs (T+1,42), actions (T,4), success, terminated, truncated, env_states/...}`，键名、dtype、形状一致；我们只多一个 `obs_rgb/` 组，队友代码会忽略它。
- **同一种子几乎是同一份数据**：种子 0–9 两边长度逐条相同，动作差 ≤ 6e-6，`obs` 差 ≤ 2.4e-4（仅第 0 帧的 `is_grasped` 差 1）。官方示范用的也是同一个运动规划器、同一套种子。
- **官方数据带有 §四.3、§四.4 的问题**：1000 条里 996 条第 0 帧 `is_grasped` 为 1（只有 4 条对，因为队友用 4 个进程转换，每个进程的第一条不受影响）；`env_states[0]` 的偏差与我们修正的量逐条相同。

随后把导出改成官方的命名和 JSON（导出到 `data/smoke0925c/`）：HDF5 与 `data/smoke0925b/` 逐位相同，只改文件名、目录和 JSON。JSON 的顶层字段（`episodes`、`env_info`、`commit_info`）、每条示范的 6 个字段及其顺序与官方相同；`env_info` 取 state 转换的，`obs_mode` 为 `state`，与 `traj_N/obs` 一致。其余信息移到 `.export_info.json`。仍然不同的只有取值：`env_kwargs` 是 ManiSkill 3.0.1 实际写出的（`sensor_configs`、`sim_backend: physx_cpu`、`reset_kwargs.options: null`），官方是更早版本写的（`shader_dir`、`sim_backend: cpu`、`options: {}`），含义相同，没有改；官方 JSON 的 CRLF 换行来自队友的 Windows，也没有模仿。

导出图像改用 h5py 的自动分块（ManiSkill RecordEpisode 的写法，gzip 5 不变）：原先每帧一块只压缩到 1.4–1.8 倍，同一批 12 个文件从 419 MB 降到 196 MB（单相机任务降到 14–17%，双相机 52–61%），内容逐位相同；按此估算完整数据集（每任务 400 + 50 条）的导出约 6.5 GB。`data/smoke0925c/` 是改之前导出的，内容相同、体积大一倍。

在 wsl 上用 VariDP 自己的工具验证新布局：`scripts/check_datasets.py --demo-dir data/smoke0925c/train` 扫到 6 个数据集，`obs_mode` 全为 `state`，维度、条数、长度、动作范围正确；`dp_lib.find_dataset` 自动找到 4 个 4 维任务（7 维任务要 `--h5`，因为它只找 `pd_ee_delta_pos` 后缀，官方数据同样如此）。`make_template_tasks.py` 为 PlugCharger 生成的配置，除数据路径外与队友基于官方数据写的 `task_06` 完全相同（`obs_dim 46`、动作 7、`pd_ee_delta_pose`、200 步）。`check_rgb_obs.py` 在新文件上 24 个首帧全部一致。

## 四、ManiSkill 3.0.1 的数据问题（已处理）

1. **`--use-env-states` 录到的不是被设置的状态。** 重放循环先 `env.step(a)` 并由 RecordEpisode 录下结果，然后才 `set_state_dict(s[t+1])`；录下的每一帧都是「从正确状态模拟一步」的预测，delta 控制模式下位置偏到厘米级（PlugCharger 关节角偏 0.5 rad）。**处理**：state 也从原始轨迹独立转换一次；两次转换的 `env_states` 逐位相同（CPU 物理是确定性的），导出时逐条校验。注：9.23 PickCube 的 pilot state 数据也是用 `--use-env-states` 做的。
2. **转换会把失败的尝试拼进同一条示范。** PlugCharger 种子 3 的原始轨迹 146 步，转换后 457 步：两次失败尝试（151、159 步）和最后成功的一次首尾相接，每次重置那一步是一行随机占位动作，关节角跳回初始位姿（单步 0.48 rad）。**处理**：导出时单步关节变化超过 0.2 rad（Panda 物理上每步最多约 0.13）的示范按转换失败丢弃，写进 JSON 的 `rejected`。
3. **第 0 帧的接触类观测是上一条示范的残留。** 同一文件的多条示范在一个环境里依次重放，每条开头设置状态后不推进物理就取观测，`is_grasped` 读到上一条结尾的接触：PickCube 除每个文件第一条外，第 0 帧都记成 1。**处理**：`first_frame_obs.py` 生成正确的第 0 帧，导出时替换并记入 `first_frame_fix`。
4. **转换文件的 `env_states[0]` 错了一位。** 重放代码先 `ori_env_states = ori_env_states[1:]` 再用 `[0]` 替换第一条记录，记下的是原始轨迹 t=1 的状态（方块位置差几毫米，速度非零）；第 0 帧的观测和图像则是在真正的初始状态下取的。**处理**：同 3，导出时一并替换为 reset 状态。

## 五、需要修订 final-plan 的地方（待定）

1. **7 维任务转换成功率低于 §2.3 的 90% 关卡**：PegInsertionSide 11/15（73%），PlugCharger 去掉中途重置后 9/15（60%）。按事先写定的规则应退回 `pd_joint_pos`。样本只有 15 条，正式采集前建议先用约 50 条再测一次。
2. **评估回合长度按 §1 的规则都要变**：规则是 max(官方值, 示范平均长度 × 2)，再向上取整到 50。用本轮示范（训练集，条数少，仅供参考）算：

   | 任务 | 平均长度 | 平均 × 2 | 官方 / 计划现值 | 按规则 |
   | --- | --- | --- | --- | --- |
   | PickCube | 72.6 | 145 | 100 | 150 |
   | StackCube | 108.1 | 216 | 200 | 250 |
   | PushCube | 68.6 | 137 | 100 | 150 |
   | PullCube | 72.6 | 145 | 100（按 PushCube） | 150 |
   | PegInsertionSide | 154.8 | 310 | 300 | 350 |
   | PlugCharger | 176.8 | 354 | 200（暂定） | 400 |

   PlugCharger 最明显：最长的有效示范 234 步，已超过现在的 200。本轮评估用的仍是计划现值。正式长度等 400 条示范采完后按同一规则确定。
3. **三个任务有两路相机**：StackCube、PegInsertionSide、PlugCharger 默认使用 `panda_wristcam`，图像为 6 通道；§3 的观测描述要写明，给队员的数据说明里也要写。
4. **评估后端**：VariDP 的 `eval.py` 默认在有 CUDA 时用 `physx_cuda`，而本计划统一用 `physx_cpu`，同一种子在两种后端下的初始状态并不相同。全组要统一口径。

## 六、偏差

1. **阶段 2 第一次运行失败**：state 文件有问题（§四.1），12 个导出全部被校验拦下。修正后重跑阶段 1、2。
2. **阶段 2 重跑时被残留的空目录挡住**：失败的导出留下了 `data/smoke0925/<task>/` 空目录。确认为空后用 `rmdir` 删除；`fb2529f` 起导出失败时会删掉本次运行自己建的目录。期间我一度误把旧日志当成了新的失败。
3. **wsl 在 GPU 仿真评估启动 1 秒后被 Windows 一侧关机**（19:13:42，logind「The system will power off now!」，此前没有 OOM 或 panic）。之前训练时内核报过一次 `dxgkrnl` 的 WARNING，9.23 也出现过，当时没有影响。评估改用 CPU 仿真（VariDP 文档中写明的备选），在 WSL 上还需要：只开 1 个环境；通过 `scripts/smoke/varidp_eval.py` 给 `gym.make` 补 `render_backend="none"`（其默认渲染设备需要 NVIDIA Vulkan，WSL 没有）。VariDP 的代码未改。GPU 仿真与 WSL 的问题尚未单独复现。
4. `check_rgb_obs.py` 的 `--render-backend` 和 `first_frame_obs.py` 在提交前通过标准输入或临时副本在远端运行过，内容与本次提交一致。

## 七、数据位置与待清理

| 位置 | 内容 | 状态 |
| --- | --- | --- |
| ubuntu、Mac `demos-smoke0925b/` | 本轮原始轨迹、两次转换、旁路文件（185 MB） | 有效 |
| Mac、wsl `data/smoke0925c/` | 修正后的导出，官方命名与 JSON | **有效，给队员用这份** |
| Mac、wsl `data/smoke0925b/` | 与 c 的 HDF5 相同，旧命名与 JSON | 待删除（需确认） |
| ubuntu、Mac `demos-smoke0925/` | 第一次运行，state 文件有问题 | 待删除（需确认） |
| Mac、wsl `data/smoke0925/` | 修正第 0 帧前的导出，阶段 4 训练用的是它 | 待删除（需确认） |
| wsl `~/teammates/VariDP/train/runs/smoke_*` | 本轮 checkpoint 与评估 JSON | 验证用，可删 |

这些都是一次性验证数据，未写入 `manifests/`。

## 八、复现

```bash
# ubuntu
OUT=demos-smoke0925b scripts/smoke/gen_data.sh
# Mac
rsync -a ubuntu:Coding/dp-manip/demos-smoke0925b/ demos-smoke0925b/
.venv/bin/python scripts/smoke/replay_stats.py demos-smoke0925b
SRC=demos-smoke0925b DST=data/smoke0925c scripts/smoke/export_data.sh
rsync -a data/smoke0925c/ wsl:projects/dp-manip/data/smoke0925c/
# wsl
REPOS=varidp scripts/smoke/setup_teammates.sh
DATA=$PWD/data/smoke0925c VARIDP_BACKEND=cpu scripts/smoke/run_varidp.sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  .venv/bin/python scripts/smoke/check_rgb_obs.py data/smoke0925c/*/*/motionplanning/*.h5 --num 5 --render-backend cpu
```
