# configs

每个 TOML 是一次 DP 实验的完整定义，训练（`scripts/train_dp.py`）、验证、测试（`scripts/eval_dp.py`）和回放检查（`scripts/replay_check.py`）共用同一份；训练时解析后的配置写入 `results/<exp>/config.json` 和每个 checkpoint，评估直接读 checkpoint 里的配置。

| 段 | 内容 |
| --- | --- |
| `[task]` | 环境 id、控制模式、观测模式、仿真后端、评估回合长度 `max_episode_steps` |
| `[data]` | state 重放后的 H5 路径（相对仓库根）；`num_demos` 取前 N 条 |
| `[policy]` | 观测历史 `obs_horizon`、执行步数 `act_horizon`、预测长度 `pred_horizon`、UNet 规模、扩散步数 |
| `[train]` | 随机种子、迭代数、batch、学习率、日志 / 验证 / 保存频率 |
| `[eval]` | 验证种子段（训练中选 `best.pt`）与测试种子段（只用于报告），两段互不重叠，也不能与示范种子重叠 |

命令行可用 `--set section.key=value` 临时覆盖，例如 `--set data.num_demos=5 --set train.total_iters=300`。

当前配置：

- `pickcube_state_jointpos.toml`：PickCube，`pd_joint_pos`，10 条示范，用于跑通链路。
- `pickcube_state_jointpos_100.toml`：PickCube，`pd_joint_pos`，100 条示范（ubuntu 单进程采集，种子从 0 起）；数据效率实验用 `--set data.num_demos=N` 取前 N 条。
