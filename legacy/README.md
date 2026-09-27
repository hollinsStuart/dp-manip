# Legacy：已冻结的 state-based 工作流

这里保存统一 RGB pipeline 之前的本地 state-based 工作流，**只作调试和回归参考**，
不参与任何正式实验，也不得被正式代码 import 或调用。

正式入口只有：

```text
dp_manip/ + configs/tasks/ + configs/experiments/
scripts/run_experiment.py   # 统一实验入口
scripts/train_dp.py         # Slurm sweep 用的薄 CLI
scripts/sweep.py + slurm/   # 数组作业
```

## 内容

| 路径 | 说明 |
| --- | --- |
| `state/run_cpu.py` | 旧的本地运动规划示范生成脚本（ManiSkill panda 示例的本地副本，增加 `render_backend="cpu"`）。正式数据现由 `maniskill-demogen` 生成 |
| `state/docs/` | 9.22–9.24 的 state-based 试验记录与旧 WSL/Ubuntu 操作说明。文首已加归档说明；跨文档链接已按新位置调整，命令里的 `patches/`、`manifests/`、`run_cpu.py` 等路径仍是归档前的旧布局 |
| `state/patches/` | 旧 Ubuntu 环境用的 `mani_skill_mplib_0_2_1.patch` |
| `state/manifests/` | 旧 WSL/Ubuntu 数据的 sha256 清单 |
| `state/environment/`、`state/mplib-probe-overrides.txt` | 旧环境冻结记录与探针覆盖项 |

`configs/*_rgb.toml` 的兼容跳转不在 legacy 里：它们只有几行，仍由
`dp_manip.config` 支持并有测试覆盖，方便旧命令定位到 canonical task config。

## 规则

- 正式代码（`dp_manip/`、`scripts/`、`slurm/`、`tests/`）不得 import 或执行这里的文件；
  `tests/test_legacy_boundary.py` 会静态检查这一点，并确认这些资产只存在于 `legacy/` 下。
- 不要把新功能加到这里。新实验只写 config，不写新 pipeline。
- `VariDP/` 是 backbone 实现的 donor，已冻结，同样不参与正式实验；它保留在课程集成仓库
  `hryang1130/7606C` 里（[`VariDP/LEGACY.md`](https://github.com/hryang1130/7606C/blob/main/VariDP/LEGACY.md)），
  不随本 standalone 仓库分发。UNet / Transformer / MLP 的 canonical 实现已经在
  `dp_manip/backbones/` 里。
- 不要在这里修历史记录；需要新结论时写新的 run 记录。

## 删除条件

等集群上 RGB UNet baseline、Transformer、MLP 三条 arm 全部通过 Gate A
（train → save checkpoint → load checkpoint → 闭环 evaluate）后，可以删除本目录以及
VariDP 中与 canonical 实现重复的部分。删除前确认 git tag `pre-unified-pipeline`
（指向 Phase 0 冻结的 RGB baseline，commit `834be80`）仍然存在——它是重构前状态的恢复点。
