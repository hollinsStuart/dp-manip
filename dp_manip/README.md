# dp_manip package

当前实现是共享的 RGB Diffusion Policy：

- `data.py` 直接读取 `maniskill-demogen` 的 `obs_rgb/{rgb,state}`，图像懒加载；
- `vision.py` 用 GroupNorm ResNet-18 编码每个相机；
- `observation_encoder.py` 把 RGB + 非特权 proprioception 编码成共享的 `(B, To, Dobs)` 序列；
- `backbones/` 提供统一的 `NoisePredictor` 接口与 UNet / Transformer / MLP 三种主干；
- `policy.py` 负责动作归一化、DDPM，并把观测序列原样交给 noise predictor；
- `trainer.py` 是唯一训练 pipeline；`training.py` 提供 EMA、RNG state 与原子 checkpoint；
- `invariants.py` / `metadata.py` 负责 Gate B 语义与 run metadata；
- `envs.py` / `evaluate.py` 负责固定 seed 的 RGB 闭环评估。

`conditional_unet1d.py` 源自 ManiSkill 官方 DP baseline（Apache-2.0，见
`LICENSE-ManiSkill`）。其余训练结构已针对 RGB 数据与 Slurm 集群重写。
