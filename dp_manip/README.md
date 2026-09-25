# dp_manip

Diffusion Policy training and evaluation for ManiSkill 3.0.1 tasks, from state or RGB observations
(`task.obs_mode = "state"` / `"rgb"`).

## Provenance

Based on ManiSkill's official DP baseline, `examples/baselines/diffusion_policy`
at [haosulab/ManiSkill@62ff3a5](https://github.com/haosulab/ManiSkill/tree/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/examples/baselines/diffusion_policy)
(Apache-2.0, license in [LICENSE-ManiSkill](./LICENSE-ManiSkill)). That baseline in turn follows
Diffusion Policy by Chi et al. ([paper](https://arxiv.org/abs/2303.04137),
[code](https://github.com/real-stanford/diffusion_policy), MIT).

| File | Relation to the baseline |
| --- | --- |
| `conditional_unet1d.py` | copied unchanged |
| `policy.py` | `Agent` from `train.py` / `train_rgbd.py`: same UNet / DDPM (100 steps, squaredcos_cap_v2, epsilon, clip) |
| `obs_encoder.py` | `PlainConv` copied from `diffusion_policy/plain_conv.py`; `ObsEncoder` follows `Agent.encode_obs` of `train_rgbd.py` |
| `envs.py` | CPU branch of `make_env.py`, adapted to 3.0.1 |
| `data.py`, `evaluate.py`, `config.py`, `scripts/train_dp.py`, `scripts/eval_dp.py` | rewritten |

## Deliberate differences from the baseline

1. **Action normalization.** Actions are min-max scaled per dimension to [-1, 1]
   from the training demos (stats saved in the checkpoint). The baseline assumes the
   env action space already is [-1, 1] (true for `pd_ee_delta_pos`); absolute
   `pd_joint_pos` targets are not, and the DDPM sampler clips to [-1, 1].
2. **End padding for absolute control modes.** Absolute modes repeat the last
   action; the baseline only defines padding for delta modes.
3. **Final window included.** The window whose current step is the last action
   is also a training sample (the baseline's range stops one short).
4. **Sampling with replacement.** Batches are drawn uniformly with replacement from
   precomputed windows. The baseline's epoch sampler with `drop_last=True` yields no
   batches when there are fewer windows than `batch_size` (10 PickCube demos: 726 < 1024).
5. **Fixed evaluation seeds, split into validation and test.** Training-time
   evaluation (and `best.pt` selection) uses validation seeds; reported numbers come
   from `scripts/eval_dp.py` on disjoint test seeds. The baseline resets without seeds.
6. **No rendering during evaluation** unless videos are requested, so state-only
   evaluation does not need Vulkan.
7. **Logging** to JSON files under `results/<exp>/` instead of TensorBoard / W&B.
8. **One agent for state and rgb.** Observations are dicts (`{"state"}` or
   `{"state", "rgb"}`) that `ObsEncoder` turns into the UNet's conditioning vector:
   a plain flatten for state (no parameters, so state checkpoints are unchanged),
   PlainConv (256-d per frame, all cameras stacked on channels) + state for rgb.
   Images stay channel-last uint8 until the encoder, in the demos and the env alike.
9. **Frames stored once.** Training windows hold frame indices instead of copies,
   so images (uint8) are kept on the GPU once: 400 PegInsertionSide demos are about
   8 GB. The baseline also keeps rgb uint8 on the GPU, per episode.
10. **RGB eval cameras match the demo conversion**: `render_backend="cpu"` and the
   `minimal` sensor shader (`rgb_env_info` of the export), then
   `FlattenRGBDObservationWrapper(rgb=True, depth=False)` and `FrameStack`.

## Where RGB evaluation runs

Rendering after a scene reconfiguration (`reconfiguration_freq=1`, every reset) is
correct on Linux with the NVIDIA driver and with Mesa lavapipe (ubuntu, 9.26: identical
images with and without reconfiguration). On the Mac, MoltenVK renders every frame
after a reconfiguration green-washed, and fails to compile shaders in
`AsyncVectorEnv` workers, so the Mac cannot evaluate rgb policies (training works, on MPS).

Unchanged on purpose: observations are not normalized and images are not
augmented (train_rgbd.py defines no augmentation); EMA is created as
`EMAModel(power=0.75)` exactly as in the baseline (diffusers ignores `power`
without `use_ema_warmup=True`).
