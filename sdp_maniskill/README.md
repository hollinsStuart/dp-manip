# sdp_maniskill

Plugs ManiSkill data into [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) (tested against `5ba07ac`) for cluster training. Parameter values come from [docs/final-plan.md](../docs/final-plan.md).

| File | Role | Status |
| --- | --- | --- |
| `config/task/maniskill_<task>_lowdim.yaml` | Stanford **task** config: env, control mode, dims, episode length, data paths, dataset and runner | Done for the 6 tasks + LiftPegUpright fallback. PlugCharger `obs_dim` and LiftPegUpright `max_episode_steps` are `???` until their data is collected |
| `dataset.py` | `ManiSkillLowdimDataset`: reads `scripts/export_stanford_dp.py` output; separate val file; §3 normalization | Done |
| `env_runner.py` | `ManiSkillLowdimRunner`: validation rollouts on seeds 5000–5049 | **Not written.** The upstream workspace builds the runner before training, so training cannot start yet |
| workspace config | §3 training values (UNet size, batch 1024, 100k steps, `final.pt` only) | **Not written.** Upstream defaults differ, see below |

## Data layout

Per task, export on the Mac and copy to `data/maniskill/<task>/` next to upstream `train.py`:

```bash
python scripts/export_stanford_dp.py <pool>.state.<mode>.physx_cpu.h5 -o export/pickcube/pickcube_train.hdf5 --split train --num-demos 400 --subsets 25,50,100,200,400
python scripts/export_stanford_dp.py <val>.state.<mode>.physx_cpu.h5 -o export/pickcube/pickcube_val.hdf5 --split val --num-demos 50
```

The task config reads `<task>_train_n<N>.hdf5` and `<task>_val.hdf5`.

## Running (once the runner exists)

```bash
PYTHONPATH=/path/to/dp-manip python train.py --config-name=train_diffusion_unet_lowdim_workspace \
  task=maniskill_pickcube_lowdim task.num_demos=25 training.seed=1 \
  'hydra.searchpath=[file:///path/to/dp-manip/sdp_maniskill/config]'
```

## Upstream workspace defaults that differ from final-plan §3

`train_diffusion_unet_lowdim_workspace.yaml`: UNet `down_dims [256, 512, 1024]` and step embedding 256 (plan B0: `[64, 128, 256]`, 64, 4.4M params); batch 256 (plan 1024); training length in epochs (plan 100k steps); top-k checkpoints by `test_mean_score` (plan: `final.pt` only); `training.seed 42` (plan 1–5); EMA decay `1 − (1 + t)^−0.75` from Stanford's own `EMAModel` (plan: diffusers `EMAModel`, which ignores `power` and ramps as `(1 + t) / (10 + t)`; both cap at 0.9999). Horizons 16/2/8, DDPM 100 steps `squaredcos_cap_v2` ε-prediction with clipping, and AdamW 1e-4 with cosine and 500 warmup steps already match.

Upstream pins `hydra-core==1.2.0`, which does not import on Python 3.11; 1.3.2 composes these configs the same way.
