# Phase 0 — Frozen RGB baseline

This reference freezes the RGB pipeline immediately before the layered-config refactor.

## Reference identity

- Git commit: `834be805df77bd53043bf7270da3fef0fab38d55`
- Task: `PickCube-v1`
- Control mode: `pd_ee_delta_pos`
- Physics/evaluation backend: `physx_cpu`
- Machine-readable manifest: `baselines/phase0/pickcube_rgb.json`
- Reusable small-budget override: `configs/regression/pickcube_phase0.toml`

The formal scientific configuration is recorded under `canonical_config` in the manifest.
It preserves the pre-refactor horizons, visual encoder, UNet, optimizer, cosine warmup,
EMA, DDPM settings, batch size, training budget and evaluation seeds. The regression
override changes only execution budget: 25 training demos, one validation demo, five
optimizer steps, batch size one and two inference denoising steps.

Phase 7 amended the manifest to schema version 2 by adding the structural selector
`policy.backbone = "unet"` under `canonical_config.policy`. No scientific hyperparameter
value changed; the selector only records which noise-prediction backbone is canonical.

Phase 9 amended the manifest to schema version 3 by recording the donor Transformer
structure (`policy.transformer_layers/heads/embed_dim/dropout_emb/dropout_attn/
causal_attn/cond_layers`) migrated from VariDP. The fields are architecture definitions,
not scientific hyperparameters, and every arm resolves the same baseline values.

Phase 10 amended the manifest to schema version 4 by recording the donor MLP structure
(`policy.mlp_hidden_dim/mlp_layers/mlp_time_embed_dim`) migrated from VariDP, again
without changing any scientific hyperparameter.

Phase 11 amended the manifest to schema version 5 by adding `policy.mlp_obs_feat_dim = 256`:
the MLP arm regains VariDP's observation MLP (`To*Dobs -> 256 -> 256`), as registered in
`docs/final-plan.md` §6 B2. No scientific hyperparameter changed.

Phase 17 (cleanup) amended the manifest to schema version 6 by adding
`train.betas = [0.95, 0.999]`. The optimizer betas had been hardcoded in the trainer; the value
is unchanged and now resolves through `baseline.toml` like every other training
hyperparameter.

A second Phase 17 amendment (schema version 7) lowered `train.num_workers` from 8 to 4 to fit
the 4 CPUs of a GPU job on the HKU cluster (the Slurm scripts request `--cpus-per-task=4`).
Batches are drawn from `(seed, step)` and workers consume no randomness, so training is
unchanged.

The single-job dual-GPU refactor (schema version 8) lowered `train.num_workers` from 4 to 3
because one Slurm job now runs two trainers on 8 CPUs. Batches are still drawn from
`(seed, step)` and workers consume no randomness, so training is unchanged; single runs keep
overriding `train.num_workers` through the normal config layer.

## Frozen behavior

### Dataset and preprocessing

- HDF5 adapter reads `traj_i/obs_rgb/rgb`, `traj_i/obs_rgb/state` and `traj_i/actions`.
- RGB remains lazy-loaded per temporal window and is converted from `uint8` to `[-1, 1]`.
- Training augmentation is random shift with padding 4.
- The shared GroupNorm ResNet-18 emits 128 features per camera.
- The non-privileged 29-dimensional proprioception vector is z-scored from the selected
  training subset only. Observation frame `T` is excluded to align with actions `0:T`.
- Actions use training-subset min/max normalization inside DDPM and are restored to
  environment units before `env.step`; `pd_joint_pos` actions are not clipped.

### Policy and optimization

- Observation/prediction/action horizons: `2 / 16 / 8`.
- Conditional UNet dims: `[256, 512, 1024]`; diffusion embedding 256; kernel 5; groups 8.
- DDPM: squared-cosine schedule, epsilon prediction, sample clipping, 100 training and
  100 formal inference iterations.
- AdamW: LR `1e-4`, betas `(0.95, 0.999)`, weight decay `1e-6`.
- Cosine schedule with 500-step warmup; gradient norm clipping at 1.0.
- EMA target decay `0.9999`, using the original early-step decay ramp.

### Checkpoint contract

Inference checkpoints contain config, model, normalization, step, train-data metadata and
validation-data metadata. Resume checkpoints additionally contain optimizer, scheduler,
AMP scaler and EMA state. The exact version-1 keys and checkpoint SHA-256 values are in
the manifest. Checkpoint binaries remain in the ignored remote `runs/` directory and are
not committed.

## Captured regression result

The reference run used 25 train demos (seeds 0–24), one validation demo (seed 4000),
fixed training seed 1 and commit `834be80`.

| Metric | Captured value |
| --- | ---: |
| Parameters | 80,957,124 |
| Training loss, steps 1–5 | 1.5008505583, 0.8145254850, 1.0319931507, 0.9992038608, 0.8655537367 |
| Validation loss, steps 1 and 5 | 1.0786543849, 1.0720566778 |
| Validation rollout seed | 5000 |
| Episode length | 100 |
| `success_once` / `success_at_end` | 0 / 0 |

The zero rollout success is expected for a five-step model; this run verifies the lifecycle,
not task performance. The verified lifecycle was:

```text
read RGB dataset → train → save step/final/resume checkpoints
                 → load final checkpoint → closed-loop physx_cpu evaluation
```

## Reproduction

With the exported dataset rooted at `$DATA_ROOT`:

```bash
python scripts/train_dp.py \
  --config configs/tasks/pickcube.toml \
  --experiment configs/regression/pickcube_phase0.toml \
  --data-root "$DATA_ROOT" \
  --device cpu \
  --exp phase0_pickcube_rgb_regression

python scripts/eval_dp.py \
  runs/phase0_pickcube_rgb_regression/checkpoints/final.pt \
  --split val --episodes 1 --num-envs 1 --device cpu --render-backend cpu
```

For an exact historical replay, detach commit `834be80` and express the same regression
values with repeated `--set` flags, because that commit predates layered configs.

## Environment notes

The captured run used CPU because Torch `2.14.0+cu130` did not include kernels for the
available GTX 1080 Ti (`sm_61`), and installing a compatible CUDA build exceeded the
remote disk quota. Formal experiments are still required to run on compatible cluster
GPU nodes. A pre-existing bug was also observed: on commit `834be80`, `--resume never`
mistakes the newly created checkpoint directory for a non-empty run, so the reference
used the default `--resume auto`.
