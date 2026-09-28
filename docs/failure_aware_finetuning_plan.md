# Failure-Aware Diffusion Policy Plan

> Project: DASC7606C ManiSkill RGB Diffusion Policy  
> Status: Implementation-ready working plan  
> Branch: `exp/failure-aware-dp`  
> Primary idea: learn a separate failure action distribution from baseline-policy rollouts and use it as negative guidance during diffusion inference.  
> Reference idea: *Failing Forward: Adaptive Failure-Informed Learning for Vision-Language-Action Models* (AFIL), arXiv:2605.08434.

### Revision notes (2026-09-28)

The method (§2, §5.1–5.3) is unchanged. The experimental design was tightened so that a positive result can actually be attributed to *failure* modeling:

1. **Controls added** (§6.1): negative guidance from a model fine-tuned on the baseline's *successful* rollouts (same data source, same budget), and self-imitation on those successes. Without them, a gain could simply come from "guidance away from any fine-tuned copy of the policy" (autoguidance effect) or from "more on-policy data".
2. **Collect once, split both ways** (§3.3): the same rollouts provide the failure set *and* the success set for the controls; successes are no longer dropped.
3. **Multiple baseline checkpoints** (§4.1): the method is applied independently to training seeds 1–3 of the baseline, and results are paired per checkpoint.
4. **Task-selection band** (§1): tasks near 0 % or near 100 % baseline success are excluded.
5. **Seed ranges corrected** to the values actually enforced by `dp-manip` (§3.5), plus a dedicated guidance-tuning range.
6. **Adaptive-λ scale** (§5.2, §5.5): with realistic cosine values `(1 − c)/2` is ≪ 1, so the adaptive arm would be almost a no-op on the fixed-arm α grid. The adaptive grid is now calibrated to the same mean λ.
7. **Offline gate** (§6.7) before closed-loop tuning, a pre-registered primary hypothesis and statistics (§6.5), an `α = 0` bit-exact reproduction test (§8), and an evaluation budget (§10).

### Parameters locked (2026-09-28)

All open parameters are fixed in **§11** before any failure-aware run. Main consequences: PegInsertionSide only, with a task-agnostic interface for a later PlugCharger extension (§1, §12); failure trajectories truncated at 1.5 × the median successful length (§3.3); fine-tuning lr/steps chosen by an offline-only pilot (§4.4); C3 dropped; total budget capped at **24 GPU-hours** (§10).

---

## 1. Goal and Scope

The project already has a normal RGB-based Diffusion Policy baseline trained from successful expert demonstrations. The failure-aware study should extend that baseline with minimal architectural disruption and limited additional GPU cost.

The current preferred design is **not** an auxiliary binary failure classifier. Instead, it uses two action generators:

```text
                     ┌── Success Diffusion Policy ──> eps_success
RGB + proprioception ┤
                     └── Failure Diffusion Policy ──> eps_failure
                                           │
                                           ▼
                                  Negative Guidance
                                           │
                                           ▼
                                        Action
```

The success model learns the normal successful action distribution. The failure model learns the action distribution observed in failed closed-loop rollouts of the baseline policy. During denoising, the final prediction moves toward the success model and away from the failure model.

### Primary research question

> Can a standard RGB Diffusion Policy become more robust on difficult ManiSkill tasks by explicitly modeling its own failure distribution and using that distribution as negative guidance during diffusion inference?

### Scope constraints

- Keep the existing RGB observation pipeline.
- Keep the existing U-Net / noise-predictor architecture.
- Reuse the trained baseline checkpoint.
- Do not require frame-level failure-onset annotation.
- Run the study on **PegInsertionSide only**; keep everything task-agnostic so PlugCharger can be added later (§12).
- Keep baseline evaluation seeds untouched.
- Avoid unrelated refactors in this branch.

### Task (fixed)

The study runs on **PegInsertionSide-v1 (task 5)**: `pd_joint_pos`, 8-dim actions, 300-step episodes. Nothing in the code or protocol may be specific to this task; PlugCharger-v1 (task 6) is the planned extension and must only need a new per-task lock file (§12).

**Baseline cell rule.** Read the main-track **validation-rollout** results only (`success_once`, seeds 5000–5049, `final.pt`, mean over training seeds 1–5); never test results.

1. Use the `UNet · N = 100` cell if its mean validation success is **≥ 0.15**.
2. Otherwise use the track-A `UNet · N = 200` cell if its mean is ≥ 0.15. This mirrors the main project's rule for hard tasks in track B (`docs/final-plan.md` §6).
3. Otherwise use whichever of the two cells has the higher mean, and run in **low-success mode**:
   - collection keeps the 600-rollout cap; K is lowered to the number of successes found (§3.4);
   - if a checkpoint yields fewer than **50** successes, C1/C2 are not run for the whole study, H2 is reported as untestable, and the offline gate (§6.7) uses the expert validation demonstrations (seeds 4000–4049) in place of held-out successful rollouts.

Why a floor: near 0 % success, "failure" is the policy's whole action distribution, guidance has nothing to contrast against, and the success-rollout control cannot be trained. The rule is written down before any failure-aware run and only reads baseline-track results, so it cannot favor the method.

---

## 2. Final Method

### 2.1 Success policy

The success policy is the existing trained RGB Diffusion Policy:

```text
successful expert demonstrations
        ↓
RGB Diffusion Policy
        ↓
baseline checkpoint
```

During the failure-aware experiment:

```text
success_policy = baseline checkpoint
success_policy parameters = frozen
```

No retraining of the success policy is required.

### 2.2 Failure policy

The failure policy uses the **same `DiffusionPolicy` architecture** as the success policy.

It is initialized from the baseline checkpoint rather than trained from scratch:

```text
baseline checkpoint
        ↓ copy
failure policy
        ↓
fine-tune on failed baseline rollouts
```

Recommended MVP training policy:

```text
Observation / vision encoder   frozen
Noise predictor / U-Net        trainable
Diffusion scheduler            unchanged
Action normalization           copied from baseline
Proprio normalization          copied from baseline
```

Freezing the observation encoder has three benefits:

1. reduces additional compute;
2. reduces overfitting on a small failure dataset;
3. keeps success and failure policies in the same observation-feature space.

Implementation notes that follow from the current `dp-manip` code:

- The ResNet-18 encoder uses **GroupNorm**, not BatchNorm, so a frozen encoder has no running statistics that could drift during fine-tuning; its state dict stays bit-identical to the baseline's.
- Set `requires_grad = False` on the encoder **before** constructing `EMA`: `dp_manip.training.EMA` only shadows trainable parameters, so the EMA then covers the noise predictor only.
- Initialize from the baseline's **EMA weights** (the `model` entry of `final.pt`); no optimizer state is carried over. The EMA ramp `min(0.9999, (1+n)/(10+n))` restarts at `n = 0`, so after 5k–20k steps the effective decay is ≈ 0.998–0.9995 — the failure checkpoint reflects the last ~0.5k–2k steps, not the initialization.

The failure policy therefore learns approximately:

```text
p_failure(action | observation)
```

while the success policy models:

```text
p_success(action | observation)
```

### 2.3 What actually changes in the model

The original `DiffusionPolicy` and U-Net do **not** need a new failure head.

The new model-level component is a wrapper:

```text
DiffusionPolicy (success) ─┐
                           ├── FailureGuidedPolicy
DiffusionPolicy (failure) ─┘
```

Conceptually:

```python
class FailureGuidedPolicy:
    def __init__(self, success_policy, failure_policy, alpha, adaptive):
        ...
```

The wrapper combines the two noise predictions during diffusion sampling.

---

## 3. Failure Data Collection

### 3.1 Data source

Failure data should come from **closed-loop rollouts of the trained baseline policy**.

```text
expert demonstrations
        ↓
train baseline
        ↓
baseline checkpoint
        ↓
closed-loop ManiSkill rollout
        ↓
     episode
    /       \
success   failure
  save       save
  (controls) (method)
```

This produces failures that are close to the policy's real deployment-time failure distribution. Successful episodes from the **same** rollouts are kept too: they feed the control arms (§6.1) and the offline discriminability gate (§6.7).

Do **not** use random-action trajectories as the main negative dataset.

### 3.2 Failure criterion

For the first implementation:

```text
failure = success_once == False
```

An episode is considered a failure only if the task was **never successful at any point** during the rollout.

This is preferred over `success_at_end == False`, because an episode that completes the task and later disturbs the object should not automatically be grouped with episodes that never solved the task.

### 3.3 Whole-trajectory labeling

If an episode fails, save the **whole trajectory** as failure data.

Do not manually annotate:

```text
frames 0-40 = good
frames 41-70 = bad
```

The first version deliberately avoids defining a precise failure onset.

Adaptive guidance later reduces the impact of failure data in regions where the success and failure predictors agree.

Successful episodes are stored as a separate dataset from the same rollouts, **truncated at the end of the executed action chunk that contains the first success step** (at most `act_horizon − 1 = 7` steps after it; `RGBWindowDataset` pads the tail by repeating the last action, as for expert data). This matches the expert-demonstration convention and keeps post-success idling out of the imitation/control data.

**Failure truncation.** Failed episodes run to `max_episode_steps` (they must, to establish `success_once == False`), but a failed policy often stalls for most of the episode, and those near-static windows would dominate the failure model's training data. Failures are therefore stored truncated to

```text
L_fail = min(max_episode_steps, ceil_to_multiple_of_8(1.5 × median(L_succ)))
```

where `L_succ` are the stored lengths of the K success-train episodes of the **same checkpoint**. The same `L_fail` is applied to that checkpoint's holdout failures (§6.7). `L_fail`, and the fraction of failure steps discarded by it, are recorded in the dataset metadata.

### 3.4 Collection budget and fixed dataset size

Per task **and per baseline checkpoint** (§4.1):

```text
Collection rollouts (train):   seeds from 20000 upward, in seed order, until both classes reach K (cap: 600)
Collection rollouts (holdout): seeds 21000-21099, 100 episodes, all kept
Failure-train size K:          first K failures in seed order, K = 150
Success-train size K:          first K successes in seed order, same K
```

The dataset **size** is fixed, not the rollout count: every checkpoint's failure model and success-control model is trained on exactly K episodes, taken as a seed-ordered prefix (the same rule `dp-manip` already uses for the N-demo subsets). Rollouts proceed in blocks of `num_envs` and stop as soon as both classes have K episodes; with a baseline success rate `p` this needs about `K / min(p, 1 − p)` rollouts (300 at `p = 0.5`, 600 at `p = 0.25`). If the 600-rollout cap is reached before both classes have K episodes, K for that checkpoint is lowered to the smaller class count and the deviation is reported; the collection is not extended. Record the actual rollout count.

Perturbations (wider initialization, camera or action noise) are **not** part of the default design: the §1 band guarantees enough failures without them, and perturbed failures would come from a different state distribution than the test set. If they ever become necessary, they must be declared before collection and applied identically to the failure and success datasets.

### 3.5 Seed allocation

Keep failure-data seeds disjoint from every existing expert and evaluation range.

Ranges already fixed by `dp-manip` (`configs/baseline.toml`, `trainer.check_dataset`) and the new reserved ranges:

```text
0-3999          expert training demonstrations          (existing, enforced)
4000-4999       expert validation demonstrations        (existing, enforced)
5000-5049       policy validation rollouts              (existing; used for task selection only)
10000-10099     final test rollouts                     (existing; touched once, §6.4)
20000-20999     collection rollouts -> train sets        (new)
21000-21999     collection rollouts -> holdout sets      (new; offline gate, failure-model val loss)
22000-22099     guidance-tuning rollouts                 (new; 22000-22047 used for alpha selection, §5.5)
```

The exact number of used seeds may be smaller, but these ranges stay reserved for this study.

`trainer.check_dataset` currently rejects training data with seeds ≥ 4000. Do not relax that check globally; add a dataset-kind aware rule (`expert` → `[0, 4000)`, `rollout-train` → `[20000, 21000)`, `rollout-holdout` → `[21000, 22000)`) and keep the existing overlap check against validation and test rollout seeds.

### 3.6 Dataset schema

Failure datasets should preserve the existing `dp-manip` RGB trajectory schema wherever possible:

```text
traj_i/
├── obs_rgb/
│   ├── rgb          (T+1, H, W, 3*C)
│   └── state        (T+1, P)
├── actions          (T, A)
├── success
├── terminated
└── truncated
```

JSON metadata should additionally record provenance:

```text
dataset_type            rollout_failure | rollout_success
split                   train | holdout
source_checkpoint
source_checkpoint_hash
git_revision
task
env_id
control_mode
collection_seed
inference_seed
first_success_step      (successes only; truncation point)
success_once
success_at_end
episode_length
collection configuration
```

The failure dataset must be auditable back to the exact baseline checkpoint that generated it.

---

## 4. Failure Policy Fine-Tuning

### 4.1 Initialization

The failure policy must start from the **same baseline checkpoint** used to collect failures.

```text
failure_policy <- baseline checkpoint
```

**Multiple checkpoints.** Failures are on-policy, so they are specific to one checkpoint. Run the whole pipeline (collect → fine-tune → tune α → test) independently for the baseline checkpoints of **training seeds 1, 2, 3** (`peginsertionside_unet_n<N>_s{1,2,3}/final.pt`, with `N` from the §1 baseline cell rule). Every comparison in §6 is then paired *within* a checkpoint, which cancels most of the between-seed variance (σ ≈ 0.12 at N = 100 in the pilot) that dominates the main-track comparisons; what remains is the variance of the treatment effect itself. Seeds 4–5 are an optional extension if compute allows; the seed count is fixed before any test evaluation.

The first implementation should then:

```text
freeze observation_encoder
train noise_predictor only
```

### 4.2 Shared-coordinate-system invariant

This is a hard implementation requirement.

The success and failure predictions are only meaningfully comparable if both models operate in the same normalized action and diffusion coordinate system.

The failure policy therefore **MUST reuse the baseline checkpoint's**:

- action normalization statistics;
- proprio normalization statistics;
- observation layout and camera order;
- `obs_horizon`;
- `act_horizon`;
- `pred_horizon`;
- action dimension;
- diffusion beta schedule;
- number of diffusion training steps;
- prediction type (`epsilon`);
- backbone architecture.

In particular:

> **Do not recompute action min/max from the failure dataset.**

Otherwise `eps_success` and `eps_failure` would no longer describe the same action coordinate system.

### 4.3 Training objective

The failure model uses the standard diffusion noise-prediction loss on failed trajectories:

\[
L_{failure} = \|\epsilon - \epsilon_f(x_k, k, o)\|^2
\]

No classifier loss is required.

No reward model is required.

No failure-onset labels are required.

### 4.4 Initial training budget

Fixed recipe:

```text
initialization:    baseline checkpoint (EMA weights)
trainable module:  noise predictor only
lr schedule:       constant after 500 warmup steps   (not cosine, see below)
batch_size:        64, same as baseline
optimizer:         AdamW, betas / weight decay / grad clip / AMP as in baseline.toml
augmentation:      random_shift = 4, as in baseline
EMA:               baseline convention, restarted (§2.2)
training seed:     equal to the source checkpoint's training seed
learning_rate:     chosen by the offline pilot below, from {1e-5, 1e-4}
total_iters:       chosen by the offline pilot below, from {5k, 10k, 20k}
```

The schedule is constant (after warmup) so that a 5k or 10k checkpoint of a 20k run is the same model as a 5k or 10k run, apart from the EMA. That lets one pilot run cover all step counts. The trainer needs a `constant_with_warmup` option for this.

**Offline pilot (lr and steps).** `lr = 1e-5` for 10k steps may leave the failure model almost identical to the baseline, which would make guidance a no-op; `1e-4` may overfit 150 episodes. The choice is made **offline only**, without any closed-loop success measurement:

1. After collecting checkpoint 1's data, train two failure models on it: `lr = 1e-5` and `lr = 1e-4`, 20k steps each, saving EMA checkpoints at 5k / 10k / 20k (6 candidates).
2. For each candidate, compute `margin = gap_fail − gap_succ` (§6.7) on checkpoint 1's holdout seeds **21000–21049**.
3. Pick the candidate with the largest margin. If another candidate is within 5 % (relative) of it, prefer fewer steps, then the lower lr.
4. The chosen `(lr, steps)` is used for **every** failure and success model of every checkpoint. The chosen pilot checkpoint itself serves as checkpoint 1's failure model; it is not retrained.

The pilot never evaluates closed-loop, so it cannot select on success rate. The offline gate (§6.7) then uses the untouched holdout seeds 21050–21099.

The failure model used for guidance is always the checkpoint at the chosen step count; other checkpoints are offline diagnostics only.

The success-control model (§6.1, arm C1, which is also C2) uses exactly this recipe and the pilot-selected budget, only with the success dataset.

The main implementation should not require another 100k-step training run.

### 4.5 Trainer reuse

Do not create a second independent training stack.

Reuse the existing `dp_manip.trainer` pipeline and minimally extend it to support:

```text
initialization checkpoint
frozen module selection
externally supplied baseline normalization statistics
failure-dataset seed validation
```

A thin CLI wrapper is acceptable, but it should still call the same trainer implementation.

---

## 5. Failure Guidance Algorithm

### 5.1 Fixed negative guidance

At each diffusion denoising step, both models receive the **same**:

- observation features;
- noisy action sample `x_k`;
- diffusion timestep `k`.

Compute:

\[
\epsilon_s = f_s(x_k, k, o)
\]

\[
\epsilon_f = f_f(x_k, k, o)
\]

Then:

\[
\epsilon_{guided}
=
\epsilon_s + \lambda(\epsilon_s - \epsilon_f)
\]

or equivalently:

\[
\epsilon_{guided}
=
(1+\lambda)\epsilon_s - \lambda\epsilon_f
\]

For fixed guidance:

\[
\lambda = \alpha
\]

### 5.2 Adaptive guidance

For each batch item, flatten the prediction over prediction-horizon and action dimensions:

```text
(B, pred_horizon, action_dim)
        ↓ flatten last two dimensions
(B, pred_horizon * action_dim)
```

Compute cosine similarity:

\[
c_k = \cos(\operatorname{vec}(\epsilon_s),
           \operatorname{vec}(\epsilon_f))
\]

Use the bounded implementation:

\[
\lambda_k
=
\alpha\frac{1-c_k}{2}
\]

so that:

\[
0 \leq \lambda_k \leq \alpha
\]

Then:

\[
\epsilon_{guided}
=
\epsilon_s + \lambda_k(\epsilon_s-\epsilon_f)
\]

Interpretation:

```text
success and failure predictions similar
    -> lambda approximately 0
    -> little intervention

success and failure predictions diverge
    -> lambda increases
    -> stronger movement away from failure prediction
```

**Scale caveat.** Both predictors start from the same weights and receive the same `x_k`; at high-noise timesteps both predict roughly the injected noise, so `c_k` is expected to be close to 1 for most steps (e.g. `c = 0.98` gives `λ = 0.01 α`). On the same α grid, the adaptive arm would therefore apply ~1–5 % of the fixed arm's guidance and could look "worse" or "equal" simply because it is weaker. §5.5 calibrates the adaptive grid to the same **mean effective λ**, so that fixed vs. adaptive compares *where* guidance is applied, not *how much*.

### 5.3 Sampling contract

A single noisy action sample is initialized once:

```text
x_T ~ N(0, I)
```

For every diffusion step:

```text
same x_k
same timestep k
same observation features
        ↓
success predictor
failure predictor
        ↓
combine predictions
        ↓
one scheduler.step(...)
        ↓
x_{k-1}
```

Do not independently denoise two separate action samples and combine the final actions afterward.

### 5.4 Shared observation encoding

Because the failure observation encoder is frozen from the same baseline checkpoint, the preferred implementation may compute observation features once using the baseline encoder and feed the same features to both noise predictors.

This avoids unnecessary duplicate vision-encoder work at inference.

Make this the **required** implementation, not an optimization: computing features once guarantees both predictors see identical conditioning. `FailureGuidedPolicy` must assert at load time that the two `observation_encoder` state dicts (including the proprio normalization buffers) and `action_low/high` are bitwise equal, and refuse to run otherwise.

### 5.5 Guidance strength and selection

**Grids** (identical for every checkpoint and for the C1 control):

```text
fixed:     alpha ∈ {0.5, 1.0, 2.0}
adaptive:  alpha ∈ {0.5, 1.0, 2.0} / m
```

where `m` is the mean of `(1 − c_k)/2` over all denoising steps, measured in a **success-blind dry run**: 8 guidance-tuning episodes (seeds 22000–22007) of checkpoint seed 1, run at `α = 0` (the baseline trajectory, so `m` does not depend on any α) with the negative model evaluated only to log cosines (`GuidanceDiagnostics.mean_half_one_minus_cos`). `m` is computed once per task, recorded, and never re-estimated after success rates are seen. The adaptive arm then spans the same mean λ as the fixed arm.

**Tuning set:** seeds 22000–22047 (48 episodes) per checkpoint, pooled over checkpoints 1–3 (144 episodes per α). Spreading the tuning episodes over checkpoints keeps α from being fitted to one checkpoint; 48 per checkpoint is what the 24 GPU-hour budget allows (§10), and a multiple of `num_envs = 4` as `evaluate` requires. The 50-episode validation range 5000–5049 is not used: it is already consumed by task selection.

**Selection rule** (pre-registered):

1. For each arm, pool tuning episodes over checkpoints 1–3 (144 episodes per α).
2. Pick the α with the highest pooled `success_once`; ties within 2 episodes go to the **smaller** α. If the winner is at the edge of the grid, report that; the grid is not extended.
3. The same selected α is used for all checkpoints of that arm.
4. F vs A: the arm whose selected α has the higher pooled tuning success becomes "the guidance arm" for H1/H2 and fixes C1's guidance mode; ties go to F (the simpler one).
5. For the later PlugCharger extension, how α is transferred or re-tuned is defined in §12.

Arm B is not run on the tuning set: selection only compares α values within an arm and F against A.

---

## 6. Experimental Protocol

### 6.1 Arms

Every arm runs on each baseline checkpoint (seeds 1–3, §4.1). All fine-tuned models use the §4.4 recipe (baseline init, frozen encoder, pilot-selected lr and steps, K episodes).

| ID | Arm | Model(s) at inference | Guidance | Priority |
| --- | --- | --- | --- | --- |
| B | Baseline | baseline | none | required |
| F | Fixed Failure Guidance | baseline + failure model | fixed `λ = α` | required |
| A | Adaptive Failure Guidance | baseline + failure model | adaptive `λ_k` | required |
| C1 | Success-Rollout Guidance (control) | baseline + model fine-tuned on the K **successful** rollouts | same guidance mode as the better of F/A, own α tuned on the same grid | required |
| C2 | Self-Imitation (control) | the C1 success model itself, used as the policy | none | recommended |
| C3 | Naive Mixing (diagnostic) | baseline fine-tuned on success + failure rollouts as imitation targets | none | **dropped** (§11) |

What each control rules out:

- **C1** is the key control. Guiding a diffusion model away from a slightly different copy of itself can improve samples regardless of what that copy was trained on (the "autoguidance" effect). C1 has the same data source, dataset size, fine-tuning budget, guidance code and tuning budget as F/A; only the episode *label* differs. The claim "failure information helps" requires **A (or F) > C1**, not just > B.
- **C2** answers "why not just imitate your own successes?" — the standard, cheap way to use on-policy rollouts. It is the *same* checkpoint as C1's negative model, so it costs no extra training — only its evaluation.
- **C3** answers "is adding failed trajectories as positive data enough?". It deliberately treats failed actions as imitation targets, so it is a diagnostic, not a competitor.

C3 is dropped under the 24 GPU-hour budget (§11). If compute forces a further cut, follow §10. B, the guidance arm and C1 are the minimum for the report's claims.

### 6.2 Hypotheses (pre-registered)

- **H1 (primary):** on test seeds, the selected failure-guidance arm (A or F, whichever wins on the tuning set) has higher `success_once` than B.
- **H2 (attribution):** the same arm has higher `success_once` than C1.
- **H3 (secondary):** A ≥ F at matched mean λ (§5.5).

Only H1 is the primary claim. H2 decides how the result may be worded: if H1 holds but H2 does not, the report says "negative guidance from a fine-tuned copy helps", not "failure modeling helps". The F/A choice is made on the tuning set, never on test.

### 6.3 Fair comparison requirements

All arms must use:

- the same success-policy checkpoint within each pair;
- the same environment configuration (`physx_cpu`, `num_envs`, `max_episode_steps` from `configs/tasks/`);
- the same test seeds 10000–10099 and tuning seeds 22000–22047;
- the same inference seed (0) and the same RNG consumption per step, so that arms differ only through the guided `ε` (checked by the `α = 0` test in §8);
- the same observation history and action horizon.

Do not modify the test set while tuning.

### 6.4 Validation and test separation

| Seed range | Used for |
| --- | --- |
| 5000–5049 | task selection only (baseline-track results) |
| 21000–21049 holdout rollouts | lr/steps pilot (checkpoint 1 only, offline, §4.4) |
| 21050–21099 holdout rollouts | offline gate (§6.7), failure/success-model val loss |
| 22000–22007 | success-blind cosine dry run (`m`, §5.5) |
| 22000–22047 | α selection and F-vs-A choice |
| 10000–10099 | one final evaluation per (arm, checkpoint) after everything above is frozen |

Test evaluation happens once. No arm, α, K, checkpoint seed or failure-model checkpoint is changed after test results are seen.

### 6.5 Metrics and statistics

Primary metric `success_once`; secondary `success_at_end`, `return`, `episode_len`, `mean_inference_ms` (guided arms run two noise predictors per step, so latency roughly doubles for the UNet part; report it next to success).

Per (task, arm, checkpoint): successes / 100 with a 95 % Wilson interval. Per (task, arm): mean ± sample std over checkpoints and the range.

Paired comparisons (H1–H3) follow the main project's convention (`docs/final-plan.md` §4): pair on (checkpoint seed, test seed), count "only X succeeded" vs "only Y succeeded", and bootstrap the difference two-level (resample checkpoints, then episodes; 10,000 draws). A difference is claimed only if the 95 % interval excludes 0; otherwise report "not distinguishable at this sample size". No checkpoints or episodes are added after seeing results.

Rough sensitivity: 3 checkpoints × 100 paired episodes = 300 pairs per comparison. If ~20 % of pairs are discordant, the minimum detectable paired difference at 80 % power is roughly 7–8 percentage points; smaller gains will be reported as inconclusive.

### 6.6 Diagnostics

Log on tuning and test runs:

```text
cosine(eps_success, eps_failure)   per denoising timestep (mean, quantiles)
effective lambda                   per denoising timestep and per env step
fraction of denoising steps with lambda < 0.01 * alpha
||eps_guided - eps_success|| / ||eps_success||
predicted-x0 clip rate             fraction of samples hitting clip_sample=[-1, 1]
```

The clip rate matters because `DDPMScheduler` clips the predicted `x0`; large λ can push many samples into the clip and silently saturate actions.

Failure-type shift: classify every test episode as *succeeded and held* / *succeeded then lost* / *never succeeded* (`docs/final-plan.md` §4) and report how B → A changes these counts. A useful guidance should mainly convert "never succeeded" episodes; an increase in "succeeded then lost" is a warning sign.

### 6.7 Offline gate before closed-loop tuning

Before any closed-loop α sweep, check that the failure model learned something failure-specific. On the holdout rollouts (seeds 21050–21099; 21000–21049 are used by the §4.4 pilot), compute the denoising loss with a fixed noise/timestep seed for three models: baseline `L_B`, failure model `L_F`, success-control model `L_S`.

```text
gap_fail = L_B − L_F   on holdout failures
gap_succ = L_B − L_F   on holdout successes
```

Pass if `gap_fail > gap_succ` for every checkpoint, i.e. the failure model improved more on held-out failures than on held-out successes. Report the same quantities for `L_S` (mirror image expected).

If the gate fails, the failure model is not discriminative and closed-loop guidance would be uninterpretable. There is no further retry — the lr/steps choice was already made by the §4.4 pilot. Report the negative result (offline losses for all checkpoints) and stop, without sweeping further hyperparameters.

---

## 7. Repository and Code Organization

The current integrated repository is:

```text
7606C/
├── maniskill-demogen/   # expert demonstration generation
├── dp-manip/            # unified RGB training / evaluation pipeline
└── VariDP/              # frozen donor
```

### 7.1 Component ownership

`maniskill-demogen` should continue to generate successful motion-planning expert demonstrations.

Failure rollouts depend on a trained Diffusion Policy checkpoint, so failure-aware work belongs primarily in **`dp-manip`**.

`VariDP` remains frozen.

### 7.2 Recommended `dp-manip` additions

```text
dp-manip/
├── dp_manip/
│   ├── policy.py                 # Phase 3: get_action -> observation_features + sample_actions
│   ├── evaluate.py               # reuse; read-only RolloutObserver hook (Phase 1)
│   ├── data.py                   # reuse RGBWindowDataset unchanged
│   ├── failure_protocol.py       # NEW (Phase 1): loads and checks the protocol file
│   ├── failure_rollout.py        # NEW (Phase 1): rollout recording, raw files, dataset build
│   ├── finetune.py               # NEW (Phase 2): FinetuneSpec, freezing, init/seed/source checks
│   ├── trainer.py                # Phase 2: optional `finetune=` path; baseline path unchanged
│   ├── failure_guidance.py       # NEW (Phase 3): FailureGuidedPolicy, GuidanceDiagnostics
│   ├── failure_lock.py           # NEW (Phase 4): write-once per-task lock file
│   └── failure_study.py          # NEW (Phase 4): stage rules, arm resolution, offline losses
│
├── scripts/
│   ├── collect_rollouts.py       # NEW (Phase 1): `collect --split train|holdout`, `build`
│   ├── finetune_dp.py            # NEW (Phase 2): thin CLI over trainer.run_training
│   └── failure_study.py          # NEW (Phase 4): one subcommand per study stage, incl. `eval`
│
├── configs/
│   └── failure_aware/
│       ├── protocol.toml         # NEW (Phase 1): task-agnostic protocol (§12.1)
│       └── peginsertionside.toml # NEW: per-task lock file (§12.2); plugcharger.toml later
│
├── tests/
│   ├── test_failure_rollout.py   # NEW (Phase 1)
│   ├── test_finetune.py          # NEW (Phase 2)
│   ├── test_failure_guidance.py  # NEW (Phase 3)
│   └── test_failure_study.py     # NEW (Phase 4): rules, lock file, end-to-end study on fakes
│
└── slurm/
    └── failure_aware.sbatch      # NEW (Phase 4): one study step per single-GPU job, requeue on exit 75
```

### 7.3 `policy.py` change boundary

Do not put failure-specific branches into the baseline `DiffusionPolicy.get_action()` implementation.

A small reusable primitive is acceptable. Implemented (Phase 3): `get_action` is split into `observation_features` plus

```python
def sample_actions(self, obs_features, *, generator=None, noise_fn=None): ...
```

which owns the initial noise, the scheduler loop, RNG consumption and action slicing; `noise_fn(sample, timestep)` optionally replaces the policy's own noise prediction. `get_action` is unchanged in behavior. `FailureGuidedPolicy` samples through the baseline's `sample_actions`, so with `α = 0` it reproduces `get_action` bit for bit.

The failure-specific combination logic belongs in `failure_guidance.py`.

### 7.4 Git workflow

The course repository `hryang1130/7606C` currently vendors `dp-manip` as a Git subtree.

Therefore implementation should preferably happen first in the canonical `dp-manip` upstream, then be synchronized back into `7606C`.

Recommended branch:

```text
exp/failure-aware-dp
```

Recommended worktree / workspace name:

```text
dp-manip-failure-aware-dp/
```

Keep this branch limited to failure-aware work.

---

## 8. Implementation Phases and Acceptance Criteria

### Phase 1 — Failure rollout dataset

Implement:

```text
collect_rollouts.py
failure_rollout.py
HDF5 / JSON writer
provenance metadata
```

Acceptance criteria:

1. Load an existing baseline checkpoint.
2. Roll out a small fixed seed range.
3. Correctly classify `success_once`.
4. Save never-successful episodes to the failure dataset and successful episodes, truncated at `first_success_step`, to the success dataset (§3.3).
5. Preserve `T+1` observations vs `T` actions alignment.
6. Reload both datasets through `read_dataset_info` / `RGBWindowDataset` without special cases.
7. Record exact source checkpoint (path + sha256), inference seed and collection seeds.
8. The rollout recorder reuses `evaluate()`'s stepping logic; its per-episode `success_once` must equal `evaluate()`'s on the same seeds.

Do not modify the model in this phase.

### Phase 2 — Failure policy fine-tuning

Implement the minimum trainer extensions required for:

```text
baseline initialization
baseline normalization reuse
frozen observation encoder
failure-dataset seed rules
```

Acceptance criteria:

1. A 100-step CPU/GPU smoke run completes.
2. Only the intended trainable parameters receive gradients.
3. Baseline normalization is reused exactly.
4. A failure-policy checkpoint can be saved and reloaded via `DiffusionPolicy.from_checkpoint`.
5. Success/failure policy schemas match.
6. After fine-tuning, `observation_encoder.*`, `action_low` and `action_high` are bitwise equal to the baseline checkpoint.
7. Training data outside the dataset-kind seed range (§3.5) is rejected.
8. A `constant_with_warmup` lr schedule is available and recorded in `run.json` (§4.4).

### Phase 3 — Failure guidance

Implement `FailureGuidedPolicy`.

Required unit tests:

```text
alpha = 0
    -> guided prediction equals success prediction

success predictor == failure predictor
    -> adaptive lambda == 0

cosine = -1
    -> adaptive lambda == alpha

same noisy sample and timestep
    -> both predictors receive identical diffusion inputs

output action shape
    -> identical to DiffusionPolicy interface
```

Closed-loop criteria:

```text
success checkpoint + failure checkpoint
    -> guided policy runs a complete ManiSkill episode without API changes to evaluate(...)

alpha = 0, same seeds, same num_envs, same inference seed
    -> per-episode results (success_once, success_at_end, episode_len, return)
       identical to the plain baseline evaluate(...) output
```

The `α = 0` reproduction is the strongest end-to-end check that guided and baseline arms consume the same randomness and conditioning; every later paired comparison relies on it. If it passes on the cluster, existing baseline test results for the same `final.pt`, code revision and inference seed may be reused as arm B.

### Phase 4 — Experiment integration

Integrate:

- fixed guidance;
- adaptive guidance;
- alpha sweep on the guidance-tuning seeds;
- control arms C1 and C2;
- Slurm launch support;
- run naming;
- evaluation JSON metadata;
- diagnostics.

Acceptance criteria:

1. All arms of §6.1 run on the same tuning seeds (22000–22047) and test seeds (10000–10099).
2. The dry-run `m`, the α grids, and the selected α per arm are recorded before any test run.
3. Test evaluation cannot silently use a different success checkpoint or failure checkpoint.
4. Result files record both checkpoint paths/hashes and the guidance mode.

### Phase 5 — Final study

After the implementation is frozen:

1. fix PegInsertionSide's baseline cell with the §1 rule and write the per-task lock file (§12.2);
2. collect checkpoint 1's rollouts; measure GPU-hours per 100 episodes and re-project §10;
3. run the offline lr/steps pilot on checkpoint 1 (§4.4);
4. collect checkpoints 2–3; build failure/success datasets (K = 150, `L_fail` truncation);
5. fine-tune the remaining failure and success models;
6. pass the offline gate (§6.7), or stop and report;
7. success-blind dry run → `m` → adaptive grid;
8. α sweep on tuning seeds, choose α per arm and F vs A (§5.5); then C1's α sweep;
9. freeze everything, run the test set once per (arm, checkpoint);
10. report counts, paired bootstrap for H1–H3, latency, diagnostics, failure-type shift.

**Runbook (implemented in Phases 1–4).** Every step is one job: `sbatch slurm/failure_aware.sbatch <script> <arguments>` (the QOS allows one job per user). `T=peginsertionside`, `CKPT_s` is checkpoint `s`'s `final.pt` as locked in `[checkpoints]`, `DIR_s` its rollout directory `<run root>/failure_aware/T/s<s>`:

| Step | Command | Locks |
| --- | --- | --- |
| 1 | `scripts/failure_study.py select-cell --task T --run-root <main-track run root>` | `[task]`, `[checkpoints]` |
| 2, 4 | per checkpoint: `scripts/collect_rollouts.py collect CKPT_s --split train`, `... --split holdout`, `... build DIR_s`, then `scripts/failure_study.py record-collection --task T --seed s` | `[collection.s<s>]` |
| 3 | `scripts/finetune_dp.py CKPT_1 --label failure --lr <lr> --steps 20000 --checkpoint-steps 5000 10000` for each pilot lr, then `scripts/failure_study.py pilot --task T` | `[finetune]` |
| 5 | `scripts/finetune_dp.py CKPT_s --label failure\|success --lr <locked lr> --steps <locked steps>` for the remaining models, then `record-models --task T --seed s` | `[models.s<s>]` |
| 6 | `scripts/failure_study.py gate --task T` | `[gate]` |
| 7 | `scripts/failure_study.py dry-run --task T` — also the `α = 0` reproduction check (§8 Phase 3) on the same 8 episodes | `[guidance.dry_run]` |
| 8 | `eval --task T --split tuning --arm F`, `--arm A`; `select-alpha`; `eval --split tuning --arm C1`; `select-c1` | `[guidance.selection]`, `[guidance.c1]` |
| 9 | commit the lock file; `eval --task T --split test --arm B\|F\|A\|C1\|C2` | — |

`scripts/failure_study.py status --task T` prints the locked stages and the GPU-hours recorded so far (re-projection, §10.3).

---

## 9. Expected Final Story for the Report

The intended experimental narrative is:

```text
Successful expert demonstrations
        ↓
train standard RGB Diffusion Policy
        ↓
baseline policy
        ↓
collect the baseline policy's own failed rollouts
        ↓
fine-tune a second DP to model failure actions
        ↓
compare (paired, same checkpoint, same test seeds):
    baseline
    fixed / adaptive negative guidance from the failure model
    negative guidance from a success-rollout model   (control)
    self-imitation on successful rollouts            (control)
        ↓
evaluate on exactly the same held-out test seeds
```

The study tests progressively stronger claims, each with its own evidence:

1. A standard Diffusion Policy provides the baseline performance (main track).
2. Explicitly modeling the policy's failure distribution gives a usable negative action model — offline gate §6.7.
3. Negative guidance from that model improves closed-loop success — H1.
4. The improvement comes from *failure* information, not from guidance against any fine-tuned copy or from extra on-policy data — H2 and C2.
5. Adaptive weighting is better than uniform weighting at the same mean strength — H3.

A negative or inconclusive answer at any step is still a reportable result; the pre-registered rules above decide the wording.

The implementation is intentionally conservative: the original successful policy remains frozen and checkpoint-compatible, while the new failure-specific behavior is isolated in a second noise predictor and an inference-time wrapper.

---

## 10. Evaluation and Training Budget

**Cap: 24 GPU-hours in total** for the PegInsertionSide study (3 checkpoints). The PlugCharger extension is budgeted separately (§12.4). One GPU-hour = one GPU allocated for one hour of wall time, including rollouts whose time is mostly CPU physics and rendering.

### 10.1 Unit costs

| Unit | GPU-hours | Source |
| --- | ---: | --- |
| 10k fine-tuning steps (batch 64) | 0.4 | **measured** speed: UNet smoke on RTX 4080 SUPER, 0.13 s/step (`../4080_test_results.md`), plus ~5 min overhead; freezing the encoder can only make it faster |
| 100 baseline episodes | 0.25 | **estimate, not measured** (PegInsertionSide length 300, DDPM 100 steps, `num_envs = 4`) |
| 100 guided episodes | 0.40 | estimate: shared encoder + two noise predictors ≈ 1.6× baseline |

Everything below assumes the worst-case episode length (PegInsertionSide, 300 steps); shorter tasks (100–200 steps) cost proportionally less. The rollout unit costs are the largest uncertainty and are **re-measured on checkpoint 1's collection** (§8 Phase 5, step 2) before anything else is committed.

### 10.2 Plan

| Stage | Episodes (guided) | Typical GPU-h | Worst GPU-h | Notes |
| --- | ---: | ---: | ---: | --- |
| Collection, 3 checkpoints | 1,500 (0) – 2,100 (0) | 3.8 | 5.3 | train until both classes reach K (300–600) + 100 holdout |
| lr/steps pilot (checkpoint 1) | — | 1.6 | 1.6 | 2 runs × 20k steps; the chosen one is reused as checkpoint 1's failure model |
| Remaining fine-tunes | — | 2.0 | 4.0 | 5 runs (ckpt 1 success; ckpt 2–3 failure + success) at 10k or 20k steps |
| `α = 0` reproduction check | 8 (8) | 0.05 | 0.05 | same episodes as the dry run, plus 8 baseline episodes |
| Dry run for `m` | 8 (8) | 0.05 | 0.05 | checkpoint 1 |
| Tuning: F, A, C1 × 3 α × 48 eps × 3 ckpts | 1,296 (1,296) | 5.2 | 5.2 | §5.5 |
| Test: F, A, C1 × 3 ckpts × 100 | 900 (900) | 3.6 | 3.6 | |
| Test: C2 × 3 ckpts × 100 | 300 (0) | 0.75 | 0.75 | |
| Test: B × 3 ckpts × 100 | 300 (0) | 0 | 0.75 | reused from the main track if the `α = 0` check passes; otherwise rerun |
| **Total** | | **≈ 17.1** | **≈ 21.4** | reserve 2.6–6.9 GPU-h for failed or preempted jobs |

### 10.3 If the re-projection exceeds 24 GPU-hours

After step 2 of Phase 5, recompute §10.2 with the measured rollout cost. If the worst case exceeds 24 GPU-hours, apply these cuts **in order**, only as far as needed, and record them in §11 before continuing:

1. Holdout rollouts 100 → 60 per checkpoint (pilot 21000–21029, gate 21030–21059).
2. Tuning 48 → 28 episodes per checkpoint (84 pooled per α).
3. Drop C2 from the test set (the self-imitation question is then reported as not tested).
4. Drop the losing F/A arm from the test set (H3 is then reported from tuning episodes only, descriptively, with no claim).

Never cut: the 3 checkpoints, 100 test episodes per (arm, checkpoint), and the arms B, guidance arm, C1.

---

## 11. Locked Parameters (2026-09-28)

Fixed before any failure-aware run. Changing any of them after tuning or test results exist invalidates the pre-registration; a change made before that is recorded here with its date and reason.

Change log (all before any failure-aware run):

- 2026-09-28 — guidance tuning 50 → **48** episodes per checkpoint and the `m` dry run 10 → **8** episodes. `evaluate` runs full waves of `eval.num_envs = 4` episodes; 50 and 10 are not multiples of 4. The machine-readable values live in `configs/failure_aware/protocol.toml`, and `FailureProtocol.check_against` rejects counts that do not divide `num_envs`.

| # | Parameter | Value | Section |
| --- | --- | --- | --- |
| 1 | Task | PegInsertionSide-v1 only; PlugCharger via §12 later | §1 |
| 2 | Baseline cell | N = 100 if mean val success ≥ 0.15, else N = 200 if ≥ 0.15, else higher of the two in low-success mode (C1/C2 dropped if < 50 successes) | §1 |
| 3 | Baseline checkpoints | training seeds 1, 2, 3 of the chosen cell | §4.1 |
| 4 | Failure truncation | `L_fail = min(max_steps, ceil8(1.5 × median success length))`, per checkpoint | §3.3 |
| 5 | Dataset size K | 150 failures and 150 successes per checkpoint; rollout cap 600 | §3.4 |
| 6 | Fine-tune lr and steps | offline pilot: lr ∈ {1e-5, 1e-4} × steps ∈ {5k, 10k, 20k}, max `gap_fail − gap_succ` | §4.4 |
| 7 | C2 data | the K successful rollouts only (C2 = C1's success model) | §6.1 |
| 8 | C3 | dropped | §6.1 |
| 9 | α grid | fixed {0.5, 1, 2}; adaptive {0.5, 1, 2} / `m`; not extended | §5.5 |
| 10 | Budget | ≤ 24 GPU-hours; cut order §10.3 | §10 |

Defaults (no discussion needed, listed so they are not changed silently):

| Parameter | Value |
| --- | --- |
| Collection inference seed / `num_envs` | 0 / 4, as in the main track |
| Success-episode truncation | end of the action chunk containing the first success step |
| Failure criterion | `success_once == False` |
| Fine-tune schedule | constant lr after 500 warmup steps |
| Fine-tune batch / optimizer / AMP / augmentation | as in `baseline.toml` (64 / AdamW / on / random_shift 4) |
| Fine-tune training seed | equal to the source checkpoint's seed |
| Guidance timesteps | all 100 denoising steps |
| C1 guidance mode | the tuning-set winner of F vs A; ties → F |
| α tie-break | within 2 pooled tuning episodes → smaller α |
| Evaluation `num_envs` | 4 for every rollout (required by the `α = 0` check) |
| Episodes | tuning 48 per checkpoint; test 100 per checkpoint (10000–10099) |
| Failure types | automatic 3-way classification for all test episodes; PegInsertionSide stage subtypes from videos, 5 episodes per class per arm |
| Offline gate | fixed noise/timestep seed, all windows of holdout 21050–21099; pass iff `gap_fail > gap_succ` on all 3 checkpoints |
| Prerequisite | main-track PegInsertionSide `unet_n100` seeds 1–5 (and `unet_n200` if §1 falls back to it) `final.pt` and their validation rollouts are finished |

---

## 12. Extension Interface: PlugCharger (task 6)

No PlugCharger run is part of the current study. This section fixes what the PegInsertionSide implementation must look like so that PlugCharger (or any other task) can be added later by writing configuration only.

### 12.1 Task-agnostic requirements (apply now)

- **No task branches.** Every new script takes `--task <name>` and resolves `configs/tasks/<task>.toml`, like `scripts/run_experiment.py`; no `if task == ...` anywhere. Env id, control mode, `max_episode_steps`, cameras and action dimension come from the task config and the baseline checkpoint.
- **Derived lengths come from data or config.** `L_fail` from the success lengths (§3.3), success truncation from `act_horizon`, episode length and Slurm time limits from `max_episode_steps`. Nothing hard-codes 300.
- **Seed ranges are task-independent.** The ranges in §3.5 are the same numbers for every task.
- **Output layout and run names include the task:**

  ```text
  <run_root>/failure_aware/<task>/s<ckpt_seed>/{rollouts,failure_model,success_model,eval}/
  <task>_unet_n<N>_s<ckpt_seed>_<arm>[_a<alpha>]
  ```

- **Two config layers.**
  - `configs/failure_aware/protocol.toml` — task-agnostic protocol (not under `configs/experiments/`, whose files are experiment-grid specs): arms, α grids, K, rollout cap, truncation factor 1.5, pilot grid, tuning/test episode counts, seed ranges.
  - `configs/failure_aware/<task>.toml` — per-task **lock file**, filled stage by stage and never edited after the stage that fills it:

  ```toml
  [task]                 # name, baseline cell N, mode (normal | low-success), val success per seed
  [checkpoints.s1]       # path, sha256 (one table per ckpt seed)
  [collection.s1]        # rollouts, successes, failures, K, L_fail, low_success, summary sha256
  [finetune]             # lr, steps, source = "pilot" | "transferred:<task>", pilot candidates and margins
  [models.s1]            # failure / success model paths and sha256
  [gate]                 # passed; per ckpt seed: losses, gap_fail, gap_succ, pass
  [guidance.dry_run]     # m, mean cosine, alpha0_reproduces_baseline
  [guidance.selection]   # per arm: grid value, alpha, pooled successes, grid edge; guidance_arm
  [guidance.c1]          # C1 grid value, mode, alpha, pooled successes
  ```

  `dp_manip/failure_lock.py` writes each section once and refuses to replace it. GPU-hours are not locked (they grow); `failure_study.py status` sums them from the rollout, fine-tuning and evaluation records.

  Test results live in the evaluation JSON files, never in the lock file. Evaluation scripts read checkpoints, α and guidance mode **only** from the lock file (§8 Phase 4, criterion 3).
- **Tests cover both tasks from the start.** Config-resolution and lock-file tests, and the synthetic-data dataset/rollout-schema tests, are parametrized over `peginsertionside` and `plugcharger`, so the PlugCharger path is exercised even though it is not run.

### 12.2 PegInsertionSide lock file

`configs/failure_aware/peginsertionside.toml` is written during §8 Phase 5 by the runbook's commands: `[task]` and `[checkpoints]` in step 1, `[collection.*]` in steps 2 and 4, `[finetune]` in step 3, `[models.*]` in step 5, `[gate]` in step 6, `[guidance.*]` in steps 7–8. It is committed before the test evaluation (step 9).

### 12.3 What transfers to PlugCharger and what is re-derived

Default extension design: a **reduced replication with transferred hyperparameters**. It tests whether the method, tuned on PegInsertionSide, works on a second task without re-tuning, and it is the cheapest option.

| Item | PlugCharger |
| --- | --- |
| Baseline cell, low-success mode | re-derived with the §1 rule on PlugCharger's own validation results |
| Checkpoints | training seeds 1–3 of that cell |
| K, rollout cap, truncation factor, seed ranges, episode counts | same as PegInsertionSide |
| `L_fail` | re-derived from PlugCharger's own success lengths |
| Fine-tune lr and steps | transferred from the PegInsertionSide lock file (no pilot) |
| `m` | re-measured (8-episode success-blind dry run) |
| Guidance arm (F or A) | transferred |
| α | the selected grid value from `{0.5, 1, 2}` is transferred; the adaptive arm divides it by PlugCharger's own `m`; C1's grid value is transferred the same way |
| Offline gate | re-run on PlugCharger's holdout |
| Test arms | B, guidance arm, C1; C2 only if budget allows |
| Hypotheses | H1 and H2, reported as a replication; H3 not tested |

The alternative is the **full design** on PlugCharger (own pilot and α sweep, as in §4.4 and §5.5). Which of the two designs is used must be written into `configs/failure_aware/plugcharger.toml` before any PlugCharger collection starts.

### 12.4 Budget estimate

PlugCharger episodes are 200 steps, so rollouts cost about 2/3 of PegInsertionSide's; fine-tuning costs the same. Unit costs are re-measured on the first PlugCharger collection, as in §10.

| Design | Typical GPU-h | Worst GPU-h |
| --- | ---: | ---: |
| Reduced replication (default) | ≈ 6.8 | ≈ 10.7 |
| Full design | ≈ 12.9 | ≈ 16.4 |

### 12.5 Prerequisites

- The PegInsertionSide lock file is complete (it supplies the transferred values).
- PlugCharger's main-track `unet_n100` seeds 1–5 (and `unet_n200` if the §1 rule falls back) are finished with validation rollouts.
- A separate budget is approved; the extension does not draw on the 24 GPU-hours of §10.

---

## Appendix A — Alternative Design: Auxiliary Failure Head

An earlier design proposed attaching a binary failure classifier to the shared observation representation:

```text
                         ┌── Conditional U-Net ──> action / noise prediction
RGB ──> Vision Encoder ──┤
                         └── Failure Head ───────> p(failure)
```

This remains a valid alternative experiment, but it is **not the preferred MVP**.

Its main limitation is that a classifier can learn to recognize failure without directly specifying how the action distribution should change to avoid it.

It may still be useful later as:

- a diagnostic failure detector;
- a comparison against action-space negative guidance;
- a confidence / early-warning signal;
- an auxiliary head in a more advanced architecture.

Do not implement this head in the first failure-aware branch unless the dual-policy guidance plan is blocked.

---

## Appendix B — Decisions Intentionally Deferred

The following are not required for the first implementation:

- frame-level failure-onset detection;
- failure-recovery demonstrations;
- contrastive representation learning;
- reward modeling;
- online iterative failure recollection;
- shared-weight dual heads inside one U-Net;
- failure-conditioned U-Net adapters;
- OOD perturbation benchmark beyond what is needed to obtain enough failures.

These can be considered only after the MVP experiment runs end to end.
