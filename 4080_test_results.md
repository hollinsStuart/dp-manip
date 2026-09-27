## System resources

```bash
u3684139@gpu-4080-403:~/7606C/dp-manip$ echo "=== SLURM allocation ==="
echo "JOB_ID=$SLURM_JOB_ID"
echo "CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"
echo "CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "MEM_PER_NODE=$SLURM_MEM_PER_NODE"
echo "MEM_PER_CPU=$SLURM_MEM_PER_CPU"
echo "GPUS=$SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
=== SLURM allocation ===
JOB_ID=135448
CPUS_ON_NODE=4
CPUS_PER_TASK=
MEM_PER_NODE=
MEM_PER_CPU=
GPUS=
CUDA_VISIBLE_DEVICES=0
```

## Step1:

```bash
u3684139@gpu2gate1:~/7606C/dp-manip$ .venv/bin/python scripts/inspect_dataset.py --data-root "$DATA_ROOT" --config configs/tasks/peginsertionside.toml
PASS PegInsertionSide-v1: train 400 demos/60558 steps; val 50; RGB (128, 128, 6) ['base_camera', 'hand_camera']; proprio 25; action 8 (pd_joint_pos); action range [-2.641, 2.882]
```

## Step 2: GPU Smoke test

```bash
u3684139@gpu-4080-403:~/7606C/dp-manip$ srun --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
srun: error: Unable to create step for job 135448: More processors requested than permitted

# 4 CPU is the maximum.
u3684139@gpu-4080-403:~/7606C/dp-manip$ srun --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=02:00:00 --pty bash
```

### Verify CUDA

```bash
u3684139@gpu-4080-403:~/7606C/dp-manip$ .venv/bin/python scripts/verify_cuda.py
nvidia-smi --query-gpu=name,memory.total --format=csv
torch 2.14.0+cu130
cuda available: True
cuda:0: NVIDIA GeForce RTX 4080 SUPER, 15.6 GiB
name, memory.total [MiB]
NVIDIA GeForce RTX 4080 SUPER, 16376 MiB
```

### 3 Backbones

```bash
u3684139@gpu-4080-403:~/7606C/dp-manip$ mkdir -p "$SMOKE_ROOT"
for b in unet transformer mlp; do
  .venv/bin/python scripts/run_experiment.py \
    --task peginsertionside --experiment backbone --value $b --seed 1 \
    --data-root "$DATA_ROOT" --output-root "$SMOKE_ROOT" --exp smoke_$b \
    --set train.total_iters=200 --set "train.validation_steps=[200]" --set "train.checkpoint_steps=[200]" \
    2>&1 | tee "$SMOKE_ROOT/smoke_$b.log"
done
backbone=unet seed=1 backbone=unet num_demos=100 device=cuda
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:431: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:437: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
number of parameters: 73.28M
smoke_unet: 100 train demos / 14934 windows; 50 validation demos; 84.5M params; cuda
[000001/200] loss=1.19189 lr=4.00e-07 elapsed=5s
[000100/200] loss=0.63714 lr=2.02e-05 elapsed=18s
[000200/200] loss=0.16073 lr=4.02e-05 elapsed=31s
[000200] fixed validation denoising loss=0.20123
finished smoke_unet in 0.03 h; checkpoint: /userhome/cs5/u3684139/dp-smoke/smoke_unet/checkpoints/final.pt
backbone=transformer seed=1 backbone=transformer num_demos=100 device=cuda
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:431: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:437: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
smoke_transformer: 100 train demos / 14934 windows; 50 validation demos; 20.3M params; cuda
[000001/200] loss=1.12186 lr=4.00e-07 elapsed=3s
[000100/200] loss=0.46157 lr=2.02e-05 elapsed=16s
[000200/200] loss=0.28416 lr=4.02e-05 elapsed=29s
[000200] fixed validation denoising loss=0.21883
finished smoke_transformer in 0.02 h; checkpoint: /userhome/cs5/u3684139/dp-smoke/smoke_transformer/checkpoints/final.pt
backbone=mlp seed=1 backbone=mlp num_demos=100 device=cuda
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:431: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:437: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
smoke_mlp: 100 train demos / 14934 windows; 50 validation demos; 11.7M params; cuda
[000001/200] loss=1.37757 lr=4.00e-07 elapsed=3s
[000100/200] loss=1.04440 lr=2.02e-05 elapsed=15s
[000200/200] loss=0.98075 lr=4.02e-05 elapsed=27s
[000200] fixed validation denoising loss=0.98871
finished smoke_mlp in 0.02 h; checkpoint: /userhome/cs5/u3684139/dp-smoke/smoke_mlp/checkpoints/final.pt
```

### Peak GPU & Total Time

```bash
u3684139@gpu-4080-403:~/7606C/dp-manip$ for b in unet transformer mlp; do
  .venv/bin/python -c "import json,sys; s=json.load(open(sys.argv[1])); print(sys.argv[2], 'loss', round(s['final_train_loss'],4), 'peak_alloc_MB', round(s['peak_gpu_mem_allocated_mb']), 'peak_reserved_MB', round(s['peak_gpu_mem_reserved_mb']), 'wall_s', round(s['wall_time_s_this_invocation']))" \
    "$SMOKE_ROOT/smoke_$b/summary.json" $b
done
unet loss 0.1607 peak_alloc_MB 3678 peak_reserved_MB 4498 wall_s 92
transformer loss 0.2842 peak_alloc_MB 2518 peak_reserved_MB 3116 wall_s 86
mlp loss 0.9807 peak_alloc_MB 2299 peak_reserved_MB 2902 wall_s 84
```

## Step 3: Closed-loop Evaluation

```bash
u3684139@gpu-4080-403:~/7606C/dp-manip$ for b in unet transformer mlp; do
  .venv/bin/python scripts/eval_dp.py "$SMOKE_ROOT/smoke_$b/checkpoints/final.pt" \
    --split val --episodes 2 --num-envs 2 2>&1 | tee "$SMOKE_ROOT/eval_$b.log"
done
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/sapien/_vulkan_tricks.py:42: UserWarning: Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.
  warn("Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.")
number of parameters: 73.28M
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/sapien/_vulkan_tricks.py:42: UserWarning: Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.
  warn("Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.")
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/sapien/_vulkan_tricks.py:42: UserWarning: Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.
  warn("Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.")
val: success_once=0.000 success_at_end=0.000 (2 episodes); /userhome/cs5/u3684139/dp-smoke/smoke_unet/eval/val_final.json
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/sapien/_vulkan_tricks.py:42: UserWarning: Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.
  warn("Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.")
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/sapien/_vulkan_tricks.py:42: UserWarning: Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.
  warn("Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.")
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/sapien/_vulkan_tricks.py:42: UserWarning: Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.
  warn("Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.")
val: success_once=0.000 success_at_end=0.000 (2 episodes); /userhome/cs5/u3684139/dp-smoke/smoke_transformer/eval/val_final.json
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/sapien/_vulkan_tricks.py:42: UserWarning: Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.
  warn("Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.")
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/sapien/_vulkan_tricks.py:42: UserWarning: Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.
  warn("Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.")
/userhome/cs5/u3684139/7606C/dp-manip/.venv/lib/python3.11/site-packages/sapien/_vulkan_tricks.py:42: UserWarning: Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.
  warn("Failed to find system libvulkan. Fallback to SAPIEN builtin libvulkan.")
val: success_once=0.000 success_at_end=0.000 (2 episodes); /userhome/cs5/u3684139/dp-smoke/smoke_mlp/eval/val_final.json
```
