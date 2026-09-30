# 下一步

1. 在集群登录节点运行 `./setup.sh`。
2. 将 `maniskill-demogen/data/dataset` 放到共享存储，运行六任务 `inspect_dataset.py`。
3. Phase 0 CPU 最小闭环已完成；仍需在兼容 GPU 上做 PickCube batch 64 短 smoke并记录峰值显存，
   对 `unet` / `transformer` / `mlp` 三个 `policy.backbone` arm 各跑一次，
   并走通对应 checkpoint 的闭环 evaluate；OOM 时全实验统一降 batch。
4. 在集群验证 `USR1 → resume.pt → requeue` 一次：确认日志出现
   `restored Python/NumPy/torch CPU/CUDA RNG state`，并跑通
   `tests/test_rng_resume.py`（含 CUDA RNG）。
5. 用 `scripts/check_experiment.py --experiment data_size` 和 `--experiment backbone`
   做一次 Gate B drift 检查，再提交核心 96 组训练和固定测试评估。
6. 汇总成功率前，对每个实验加 `--run-root "$RUN_ROOT"` 再跑一次 `check_experiment.py`，
   确认实际训练用的配置没有 drift；再按预注册规则判断哪些任务增加 N=400。
7. **数据加载 I/O：`data.preload` 已设为默认（phase 21）。** 集群 bench（job 135722，结果见
   `4080_test_results.md` Step 4）显示 lazy HDF5 读逐窗口解 gzip 把 CPU 打满（85%）、GPU 只有
   51%；preload 后 GPU 88%、吞吐 2.34×，train_loss 逐位一致。待办：在集群上用默认配置跑一次
   N=400 的 smoke，确认两个 trainer 并行时（`train_dual_gpu.sbatch`，64G）内存余量足够。

不要在拿到结果后更改 N 档、训练 seed 数、测试种子或 100k 步预算。
