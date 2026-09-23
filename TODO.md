# TODO（未完成事项）

**更新时间：9.23** ｜ 当前状态见 [STATUS.md](./STATUS.md)

> 六个任务：PickCube、PushCube、PullCube、StackCube、LiftPegUpright、PegInsertionSide（PickCube 已完成环境与 10 条示范验收，并已重放出 state 数据）。
>
> 本文件由 Mac 原 `TODO.md` 与 wsl 原 `TODO.md` 合并（9.23）。训练设备问题已经解决：训练与评估放在 wsl（RTX 4090），ubuntu 的 GTX 1080 Ti 不再用于训练。

---

## A. 多设备工作流（进行中，见 [PLAN.md](./PLAN.md)）

- [x] 阶段 0：ubuntu / wsl 打包备份 `~/dp-manip-pre-git-20260923.tgz`。
- [x] 阶段 1：Mac 建仓，写 `.gitignore`，添加 `ubuntu` / `wsl` 两个 remote。
- [x] 阶段 2：导入代码与文档，数据 rsync 到 Mac，生成 `manifests/*.sha256` 并校验，合并中文文档，首次提交。
- [ ] 阶段 3：挂接 ubuntu / wsl（需用户授权）。
- [ ] 阶段 4：编写 `scripts/sync.sh`（`push` / `fetch` / `pull-results` / `data` / `status`）。
- [ ] 阶段 5：往返验证。

## B. DP 接入准备（P1，尚未开始）

- [ ] 在 `pd_joint_pos` 8 维与 `pd_ee_delta_pos` 4 维之间正式选定控制模式。官方 IL 示例采用后者，但目前只完成了比较分析；若选后者，先转换一小批并验证成功率，再规划大规模转换。
- [ ] 明确训练用的 42 维扁平 state 向量定义、动作表示和缩放方式；不要仅凭文件名推断各维语义。训练数据与评估环境必须使用同一控制模式。
- [ ] 确定观测历史长度、动作 horizon、padding 与归一化规则，记录在 `configs/`；训练和评估共用同一份定义。
- [ ] 编写 ManiSkill H5/JSON → 训练样本的 dataset adapter；以 H5 `traj_*` 组和 JSON ID 划分 episode，不依据可能提前变真的逐步 `terminated`/`truncated`。
- [ ] 接入所需的 Diffusion Policy 模型与采样组件，补齐与训练控制模式一致的 ManiSkill 评估环境（wsl 尚未安装 ManiSkill）；只按确定的需求新增依赖，不照搬上游旧环境。
- [ ] 在小样本上做加载、单批前向/反向和评估接口检查，然后用少量 PickCube 数据跑通 **「专家轨迹 → (observation, action) 数据集 → DP 训练 → 策略评估」** 完整链路。

## C. 完成六任务专家数据集

- [ ] 对剩余五个任务各生成 1 条专家轨迹，分别确认规划成功和动作回放成功。
- [ ] 检查其他任务是否触发尚未适配的 MPlib API；如需修改，更新项目补丁并重新验证。
- [ ] 对剩余五个任务各采集 10 条成功轨迹，检查随机种子、动作有效性和逐条回放结果；按选定控制模式做 state 重放。
- [ ] 根据首轮训练结果确定正式数据量；当前规划起点为每任务约 200 条成功示范，并非课程硬性要求。
- [ ] 固定训练/验证数据划分及独立的评估随机种子，记录各任务数据量、轨迹长度分布和质量检查结果；新数据同步更新 `manifests/`。

## D. DP 基线、研究实验与交付

- [ ] 为六个任务分别训练、评估 DP baseline，记录训练配置、checkpoint、日志与计算资源。
- [ ] 在未见过的初始条件上评估，报告各任务成功率、评估次数和失败案例。
- [ ] 选定并完成一项受控研究实验（数据量、数据质量、超参数或架构）；可以考虑使用 25 / 50 / 100 / 200 条示范构造嵌套子集，研究数据效率。
- [ ] 整理六任务的专家生成过程和策略 rollout 视频。
- [ ] 完成报告、演示、代码与数据、训练 checkpoint、复现说明及贡献声明。

## E. 项目收尾与可复现性

- [ ] 检查 `~/Coding/dp-manip-old-backup` 中是否仍有独有的 Git 历史、未提交改动、脚本或轨迹。
- [ ] 检查 `~/Coding/dp-manip-clean-venv-backup` 是否还有需要保留的信息。
- [ ] 将干净安装、版本覆盖、补丁应用、Vulkan 环境变量和专家采集命令写入复现文档。
- [ ] 依赖拆分：`pyproject.toml` 改为 `train` / `expert` 依赖组并重新 lock（会改动 wsl 环境，需单独验证）。
- [ ] 确认备份内容已妥善保留后，再删除旧目录、临时虚拟环境、wsl 的 `.git.bak` 与 `dp-manip-pre-git-*.tgz`；删除前须用户确认。
