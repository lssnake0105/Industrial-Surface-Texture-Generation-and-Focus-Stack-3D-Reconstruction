# 本轮自主研究日志：Minimal Ablation Runner Smoke

日期：2026-06-19  
范围：`submission_planning/` 与 `tmp/ablation_results/runner_smoke/`。  
边界：本轮未修改 `src/`，未训练模型，未保存 checkpoint，未保存预测数组或实验图像。

## 1. 本轮研究目标

上一轮已经确认 final-method 消融应以 `src/train_focus_resunet_loss_experiment.py` 的 upgraded 38-channel Focus-ResUNet 路径为主。本轮目标是把这个结论推进到可执行接口层面：验证 ABL-00/02/03/04 是否能够共享最终模型输入空间，并明确 ABL-01 是否需要单独设计。

该目标服务于投稿闭环中的核心消融缺口。当前论文仍缺少模块贡献的实测证据，但在训练前必须先确认 runner 接口、通道 mask 和损失计算不会偏离最终模型路径。

## 2. 采用的方法

本轮新增了无训练 smoke runner。它读取 P10 synthetic scenario，构造 base 22-channel features，再通过 `augment_features()` 生成 upgraded 38-channel features。随后截取 64 x 64 center patch，对每个核心 ABL 变体应用修正后的通道策略。

ABL-00 保留完整 38 通道。ABL-02 在 upgraded feature 中 zero channels 34-37，对应 DFF/GADFF depth 和 confidence prior。ABL-03 zero channels 17-32，对应 16 个 focal-difference input channels。ABL-04 zero channel 33，对应 explicit risk / glare cue。ABL-01 只记录 raw 17-channel focus-stack 设计，不用 final Focus-ResUNet 做伪前向。

为了避免训练，本轮只使用随机初始化的 `FocusResUNet`，设置为 `eval()`，并在 `torch.no_grad()` 下计算前向输出和有限 loss。该 loss 只用于证明接口可运行，不能解释为性能。

## 3. 已完成的任务

已新增 `submission_planning/tools/run_ablation_variant_smoke.py`。该工具只做 P10 patch 级别的 shape/config smoke，不创建 optimizer，不执行反向传播，不写入 checkpoint、prediction 或 figure。

已生成 `tmp/ablation_results/runner_smoke/minimal_ablation_runner_smoke.json` 和 `.md`。当前结果为 `pass`，共 35 项检查，0 个 error，0 个 warning。

已新增 `submission_planning/autonomous_research/minimal_ablation_runner_smoke.md`，把临时 smoke 报告整理成研究包中的正式说明节点。该文档明确 runner smoke 的边界、通道定义、variant 决策和下一步训练 runner 要求。

## 4. 关键发现

ABL-00、ABL-02、ABL-03 和 ABL-04 均可在 upgraded 38-channel Focus-ResUNet 输入空间中完成前向形状检查，输入输出形状均为 `[1, 38, 64, 64] -> [1, 1, 64, 64]`。

ABL-01 不适合通过简单 mask final model 来代表 direct image-to-depth。它需要单独 lower-prior runner 或架构说明，否则会把最终模型结构、残差约束和 hybrid loss 的影响混入“无先验”对照中。

ABL-04 当前仅移除 explicit risk channel。若论文希望更严格地证明 glare-aware 信息贡献，还需要进一步决定是否同时剥离 GADFF 或 confidence 中隐含的 glare-related signal。

## 5. 对研究计划的更新

上一轮计划把下一步定义为 minimal ablation runner design。本轮已经完成 shape/config smoke，因此计划应更新为 training runner implementation。

下一阶段需要实现真正训练 runner，但仍应保持输出隔离在 `tmp/ablation_results/<run_id>/`。每个 run 至少需要写入 seed、split、feature switch、source entry、local git status、synthetic metrics、real no-reference metrics、failure notes 和 claim eligibility。

ABL-01 被提升为单独的架构设计任务。它不能继续作为普通 channel mask 项处理，应先写出 lower-prior 模型或 runner 的结构选择，再进入训练。

## 6. 已同步更新的研究包内容

已更新 `research_package_integrity_audit.md` 与 `audit_research_package_integrity.py`，使完整性审计能够检查 `minimal_ablation_runner_smoke.md`、`run_ablation_variant_smoke.py` 和 `tmp/ablation_results/runner_smoke/` 下的 pass 报告。

已更新 `research_task_board.md`，新增 D44，并把 R27 从 minimal runner design 更新为 ablation training runner implementation，同时新增 R28 用于 ABL-01 lower-prior architecture decision。

已更新 `research_index.md`、`experiment_roadmap.md`、`ablation_execution_protocol.md` 和 `submission_gap_closure_plan.md`，统一记录 runner smoke 已完成、训练仍未开始、模块贡献仍未验证。

## 7. 当前证据边界

本轮可以支持的表述是：ABL-00/02/03/04 的 final-method runner 接口和 channel-control 方案已经通过无训练 shape smoke；ABL-01 已被识别为需要单独 lower-prior runner。

本轮不能支持的表述包括：w/o prior、w/o focal difference 或 w/o glare cue 已经降低或提高 MAE；某个模块已被证明有效；smoke diagnostic loss 可以进入论文表格；ABL-01 已经具备公平训练结果。

## 8. 下一轮建议

若继续消融线，优先实现 `run_ablation_variant_training.py` 或等效训练脚本，先支持 ABL-00 和 ABL-03 的小规模训练/日志写入，再扩展到 ABL-02 和 ABL-04。ABL-01 需要先完成 lower-prior architecture decision。

若继续 SOTA 线，优先推进 DFV repository inventory 和 P10 prediction contract。当前投稿风险最高的两项仍是外部 deep DFF 实测对比和核心消融实测结果。
