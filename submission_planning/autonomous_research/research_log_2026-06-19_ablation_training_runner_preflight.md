# 本轮自主研究日志：Ablation Training-Runner Preflight

日期：2026-06-19  
范围：`submission_planning/` 与 `tmp/ablation_results/`。  
边界：本轮未修改 `src/`，未启动训练循环，未创建 optimizer，未保存 checkpoint、预测数组、图像或新指标结果。

## 1. 本轮研究目标

上一轮已经完成 minimal ablation runner smoke，证明 ABL-00/02/03/04 可以共享 upgraded 38-channel Focus-ResUNet 输入空间，ABL-01 需要单独 lower-prior 设计。本轮目标是进一步检查未来 training runner 的安全边界：哪些原脚本部件可以复用，哪些原脚本入口必须避开，所有产物应写到哪里。

该目标直接服务于投稿闭环中的核心消融任务。当前仍缺少实测消融指标，但如果训练 runner 直接调用原实验脚本主入口，可能会把模型、图像和报告写回原项目交付包，违背“不污染当前项目资源”的边界。

## 2. 采用的方法

本轮读取并检查 `src/train_focus_resunet_loss_experiment.py` 的训练和输出函数。代码显示，该脚本的 `main()` 会创建 `MODEL_DIR`，调用 `train_model()`，保存 `focus_resunet_hybrid_loss.pt`，再调用 `evaluate_split()`、`write_metric_plots()` 和 `write_report()`。

进一步检查发现，`OUT` 与 `MODEL_DIR` 指向 `src/结题交付包/05_图表与结果/模型与损失函数升级实验/`。这说明原脚本适合生成项目交付包结果，但不适合直接作为 ABL 训练 runner 主入口。

因此，本轮采用 preflight 方式：只验证 future runner 的计划输出路径、variant 训练状态、ABL-01 gate、既有 runner smoke 依赖和 claim_eligible 状态，不启动训练。

## 3. 已完成的任务

已新增 `submission_planning/tools/preflight_ablation_training_runner.py`。该工具检查最终方法训练脚本的输出风险、ABL-00/01/02/03/04 的 planned output、training gate 和 minimal runner smoke 依赖。

已生成 `tmp/ablation_results/training_runner_preflight/ablation_training_runner_preflight.json` 和 `.md`。当前结果为 `pass`，共 63 项检查，0 个 error，0 个 warning。

已新增 `submission_planning/autonomous_research/ablation_training_runner_preflight.md`，作为研究包中的正式说明节点。该文档明确 future runner 不应直接调用 `main()`、`evaluate_split()`、`write_metric_plots()` 和 `write_report()`。

已新增 `tmp/ablation_results/ABL-01/logs/2026-06-19_lower_prior_architecture_decision.md`，将 ABL-01 记录为 17-channel focus-stack-only lower-prior baseline 的设计任务，而非普通 channel-mask 变体。

## 4. 关键判断

ABL-00、ABL-02、ABL-03 和 ABL-04 可以在 future training runner 实现后进入训练准备阶段。它们共享 `focus_resunet_upgraded` 或 `focus_resunet_upgraded_masked` 路径，差异由 38-channel feature 中的 zero channels 控制。

ABL-01 暂时不能训练。它需要先确定 lower-prior architecture。当前建议优先设计 17-channel focus-stack-only U-Net，以更清楚地检验物理先验和 focal-difference 表征的贡献；TinyDepthNet 更适合作为内部 baseline 保留。

原 final-method 脚本中的模型、损失、特征增强和指标函数可以复用；原脚本中负责写交付包输出的函数不应直接复用。future runner 应把日志、checkpoint、预测、图像和 metrics 全部写入 `tmp/ablation_results/<run_id>/`。

## 5. 对研究计划的更新

上一轮计划是从 runner smoke 进入 training runner implementation。本轮将该计划进一步细化：training runner 的实现应先支持 ABL-00 与 ABL-03 的小规模 dry run，因为它们分别对应 full model 和 focal-difference contribution，是最关键的对照。

ABL-02 和 ABL-04 可以随后接入同一 runner。ABL-01 需要在 lower-prior 架构确定后再接入训练流程。

当前仍不应启动完整训练。下一步更合适的是实现一个带 `--dry-run`、`--run-id`、`--max-epochs`、`--train-patches`、`--val-patches` 参数的 runner，并先让 dry run 写入日志和空 metrics 模板。

## 6. 已同步更新的研究包内容

已更新 `research_package_integrity_audit.md` 和 `audit_research_package_integrity.py`，让完整性审计检查 training-runner preflight 与 ABL-01 lower-prior 决策日志。

已更新 `research_task_board.md`，新增 D45 和 D46，并将下一步 R28 更新为 ABL-00/03 small-scale training dry run。

已更新 `research_index.md`、`experiment_roadmap.md`、`ablation_execution_protocol.md` 和 `submission_gap_closure_plan.md`，统一记录 training-runner preflight 已完成，训练仍未开始。

## 7. 当前证据边界

本轮可以支持的表述是：future ABL training runner 的输出边界已经明确，原脚本的交付包输出风险已经识别，ABL-01 的 lower-prior 架构 gate 已记录。

本轮不能支持的表述包括：ABL-00/03 已训练、任何消融指标已经生成、checkpoint 已存在、模块贡献已被验证、ABL-01 已具备公平对照结果。

## 8. 下一轮建议

下一轮若继续消融线，应实现 `run_ablation_variant_training.py`，先做 ABL-00 与 ABL-03 的 dry run。dry run 应只写入 `tmp/ablation_results/<run_id>/logs/`，并保持 `claim_eligible=false`。

下一轮若继续 SOTA 线，应推进 DFV repository inventory 和 P10 prediction contract。外部 deep DFF 对比与核心消融仍是当前投稿缺口中最高优先级的两条线。
