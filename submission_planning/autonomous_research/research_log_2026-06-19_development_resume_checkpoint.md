# 本轮开发日志：从消融训练调试到可恢复断点

日期：2026-06-19  
范围：`submission_planning/`、`tmp/ablation_results/`、`tmp/research_package_audits/`、`tmp/manuscript_audits/`。  
边界：本轮只沉淀研究规划、审计脚本、受保护实验 runner 和临时实验产物；未改写 `src/`，未覆盖原始论文包、PPT 包、图像资产或既有结果目录。

## 1. 本轮研究主线

本轮工作的核心目标是把投稿前的计划推进到可审计的实验执行链路。重点从外部 SOTA 适配规划转向核心消融的受保护执行入口，逐步确认数据接口、特征通道、模型前向、loss、optimizer、checkpoint 与 debug metrics 的闭环是否可运行。

该阶段仍属于实验基础设施与 debug 阶段，不能支撑论文中的模块贡献结论。

## 2. 已完成内容

### 2.1 SOTA 与投稿缺口整理

已形成外部 SOTA 候选、复现优先级和投稿缺口闭环计划，核心文件包括：

```text
submission_planning/sota_comparison_chinese.md
submission_planning/autonomous_research/external_baseline_feasibility.md
submission_planning/autonomous_research/external_sota_next_decision_log.md
submission_planning/autonomous_research/submission_gap_closure_plan.md
```

关键更新是将 DFV、DDFFNet 设为最优先外部基线，并将 Depth Anything V2 归入 auxiliary-only 方向，用于训练策略和 foundation-depth 讨论，暂不进入 focus-stack 主数值表。

### 2.2 外部基线工具链预备

已完成 P10 单样本导出、外部预测评估器、batch evaluator、DFV/DDFFNet workspace scaffold 和 eligibility audit gate。当前阶段只证明外部基线适配链路具备前置条件，尚无 DFV 或 DDFFNet 的真实预测结果。

关键文件包括：

```text
submission_planning/tools/preflight_external_baseline_data.py
submission_planning/tools/export_one_external_baseline_sample.py
submission_planning/tools/evaluate_external_prediction.py
submission_planning/tools/evaluate_external_prediction_batch.py
submission_planning/tools/audit_external_sota_eligibility.py
submission_planning/tools/preflight_dfv_environment.py
```

### 2.3 消融实验设计与入口核验

已建立 ABL-00 到 ABL-06 的消融 run matrix，并完成特征 schema 审计、ABL-03 focal-difference 实现审计、mask smoke test、training-entry preflight、minimal runner smoke 和 training-runner preflight。

当前确定的核心消融路径为：

| Run | 作用 | 当前状态 |
|---|---|---|
| ABL-00 | Full S2R-FocusNet | 可进入受保护 runner |
| ABL-01 | Direct image-to-depth / focus-stack-only lower-prior | 需要单独 lower-prior 架构 |
| ABL-02 | w/o DFF/GADFF prior | 可进入受保护 runner |
| ABL-03 | w/o focal difference | 可进入受保护 runner |
| ABL-04 | w/o glare cue | 可进入受保护 runner |

ABL-01 已单独记录为架构决策项，后续不能直接放进 38-channel upgraded runner。

### 2.4 ABL-00/03 dry-run 与小训练 debug

已在 `submission_planning/tools/run_ablation_variant_training.py` 中完成默认 dry-run 和显式小训练模式。默认模式只检查数据、mask、模型和 loss 接口；只有传入 `--execute-training` 才会启动 optimizer/backward/update。

已执行并通过的小训练 debug 命令：

```text
python -X utf8 submission_planning/tools/run_ablation_variant_training.py --execute-training --run-id ABL-00 --run-id ABL-03 --max-epochs 1 --train-patches 8 --val-patches 4 --batch-size 1
```

小训练 debug 总览：

```text
tmp/ablation_results/training_runner_small_train/ablation_training_runner_small_train_summary.md
tmp/ablation_results/training_runner_small_train/ablation_training_runner_small_train_summary.json
```

关键结果如下：

| Run | Train loss debug | Val loss debug | Val MAE norm debug | 解释 |
|---|---:|---:|---:|---|
| ABL-00 | 0.42624453 | 0.26109813 | 0.18825971 | runner 可训练 |
| ABL-03 | 0.36811332 | 0.39810403 | 0.27409419 | zero focal-difference 路径可训练 |

这些数值只来自 1 epoch、P10 patch-sampling debug 设置。可证明训练链路可运行，不能进入论文主表。

## 3. 本轮对研究计划的纠正

### 3.1 先修正实验入口，再追求论文指标

原计划中外部 SOTA 和消融实验容易被理解为可以直接运行主训练脚本。实际检查后确认，原训练脚本会写入项目交付包和既有结果目录，因此投稿补实验必须使用受保护 runner，并将产物限制到 `tmp/ablation_results/`。

### 3.2 ABL-01 与 ABL-02/03/04 属于不同实现层级

ABL-02/03/04 可以在 38-channel upgraded feature space 中通过 channel masking 实现。ABL-01 去掉 DFF/GADFF 与所有 engineered priors 后，输入空间变成 focus-stack-only，需要单独 lower-prior 架构，不能用同一 masking 逻辑替代。

### 3.3 Debug 指标必须保持 claim gate 关闭

当前 ABL-00/03 的 debug checkpoint 和 metrics 已写入临时目录，但 `run_config.json` 仍保持：

```text
claim_eligible = false
main_table_eligible = false
```

这条边界应继续保留，直到完整 split、per-sample metrics、seed 记录和 eligibility audit 全部完成。

### 3.4 下一阶段应改为 controlled pilot

当前小训练只覆盖 ABL-00 和 ABL-03。下一步的合理实验单元应扩展到 ABL-00/02/03/04，用相同 patch sampling、epoch、batch size 和学习率做 2-3 epoch controlled pilot，并新增独立 pilot tag，避免覆盖已有 `small_training_debug` 产物。

## 4. 当前可支持的研究表述

当前可以安全表述：

1. 已完成投稿研究包、外部基线适配和核心消融的可审计执行规划。
2. 已确认 ABL-00/02/03/04 可共享 upgraded 38-channel Focus-ResUNet runner 入口。
3. 已确认 ABL-03 的 focal-difference 通道为 17-32，可通过 zero mask 构造 w/o focal-difference 变体。
4. 已确认 ABL-00/03 能完成最小 optimizer/backward/update 链路，并将 checkpoint 与 debug metrics 写入 `tmp/ablation_results/`。

当前不能安全表述：

1. ABL-00 已经在正式消融中优于 ABL-03。
2. focal-difference 的贡献已经被充分证明。
3. DFV 或 DDFFNet 已经完成公平数值对比。
4. 真实样本已经具备 calibrated height ground truth。

## 5. 当前最高优先级

| 优先级 | 任务 | 目标 |
|---:|---|---|
| 1 | 修改 `run_ablation_variant_training.py` 的输出标签机制 | 支持 controlled pilot 独立产物，避免覆盖 debug 文件 |
| 2 | 运行 ABL-00/02/03/04 controlled pilot | 获得同设置下的 pilot metrics |
| 3 | 新增 ABL pilot eligibility audit | 防止 pilot debug 数值误入论文主表 |
| 4 | 推进 DFV repo inventory | 明确外部 SOTA 复现入口、依赖和输出 contract |
| 5 | 更新 manuscript evidence table | 将新证据与 claim boundary 同步 |

## 6. 本轮结论

本轮已经把项目从“投稿计划与 SOTA 调研”推进到“可恢复、可审计、受保护的小规模实验执行链路”。最重要的断点是：ABL-00/03 小训练 debug 已完成，下一步应先增加 pilot tag/run-kind，再运行 ABL-00/02/03/04 controlled pilot。
