# 投稿实验路线图

更新日期：2026-06-18  
目标：在不改动现有项目资源的前提下，规划下一阶段可执行实验，用于把当前项目推进到可投稿形态。

## 1. 当前最小论文闭环

| 模块 | 当前是否具备 | 还缺什么 |
|---|---|---|
| 问题定义 | 已具备 | 需要统一术语为 reflective surface defect morphology reconstruction |
| 合成数据与 GT | 已具备 | 需要更清楚写明仿真因素与 domain randomization |
| 内部 baseline | 已具备 | 需要压缩模型叙事，突出一个 final method |
| 外部 SOTA | 不足 | 需要 DFV + DDFFNet，至少补一个 2022 以后 deep DFF 方法 |
| 消融实验 | matched full candidate 与 longer-budget repeat 已完成 | DFF/GADFF prior 贡献清楚；longer repeat 后 full model 仍未占优，下一步应复核辅助信号质量和融合机制 |
| 真实样本验证 | 已具备初版 | 需要坚持 no-reference 评估边界，增加 profile / spike map 可视化 |
| Related Work | 初版具备 | 需要补 2025-2026 方法和 Depth Anything V2 的训练策略意义 |

## 2. 两周优先任务

### Week 1: 外部基线与数据接口

| Day | 任务 | 产出 |
|---|---|---|
| D1 | 固定 synthetic split、统计样本尺寸和焦堆帧数 | `dataset_interface_notes.md` |
| D2-D3 | 调研 DFV 官方输入格式并写转换计划 | DFV adapter 设计说明 |
| D4-D5 | 跑通 DFV 最小样例或确认阻碍 | DFV feasibility note |
| D6 | 调研 DDFFNet 输入格式和依赖 | DDFFNet adapter 设计说明 |
| D7 | 整理外部 baseline 风险表 | 更新 `external_baseline_feasibility.md` |

### Week 2: 消融与论文整合

| Day | 任务 | 产出 |
|---|---|---|
| D8-D9 | 基于 longer repeat 结果做 gated fusion 设计或 glare cue quality audit | fusion design note or cue audit |
| D10-D11 | 整理真实样本 no-reference 可视化：profile curve, spike map, confidence map | figure plan |
| D12 | 把 Depth Anything V2 写入 Related Work 与 Discussion | manuscript paragraph |
| D13 | 更新实验表结构和公平比较原则 | experiment section notes |
| D14 | 完成投稿前 gap audit | submission readiness checklist |

## 3. 推荐主结果表

### Table A: Synthetic Quantitative Comparison

| Method | Type | Training Setting | Mean MAE | Edge MAE | High-Risk MAE | Scale Alignment | Notes |
|---|---|---|---:|---:|---:|---|---|
| Original DFF | classical | no training | 待填 | 待填 | 待填 | no | current baseline |
| DFF + post | classical + post | no training | 待填 | 待填 | 待填 | no | current baseline |
| Lee2013 | classical SFF | no training | 待填 | 待填 | 待填 | no | current baseline |
| Li2019 | classical SFF | no training | 待填 | 待填 | 待填 | no | current baseline |
| DDFFNet | deep DFF | synthetic retrain / zero-shot | 待填 | 待填 | 待填 | yes/no | external baseline |
| DFV | deep DFF | synthetic retrain / zero-shot | 待填 | 待填 | 待填 | yes/no | most important external baseline |
| Proposed S2R-FocusNet | prior-guided deep correction | synthetic train | 53.22 | 86.68 | 40.14 | no | current best overall MAE |

### Table B: Ablation Study

| Variant | DFF Prior | Focal Difference | Glare Prior | Domain Randomization | Mean MAE | Edge MAE | High-Risk MAE |
|---|---|---|---|---|---:|---:|---:|
| Full S2R-FocusNet | yes | yes | yes | yes | 待填 | 待填 | 待填 |
| w/o DFF prior | no | yes | yes | yes | 待填 | 待填 | 待填 |
| w/o focal difference | yes | no | yes | yes | 待填 | 待填 | 待填 |
| w/o glare prior | yes | yes | no | yes | 待填 | 待填 | 待填 |
| w/o domain randomization | yes | yes | yes | no | 待填 | 待填 | 待填 |
| image-to-depth direct | no | no | no | yes | 待填 | 待填 | 待填 |

### Table C: Real No-Reference Morphology Evaluation

| Method | Roughness Stability | Spike Count | Relative Dynamic Range | Edge Retention | Visual Failure Mode |
|---|---:|---:|---:|---:|---|
| Original DFF | 0.0977 | 9179.7 | 0.6964 | -0.0394 | many unstable spikes |
| DFF + post | 0.0130 | 4091.3 | 0.3944 | -0.0296 | smoothed but still spiky |
| Lee2013 | 0.0295 | 5018.7 | 0.6325 | -0.0209 | adaptive window artifacts |
| Li2019 | 0.0318 | 5433.6 | 0.6515 | -0.0233 | residual unstable peaks |
| Proposed S2R-FocusNet | 0.0078 | 2.0 | 0.5317 | 0.0009 | most stable among current methods |

## 4. Figure 计划

| Figure | 内容 | 目的 |
|---|---|---|
| Fig. 1 | Simulation-to-real pipeline | 展示合成 height map、rendered focus stack、DFF/GADFF priors、model training、real inference |
| Fig. 2 | S2R-FocusNet method diagram | 收束多个模型产物，突出一个 final method |
| Fig. 3 | Synthetic quantitative comparison | 展示 MAE / edge MAE / high-risk MAE |
| Fig. 4 | Difficult sample visual comparison | 用 P10 或最难样本展示边界和 high-risk 区域 |
| Fig. 5 | Real no-reference morphology | 展示真实样本高度图、3D surface、profile curve、spike map |
| Fig. 6 | Ablation visualization | 说明 prior、focal difference、glare prior 的贡献 |

## 5. 论文叙事更新点

### Introduction

1. 工业表面缺陷需要 3D morphology，而单纯 2D detection 难以描述缺陷深度、边界和形貌。
2. 真实微尺度表面高度 GT 难以标定，尤其在反光、弱纹理和局部眩光条件下。
3. Simulation-to-real 数据构造是本文的核心研究策略。
4. Prior-guided deep correction 将传统 DFF 的可解释性与神经网络的局部修正能力结合。

### Related Work

1. Classical SFF/DFF: Nayar, Pertuz, Lee2013, Li2019。
2. Deep DFF: DDFFNet, DFV, AiFDepthNet, DfF in the Wild, DDFS, HybridDepth。
3. Latest focus-stack trends: FAD, DualFocus, DDL-Recurrent SFF, Minimal Focal Stack。
4. Simulation-to-real / pseudo-label: Focus on Defocus, DEReD, Depth Anything V2。
5. Industrial morphology: surface defect detection review, structured illumination / microscopy 3D reconstruction。

### Discussion

1. 合成数据给出绝对误差，真实样本验证相对形貌稳定性。
2. 当前模型整体 MAE 和 edge MAE 较优，但 high-risk 区域仍需改进。
3. Depth Anything V2 说明 synthetic labels + pseudo-labeled real data 可能成为下一阶段训练策略。
4. 未来应加入 calibrated real height subset。

## 6. 审稿风险清单

| 风险 | 可能审稿意见 | 应对策略 |
|---|---|---|
| 外部 SOTA 不足 | “Comparison with recent deep DFF methods is missing.” | 补 DFV 和 DDFFNet，至少讨论 HybridDepth / DDFS / FAD |
| 真实样本无 GT | “Real-world quantitative validation is insufficient.” | 明确 no-reference 边界，补真实标定子集作为 future 或扩展实验 |
| 模型贡献分散 | “Too many model variants without clear final method.” | 收束到 S2R-FocusNet，其余作为 ablation |
| 仿真真实性 | “Synthetic data may not reflect real surfaces.” | 写清 defect geometry、roughness、texture、reflectance、glare、focus response 和 domain randomization |
| 单目深度模型比较不公平 | “Depth foundation models are not directly comparable.” | 将 Depth Anything V2 放到 auxiliary qualitative / training strategy，主表只放 focus-stack methods |

## 7. 下一步文件产物

| 文件 | 作用 | 当前状态 |
|---|---|---|
| `dataset_interface_notes.md` | 记录当前数据与外部 SOTA 适配字段 | 已完成 |
| `related_work_draft_cn.md` | 中文 Related Work 草稿 | 已完成 |
| `ablation_design.md` | 消融实验设计与预期结论 | 已完成 |
| `submission_readiness_checklist.md` | 投稿前检查表 | 已完成 |
| `claim_evidence_matrix.md` | 论文主张与证据边界矩阵 | 已完成 |
| `baseline_reproduction_decision_table.md` | 外部基线复现执行和降级规则 | 已完成 |
| `manuscript_blueprint.md` | 英文投稿论文结构蓝图 | 已完成 |
| `research_task_board.md` | 后续自主研究任务板 | 已完成 |
| `abstract_draft_en.md` | 英文 Abstract 安全版草稿 | 已完成 |
| `introduction_outline_en.md` | 英文 Introduction 五段式大纲 | 已完成 |
| `baseline_adapter_spec.md` | 外部基线数据适配规格 | 已完成 |
| `figure_table_plan.md` | 图表编号、caption 与数据源计划 | 已完成 |
| `failure_analysis_plan.md` | 失败分析样本、指标与写作计划 | 已完成 |
| `method_outline_en.md` | 英文 Method 章节大纲 | 已完成 |
| `experiments_outline_en.md` | 英文 Experiments 章节大纲 | 已完成 |
| `discussion_outline_en.md` | 英文 Discussion 章节大纲 | 已完成 |
| `latex_manuscript_assembly_plan.md` | LaTeX 普通论文组稿计划 | 已完成 |
| `dfv_smoke_test_checklist.md` | DFV 单样本 smoke test 检查清单 | 已完成 |
| `baseline_run_log_template.md` | 外部基线运行日志模板 | 已完成 |
| `external_baseline_data_preflight.md` | 外部 baseline 数据预检与 P10 单样本导出说明 | 已完成 |
| `../tools/preflight_external_baseline_data.py` | 只读检查 synthetic split 与本地样本资产是否可直接进入 DFV/DDFFNet | 已完成 |
| `../tools/export_one_external_baseline_sample.py` | 生成 P10 test sample 的临时焦堆、GT、risk 和 prior 包 | 已完成 |
| `../tools/smoke_test_external_baseline_package.py` | 验证 P10 临时包可被 dataloader 读取为 focal-stack sample | 已完成 |
| `external_prediction_evaluator.md` | 外部预测结果评估器说明与 DFF/GADFF prior 自检 | 已完成 |
| `../tools/evaluate_external_prediction.py` | 对未来 DFV/DDFFNet `.npy` 预测输出计算 MAE/P90/edge/high-risk 指标 | 已完成 |
| `batch_external_evaluation.md` | 批量外部预测评估和 method mean 汇总说明 | 已完成 |
| `../tools/evaluate_external_prediction_batch.py` | 根据 prediction manifest 生成 per-sample 与 method summary 指标表 | 已完成 |
| `dfv_ddffnet_integration_protocol.md` | DFV/DDFFNet 外部仓库接入、预测导出、批量评估和入表门槛规程 | 已完成 |
| `external_sota_next_decision_log.md` | DFV 优先、DDFFNet 次优先、入表证据和降级规则的当前决策记录 | 已完成 |
| `external_sota_eligibility_audit.md` | 外部方法进入主结果表前的证据审计规则 | 已完成 |
| `../tools/audit_external_sota_eligibility.py` | 检查 prediction manifest、batch result、run log 和样本覆盖的只读审计脚本 | 已完成 |
| `external_baseline_workspace_scaffold.md` | 外部基线临时运行目录、日志、manifest 和输出结构说明 | 已完成 |
| `../tools/scaffold_external_baseline_workspace.py` | 为 DFV/DDFFNet 生成 `tmp/external_baseline_results/<method>/` 脚手架 | 已完成 |
| `research_package_integrity_audit.md` | 自主研究包完整性、claim safety 和污染边界审计说明 | 已完成 |
| `../tools/audit_research_package_integrity.py` | 自动检查核心文档、工具脚本、DFV/DDFFNet scaffold、P10 数据包和文本风险 | 已完成 |
| `manuscript_claim_safety_audit.md` | LaTeX 稿件外部 SOTA、真实绝对精度、消融和泛化 claim 风险审计说明 | 已完成 |
| `../tools/audit_manuscript_claim_safety.py` | 自动检查当前 LaTeX 草稿是否越过现有证据边界 | 已完成 |
| `ablation_execution_protocol.md` | 消融 run matrix、证据要求、入稿门槛和执行顺序 | 已完成 |
| `templates/ablation_run_matrix_template.csv` | ABL-00 至 ABL-06 的机器可读变体矩阵 | 已完成 |
| `../tools/scaffold_ablation_workspace.py` | 生成 `tmp/ablation_results/` 下的消融运行目录、日志和指标模板 | 已完成 |
| `ablation_feature_schema_audit.md` | 当前 22 通道特征 schema 与各消融通道 mask 说明 | 已完成 |
| `../tools/audit_ablation_feature_schema.py` | 重新生成 P10 features 并核验通道顺序、mask 和非通道型消融风险 | 已完成 |
| `abl03_focal_difference_implementation_audit.md` | ABL-03 焦向差分实际实现位置和 38 通道升级特征审计 | 已完成 |
| `../tools/audit_abl03_focal_difference_implementation.py` | 核验 `augment_features()` 中 17-32 通道等于 `np.diff(stack)` | 已完成 |
| `ablation_mask_smoke_test.md` | ABL-01/02/03/04 输入 mask 的执行前置检查说明 | 已完成 |
| `../tools/smoke_test_ablation_masks.py` | 核验目标通道被置零、非目标通道不变且不生成模型结果 | 已完成 |
| `depth_anything_v2_auxiliary_protocol.md` | Depth Anything V2 作为单目基础深度辅助参考的使用边界、证据 gate 和图示方案 | 已完成 |
| `../tools/scaffold_depth_anything_v2_auxiliary_workspace.py` | 生成 `tmp/foundation_depth_auxiliary/DepthAnythingV2/` 下的 scaffold-only 清单和日志 | 已完成 |
| `submission_gap_closure_plan.md` | 将投稿缺口绑定到最小闭环证据、入稿门槛和下一步任务 | 已完成 |
| `dfv_environment_preflight.md` | DFV 外部仓库下载/运行前的本地环境、P10 数据包和输出边界 preflight | 已完成 |
| `../tools/preflight_dfv_environment.py` | 生成 DFV preflight JSON/Markdown 和 inventory log，不下载仓库或运行模型 | 已完成 |
| `ablation_training_entry_preflight.md` | 核验 ABL-00/01/02/03/04 的训练入口、feature space、脚手架和既有 smoke/audit 证据 | 已完成 |
| `../tools/preflight_ablation_training_entry.py` | 生成消融训练入口 preflight JSON/Markdown，不训练模型、不生成 checkpoint | 已完成 |
| `minimal_ablation_runner_smoke.md` | 记录 ABL-00/02/03/04 的 38 通道 runner shape smoke 与 ABL-01 lower-prior 设计边界 | 已完成 |
| `../tools/run_ablation_variant_smoke.py` | 在不训练的条件下验证 corrected ABL runner 接口、通道 mask 和前向输出形状 | 已完成 |
| `ablation_training_runner_preflight.md` | 记录未来训练 runner 的安全输出边界、禁用原脚本输出函数和 ABL-01 gate | 已完成 |
| `../tools/preflight_ablation_training_runner.py` | 验证训练 runner 计划只写入 `tmp/ablation_results/`，且不启动训练 | 已完成 |
| `ablation_training_runner_dry_run.md` | 记录 ABL-00/03 默认 dry-run，确认 runner 接口可达且未产生训练产物 | 已完成 |
| `../tools/run_ablation_variant_training.py` | 默认 dry-run 的 future training runner 骨架，当前拒绝真实训练 | 已完成 |
| `ablation_small_training_debug.md` | 记录 ABL-00/03 的 1 epoch 最小训练调试，强调不作为入稿证据 | 已完成 |
| `research_log_2026-06-18_ablation_training_entry_preflight.md` | 记录本轮自主研究完成内容、方法和计划修正 | 已完成 |
| `ablation_matched_training_preflight.md` | 记录 ABL-00/02/03/04 matched training 的 split、mask、forward/loss 和输出边界前置检查 | 已完成 |
| `../tools/preflight_ablation_matched_training.py` | 在不训练的条件下验证 matched training 前置条件，不生成 checkpoint 或 prediction | 已完成 |
| `research_log_2026-06-19_ablation_matched_training_preflight.md` | 记录本轮 matched preflight 和下一步 smoke runner 断点 | 已完成 |
| `ablation_matched_training_smoke.md` | 记录 ABL-00/02/03/04 matched smoke 的 checkpoint、history、eligibility 和证据边界 | 已完成 |
| `../tools/audit_ablation_matched_smoke_eligibility.py` | 审计 matched smoke 只能作为 runner continuity 证据，不能进入论文主表 | 已完成 |
| `research_log_2026-06-19_ablation_matched_training_smoke.md` | 记录本轮 matched smoke 和 full matched configuration 下一断点 | 已完成 |
| `ablation_full_matched_training_configuration.md` | 记录正式 matched candidate training 预算、split、tag、evaluator 缺口和 eligibility gate | 已完成 |
| `../tools/preflight_ablation_full_matched_configuration.py` | 在不训练的条件下审计 full matched candidate 配置、split 覆盖、runner 支持和下一步 evaluator 要求 | 已完成 |
| `research_log_2026-06-19_ablation_full_matched_configuration.md` | 记录本轮 full matched configuration 与 matched evaluator 下一断点 | 已完成 |
| `recovery_breakpoint_2026-06-19_full_config_to_matched_evaluator.md` | 记录下一轮恢复入口和启动 full training 前的 evaluator 前置条件 | 已完成 |
| `ablation_matched_full_split_evaluator_smoke.md` | 记录 matched evaluator 的 1-sample smoke 结果和证据边界 | 已完成 |
| `../tools/evaluate_ablation_matched_full_split_metrics.py` | 按 checkpoint tag 读取 matched checkpoints 并生成 per-sample / summary metrics | 已完成 |
| `ablation_matched_full_candidate_results.md` | 记录 full-candidate 训练、7-sample evaluator、eligibility audit 和主要解释 | 已完成 |
| `../tools/audit_ablation_matched_training_eligibility.py` | 审计 full-candidate training/evaluation 是否具备当前阶段消融证据资格 | 已完成 |
| `supervisor_update_2026-06-19.md` | 面向 supervisor 的项目改进点、实验验证成果和 prospect 汇报材料 | 已完成 |
| `research_log_2026-06-19_ablation_full_candidate_results.md` | 记录本轮训练、评估、审计和计划修正 | 已完成 |
| `recovery_breakpoint_2026-06-19_after_full_candidate_eval.md` | 记录 full candidate eval 后的 longer-budget / external SOTA 下一断点 | 已完成 |
| `ablation_matched_longer_repeat_results.md` | 记录 longer-budget repeat 和 candidate-vs-repeat 对比 | 已完成 |
| `research_log_2026-06-19_ablation_longer_repeat.md` | 记录 longer repeat 的执行结果和计划修正 | 已完成 |
| `recovery_breakpoint_2026-06-19_after_longer_repeat.md` | 记录 longer repeat 后的 gated fusion / cue audit / SOTA 下一断点 | 已完成 |

## 8. 当前外部 baseline 数据状态更新

2026-06-18 预检结果显示，presentation package 中只有 P10 样本可直接定位到 `depth_um.npy`，但没有完整 17 帧 focus stack；其余 6 个 test split 样本需要通过仿真器重新生成或建立样本名映射。已新增 P10 单样本导出器，并成功在 `tmp/external_baseline_data/samples/test_V谷_P10_宽谷粗糙平底/` 生成 17 帧 `stack/000.png` 至 `016.png`、`height_gt.npy`、`risk_layers.npy`、`high_risk_mask` 和 DFF/GADFF priors。该临时包只用于 dataloader smoke test，不能直接作为完整 SOTA 数值结果。

Dataloader-only smoke test 已通过：导出的 P10 包可读成 `[17, 540, 960]` frame tensor，并可进一步整理为 `[1, 17, 1, 540, 960]` 或 `[1, 17, 3, 540, 960]` 的 PyTorch-like 输入。下一步才是接入 DFV/DDFFNet 外部仓库和依赖环境。

已新增外部预测评估器，并用 P10 的 DFF/GADFF priors 做自检：MAE、P90、edge MAE 和 high-risk MAE 均与现有 P10 comparison table 对齐。评估器现在使用原项目 `metrics()` 的 high-risk 和 Sobel-edge mask 口径。

已新增批量外部预测评估器。当前用 P10 DFF/GADFF priors 构造了 prediction manifest，并成功生成 `per_sample_metrics.csv` 与 `method_summary_metrics.csv`。后续 DFV/DDFFNet 只要输出 `.npy` 预测并写入 manifest，就可以沿同一管线生成主表候选结果。

已新增 DFV/DDFFNet integration protocol，将下一阶段复现拆成 code inventory、data export、dataloader adapter、prediction export、metric evaluation 和 main-table eligibility gate。下一步可以进入 DFV 外部仓库/环境 smoke test，但仍需保持外部仓库和输出全部在 `tmp/` 下。

已新增外部 SOTA eligibility audit gate。该 gate 会检查预测 manifest、预测文件、batch evaluation 表、训练设置、scale mode、run log 和样本覆盖情况。现有 P10 DFF/GADFF prior smoke test 预计会被判定为 auxiliary / not main-table evidence，因为它只有单样本且不是外部 deep DFF 预测。

已新增 DFV/DDFFNet workspace scaffold，并生成 `tmp/external_baseline_results/DFV/` 与 `tmp/external_baseline_results/DDFFNet/`。当前两个方法的 `run_config.json` 均标记为 `scaffold_only_no_external_prediction`，`main_table_eligible` 为 `false`。这只说明日志、manifest 和输出目录已准备好，不代表外部模型已经运行。

已新增 research package integrity audit，并在 `tmp/research_package_audits/` 生成报告。当前审计结果为 pass：核心研究文档、工具脚本、DFV/DDFFNet 临时 workspace、P10 外部数据包和 claim-safety 文本检查均通过。

已新增 manuscript claim safety audit，并在 `tmp/manuscript_audits/` 生成报告。当前 LaTeX 草稿审计结果为 pass：未发现外部 SOTA superiority、真实绝对精度、过度泛化或未完成消融结论等风险表达。

已新增 ablation execution protocol，并生成 `tmp/ablation_results/ABL-00` 至 `ABL-06` 的运行脚手架。当前所有 ablation run 均标记为 `scaffold_only_no_training_run`，`claim_eligible` 为 `false`，因此只能作为后续实验准备，不能写作已完成消融结果。

已新增 ablation feature schema audit。当前 P10 base features 为 `[22, 540, 960]`，通道顺序是 17 层焦栈、risk、DFF、DFF confidence、GADFF、GADFF confidence。ABL-01、ABL-02、ABL-04 可先定义为 base input channel mask。进一步的 ABL-03 implementation audit 已确认 Focus-ResUNet 使用 `augment_features()` 构造 `[38, 540, 960]` upgraded features，其中 17-32 通道是 16 层相邻焦平面差分；因此 ABL-03 推荐先通过 zero channels 17-32 作为 w/o focal-difference input signal。

已新增 ablation mask smoke test。当前报告位于 `tmp/ablation_results/mask_smoke/`，结果为 pass：ABL-01/02/04 在 base 22 通道特征上按定义置零，ABL-03 在 upgraded 38 通道特征上置零 17-32 焦向差分通道，且非目标通道保持不变。该结果只证明 mask 操作正确，不构成消融性能结论。

已新增 Depth Anything V2 auxiliary protocol 和临时 workspace scaffold。当前 `tmp/foundation_depth_auxiliary/DepthAnythingV2/` 只包含 `run_config.json`、`input_manifest.csv` 和 protocol log，状态为 `scaffold_only_no_model_run`，`main_table_eligible=false`。该路径用于后续单帧 relative depth 辅助可视化，不参与 focus-stack 主数值 SOTA 表。

已将 Depth Anything V2 的定位同步到 LaTeX 草稿和 `references.bib`。正文现在把它写作 monocular foundation-depth training-strategy reference 和 single-frame auxiliary prior，并明确其输出不能混入 focus-stack 主 SOTA 数值表。

已新增投稿差距闭环计划。当前结论是：下一步主实验优先级应从 Depth Anything V2 辅助线转回 DFV 外部 baseline 和核心消融，因为这两项最直接决定稿件能否从“内部项目报告”推进到“可投稿论文”。

已新增 DFV environment preflight。当前结果为 pass：P10 临时数据包完整，DFV 结果目录存在，NumPy/PIL/PyTorch 可用，CUDA 可用且未生成任何 DFV 预测。DFV 仓库尚未位于 `tmp/external_repos/DFV/`，因此下一步应在获得网络/依赖许可后进行 repository download / code inventory。

## 9. 当前消融状态更新

已新增 ABL-00/01/02/03/04 training-entry preflight。当前结果为 pass：脚手架、schema audit、ABL-03 focal-difference implementation audit 和 mask smoke test 均可追溯。最关键的计划修正是：final-method 消融应锚定 `src/train_focus_resunet_loss_experiment.py` 的 upgraded 38-channel Focus-ResUNet 路径，早期 `src/final_dataset_training.py` 保留为基础训练/数据参考。下一步需要先设计 minimal ablation runner，并在形状、通道、metrics 写入和日志字段通过验证后再训练。

2026-06-19 runner smoke 已通过。ABL-00/02/03/04 均能在 upgraded 38-channel Focus-ResUNet 输入空间完成无训练前向形状检查，ABL-01 已记录为单独 lower-prior runner 设计项。下一步从“接口可行性”进入“训练 runner 与 metrics 写入”，但仍不能把 smoke diagnostic loss 写成消融结果。

2026-06-19 training-runner preflight 已通过。关键更新是：未来 ABL runner 不能直接调用 `train_focus_resunet_loss_experiment.py::main()` 或其输出型评估/绘图函数，因为它们会写入原项目交付包。下一步应实现一个独立 runner，把 checkpoint、日志、metrics、预测和图像全部限制在 `tmp/ablation_results/<run_id>/`。

2026-06-19 ABL-00/03 training-runner dry-run 已通过。当前 runner 只做 forward/loss interface check，不创建 optimizer、不反向传播、不保存模型或预测。下一步可以在该工具中增加受保护的小规模训练模式，再跑 ABL-00/03 的 1 epoch 临时训练。

2026-06-19 ABL-00/03 small-training debug 已通过。该运行只覆盖 P10 patch-sampling 级别的最小训练链路，证明 optimizer/backward/update 与 checkpoint/metrics 写入可在 `tmp/ablation_results/` 内完成。

2026-06-19 ABL-00/02/03/04 controlled pilot 已通过。该运行通过 `--run-kind controlled_pilot --tag 2026-06-19_controlled_pilot` 生成独立产物，避免覆盖 small-training debug 文件。pilot eligibility audit 结果为 pass，但 eligibility 仍为 debug-only。

2026-06-19 ABL-00/02/03/04 full-split debug evaluation 已通过。该运行加载 controlled-pilot checkpoints，在 7 个 synthetic test samples 上完成 tiled inference 和 MAE、edge MAE、high-risk MAE 计算。full-split eligibility audit 结果为 pass，但 eligibility 仍为 diagnostic-only。

2026-06-19 ABL-00/02/03/04 matched training preflight 已通过。该运行确认 train/validation/test split 为 27/10/7，四个核心变体均可完成 64 x 64 patch forward/loss 检查，并且 planned outputs 被限定在 `tmp/ablation_results/<run_id>/`。下一步应先进入 matched training smoke runner，再推进正式 matched full-split ablation training。

2026-06-19 ABL-00/02/03/04 matched training smoke 已通过。该运行确认四个核心变体都能在 fixed split boundary 下完成最小 optimizer/backward/checkpoint/history/log 链路，matched smoke eligibility audit 结果为 pass，但 eligibility 仍为 matched-smoke-only。

2026-06-19 full matched ablation training configuration 已通过。该配置固定正式候选训练预算为 4 epochs、128 train patches、32 validation patches、batch size 1、learning rate 0.0006，并覆盖 train/validation/test split 27/10/7。下一步应先实现 matched full-split evaluator，再启动 full-candidate training。

2026-06-19 matched full candidate training 和 7-sample evaluator 已完成。当前结果最强支持 DFF/GADFF prior 的必要性；去掉 prior 后 Mean MAE 和 high-risk MAE 显著恶化。full model 当前未优于 w/o focal difference 和 w/o glare cue，说明辅助通道融合和训练预算需要复核。面向 supervisor 的更新材料已整理到 `supervisor_update_2026-06-19.md`。

2026-06-19 longer-budget repeat 已完成。更长训练降低了所有变体的误差，但 full model 仍未优于 w/o focal difference 和 w/o glare cue。因此下一步重点应从单纯扩大训练预算转向辅助信号融合、glare cue quality audit、seed repeat，以及外部 SOTA 对比。
