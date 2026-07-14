# 自主研究任务板

更新日期：2026-06-19  
用途：跟踪后续自主研究任务，避免任务散落在对话中。  
规则：这里只记录研究规划和执行状态，不存放外部仓库、模型权重或大文件。

## 1. 状态说明

| 状态 | 含义 |
|---|---|
| Ready | 可以直接执行 |
| Needs setup | 需要环境、数据转换或外部代码 |
| Waiting | 等待用户决策、设备、标定数据或网络条件 |
| Done | 已有明确产物 |
| Deferred | 可推迟到下一版 |

## 2. 当前 Done

| ID | 任务 | 产物 |
|---|---|---|
| D1 | 中文 SOTA 清单 | `submission_planning/sota_comparison_chinese.md` |
| D2 | 自主研究索引 | `research_index.md` |
| D3 | 外部基线可行性 | `external_baseline_feasibility.md` |
| D4 | 数据接口说明 | `dataset_interface_notes.md` |
| D5 | 消融实验设计 | `ablation_design.md` |
| D6 | 投稿 readiness 清单 | `submission_readiness_checklist.md` |
| D7 | Related Work 中文草稿 | `related_work_draft_cn.md` |
| D8 | Claim-evidence 矩阵 | `claim_evidence_matrix.md` |
| D9 | 外部基线复现决策表 | `baseline_reproduction_decision_table.md` |
| D10 | Manuscript blueprint | `manuscript_blueprint.md` |
| D11 | 英文 Abstract 安全版 | `abstract_draft_en.md` |
| D12 | 英文 Introduction outline | `introduction_outline_en.md` |
| D13 | 外部 baseline 数据转换规格 | `baseline_adapter_spec.md` |
| D14 | Figure/Table 编号与 caption 草稿 | `figure_table_plan.md` |
| D15 | Failure analysis 写作草稿 | `failure_analysis_plan.md` |
| D16 | 英文 Method outline | `method_outline_en.md` |
| D17 | 英文 Experiments outline | `experiments_outline_en.md` |
| D18 | 英文 Discussion outline | `discussion_outline_en.md` |
| D19 | LaTeX manuscript assembly plan | `latex_manuscript_assembly_plan.md` |
| D20 | DFV smoke-test checklist | `dfv_smoke_test_checklist.md` |
| D21 | External baseline log template | `baseline_run_log_template.md` |
| D22 | First LaTeX manuscript draft | `../manuscript_draft/s2r_focus_stack_manuscript.tex` |
| D23 | External baseline data preflight and P10 export tools | `external_baseline_data_preflight.md`, `../tools/preflight_external_baseline_data.py`, `../tools/export_one_external_baseline_sample.py` |
| D24 | P10 external package dataloader smoke test | `../tools/smoke_test_external_baseline_package.py`, `baseline_run_logs/2026-06-18_p10_export_smoke.md` |
| D25 | External prediction evaluator smoke test | `external_prediction_evaluator.md`, `../tools/evaluate_external_prediction.py` |
| D26 | Batch external prediction evaluation utility | `batch_external_evaluation.md`, `../tools/evaluate_external_prediction_batch.py` |
| D27 | DFV/DDFFNet integration protocol | `dfv_ddffnet_integration_protocol.md` |
| D28 | External SOTA next-step decision log | `external_sota_next_decision_log.md` |
| D29 | External SOTA eligibility audit gate | `external_sota_eligibility_audit.md`, `../tools/audit_external_sota_eligibility.py` |
| D30 | External baseline workspace scaffold | `external_baseline_workspace_scaffold.md`, `../tools/scaffold_external_baseline_workspace.py`, `tmp/external_baseline_results/DFV/`, `tmp/external_baseline_results/DDFFNet/` |
| D31 | Research package integrity audit | `research_package_integrity_audit.md`, `../tools/audit_research_package_integrity.py`, `tmp/research_package_audits/research_package_integrity_audit.md` |
| D32 | Manuscript claim safety audit | `manuscript_claim_safety_audit.md`, `../tools/audit_manuscript_claim_safety.py`, `tmp/manuscript_audits/manuscript_claim_safety_audit.md` |
| D33 | Ablation execution protocol | `ablation_execution_protocol.md`, `templates/ablation_run_matrix_template.csv` |
| D34 | Ablation workspace scaffold | `../tools/scaffold_ablation_workspace.py`, `tmp/ablation_results/ABL-00` to `tmp/ablation_results/ABL-06` |
| D35 | Ablation feature schema audit | `ablation_feature_schema_audit.md`, `../tools/audit_ablation_feature_schema.py`, `tmp/ablation_results/schema_audit/ablation_feature_schema_audit.md` |
| D36 | ABL-03 focal-difference implementation audit | `abl03_focal_difference_implementation_audit.md`, `../tools/audit_abl03_focal_difference_implementation.py`, `tmp/ablation_results/ABL-03/logs/abl03_focal_difference_implementation_audit.md` |
| D37 | Ablation mask smoke test | `ablation_mask_smoke_test.md`, `../tools/smoke_test_ablation_masks.py`, `tmp/ablation_results/mask_smoke/ablation_mask_smoke_test.md` |
| D38 | Depth Anything V2 auxiliary protocol and scaffold | `depth_anything_v2_auxiliary_protocol.md`, `../tools/scaffold_depth_anything_v2_auxiliary_workspace.py`, `tmp/foundation_depth_auxiliary/DepthAnythingV2/` |
| D39 | Manuscript Depth Anything V2 synchronization | `../manuscript_draft/s2r_focus_stack_manuscript.tex`, `../manuscript_draft/references.bib` |
| D40 | Submission gap closure plan | `submission_gap_closure_plan.md` |
| D41 | DFV environment preflight | `dfv_environment_preflight.md`, `../tools/preflight_dfv_environment.py`, `tmp/external_baseline_results/DFV/preflight/dfv_environment_preflight.md` |
| D42 | Ablation training-entry preflight | `ablation_training_entry_preflight.md`, `../tools/preflight_ablation_training_entry.py`, `tmp/ablation_results/preflight/ablation_training_entry_preflight.md` |
| D43 | 本轮自主研究日志 | `research_log_2026-06-18_ablation_training_entry_preflight.md` |
| D44 | Minimal ablation runner smoke | `minimal_ablation_runner_smoke.md`, `../tools/run_ablation_variant_smoke.py`, `tmp/ablation_results/runner_smoke/minimal_ablation_runner_smoke.md` |
| D45 | Ablation training-runner preflight | `ablation_training_runner_preflight.md`, `../tools/preflight_ablation_training_runner.py`, `tmp/ablation_results/training_runner_preflight/ablation_training_runner_preflight.md` |
| D46 | ABL-01 lower-prior architecture decision | `tmp/ablation_results/ABL-01/logs/2026-06-19_lower_prior_architecture_decision.md` |
| D47 | ABL-00/03 training-runner dry run | `ablation_training_runner_dry_run.md`, `../tools/run_ablation_variant_training.py`, `tmp/ablation_results/training_runner_dry_run/ablation_training_runner_dry_run_summary.md` |
| D48 | ABL-00/03 small-training debug | `ablation_small_training_debug.md`, `tmp/ablation_results/training_runner_small_train/ablation_training_runner_small_train_summary.md` |
| D49 | 本轮开发日志与恢复断点 | `research_log_2026-06-19_development_resume_checkpoint.md`, `recovery_breakpoint_2026-06-19_ablation_debug_to_pilot.md` |
| D50 | Controlled pilot tag/run-kind support | `../tools/run_ablation_variant_training.py` |
| D51 | ABL-00/02/03/04 controlled pilot | `ablation_controlled_pilot.md`, `tmp/ablation_results/training_runner_controlled_pilot/2026-06-19_controlled_pilot_summary.md` |
| D52 | ABL pilot eligibility audit | `../tools/audit_ablation_pilot_eligibility.py`, `tmp/ablation_results/eligibility_audits/ABL_pilot_eligibility.md` |
| D53 | 本轮 controlled pilot 研究日志 | `research_log_2026-06-19_ablation_controlled_pilot.md` |
| D54 | Full-split diagnostic metrics runner | `../tools/evaluate_ablation_full_split_metrics.py` |
| D55 | ABL-00/02/03/04 full-split debug evaluation | `ablation_full_split_debug_evaluation.md`, `tmp/ablation_results/full_split_debug_eval/2026-06-19_full_split_debug_eval_summary.md` |
| D56 | Full-split diagnostic eligibility audit | `../tools/audit_ablation_full_split_eligibility.py`, `tmp/ablation_results/eligibility_audits/ABL_full_split_eligibility.md` |
| D57 | 本轮 full-split debug evaluation 研究日志 | `research_log_2026-06-19_ablation_full_split_debug_eval.md` |
| D58 | Matched training preflight | `ablation_matched_training_preflight.md`, `../tools/preflight_ablation_matched_training.py`, `tmp/ablation_results/matched_training_preflight/2026-06-19_matched_training_preflight.md` |
| D59 | 本轮 matched training preflight 研究日志 | `research_log_2026-06-19_ablation_matched_training_preflight.md` |
| D60 | Matched training preflight 恢复断点 | `recovery_breakpoint_2026-06-19_matched_training_preflight_to_smoke.md` |
| D61 | Matched smoke runner support | `../tools/run_ablation_variant_training.py` |
| D62 | ABL-00/02/03/04 matched smoke runs | `ablation_matched_training_smoke.md`, `tmp/ablation_results/training_runner_matched_smoke/2026-06-19_matched_training_smoke_summary.md` |
| D63 | Matched smoke eligibility audit | `../tools/audit_ablation_matched_smoke_eligibility.py`, `tmp/ablation_results/eligibility_audits/ABL_matched_smoke_eligibility.md` |
| D64 | 本轮 matched smoke 研究日志与恢复断点 | `research_log_2026-06-19_ablation_matched_training_smoke.md`, `recovery_breakpoint_2026-06-19_matched_smoke_to_full_config.md` |
| D65 | Matched full-candidate runner support | `../tools/run_ablation_variant_training.py` |
| D66 | Full matched training configuration preflight | `ablation_full_matched_training_configuration.md`, `../tools/preflight_ablation_full_matched_configuration.py`, `tmp/ablation_results/matched_training_full_config/2026-06-19_matched_training_full_candidate_config_preflight.md` |
| D67 | 本轮 full matched configuration 研究日志与恢复断点 | `research_log_2026-06-19_ablation_full_matched_configuration.md`, `recovery_breakpoint_2026-06-19_full_config_to_matched_evaluator.md` |
| D68 | Matched full-split evaluator implementation | `../tools/evaluate_ablation_matched_full_split_metrics.py` |
| D69 | Matched evaluator smoke on matched-smoke checkpoints | `ablation_matched_full_split_evaluator_smoke.md`, `tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_evaluator_smoke/2026-06-19_matched_evaluator_smoke_summary.md` |
| D70 | 本轮 matched evaluator 研究日志与恢复断点 | `research_log_2026-06-19_ablation_matched_evaluator_smoke.md`, `recovery_breakpoint_2026-06-19_matched_evaluator_to_full_training.md` |
| D71 | ABL-00/02/03/04 matched full candidate training | `tmp/ablation_results/training_runner_matched_full_candidate/2026-06-19_matched_training_full_candidate_summary.md` |
| D72 | Matched full-candidate 7-sample evaluator run | `tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_full_candidate_eval/2026-06-19_matched_full_candidate_eval_summary.md` |
| D73 | Full matched candidate eligibility audit | `../tools/audit_ablation_matched_training_eligibility.py`, `tmp/ablation_results/eligibility_audits/ABL_matched_training_eligibility.md` |
| D74 | Full candidate results and supervisor update | `ablation_matched_full_candidate_results.md`, `supervisor_update_2026-06-19.md` |
| D75 | 本轮 full candidate 研究日志与恢复断点 | `research_log_2026-06-19_ablation_full_candidate_results.md`, `recovery_breakpoint_2026-06-19_after_full_candidate_eval.md` |
| D76 | Matched longer-budget repeat | `tmp/ablation_results/training_runner_matched_longer_repeat/2026-06-19_matched_training_longer_repeat_summary.md` |
| D77 | Matched longer-repeat 7-sample evaluator | `tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_longer_repeat_eval/2026-06-19_matched_longer_repeat_eval_summary.md` |
| D78 | Longer-repeat results and supervisor update refresh | `ablation_matched_longer_repeat_results.md`, `supervisor_update_2026-06-19.md` |
| D79 | 本轮 longer-repeat 研究日志与恢复断点 | `research_log_2026-06-19_ablation_longer_repeat.md`, `recovery_breakpoint_2026-06-19_after_longer_repeat.md` |
| D80 | Supervisor-facing experiment report with result figures | `supervisor_experiment_report_2026-06-19.md`, `report_figures_2026-06-19/` |

## 3. Ready

| ID | 任务 | 目标产物 | 依赖 |
|---|---|---|---|
| R13 | DFV single-sample smoke test | temporary outputs under `tmp/` | `dfv_smoke_test_checklist.md` |
| R14 | Baseline run log instance | future run log file | `baseline_run_log_template.md` |
| R16 | DFV repository/environment smoke test | external repo under `tmp/external_repos/` and run log | network/dependency approval |
| R18 | DFV P10 prediction export contract test | `tmp/external_baseline_results/DFV/predictions/test_V谷_P10_宽谷粗糙平底.npy` | DFV dataloader/inference |
| R19 | DFV inventory log | `tmp/external_baseline_results/DFV/logs/<date>_inventory.md` | `external_sota_next_decision_log.md` |
| R20 | DFV eligibility audit after prediction | `tmp/external_baseline_results/eligibility_audits/DFV_eligibility_audit.md` | DFV prediction manifest and batch evaluation |
| R21 | DDFFNet loader inventory | `tmp/external_baseline_results/DDFFNet/logs/<date>_inventory.md` | external repo/code inspection |
| R22 | Minimal ablation runner design | `submission_planning/tools/run_ablation_variant_smoke.py` 或等效临时脚本 | `ablation_execution_protocol.md` |
| R23 | ABL-00 config verification | `tmp/ablation_results/ABL-00/logs/<date>_config_verification.md` | 当前训练入口和特征通道顺序核验 |
| R24 | ABL-03 zero-difference smoke runner | `tmp/ablation_results/ABL-03/logs/<date>_zero_diff_smoke.md` | upgraded 38-channel feature path verified |
| R25 | Depth Anything V2 input preprocessing decision | best-focus or AiF input selection note under `tmp/foundation_depth_auxiliary/DepthAnythingV2/logs/` | `depth_anything_v2_auxiliary_protocol.md` |
| R26 | DFV repository download / code inventory | download or place DFV under `tmp/external_repos/DFV/`, then record code inventory | `dfv_environment_preflight.md`, network/dependency approval |
| R32 | DFV repository inventory and P10 prediction contract | `tmp/external_baseline_results/DFV/logs/<date>_inventory.md`, P10 `.npy` prediction contract | `dfv_environment_preflight.md` |
| R36 | ABL-01 lower-prior training design | lower-prior focus-stack-only architecture note and runner plan | `2026-06-19_lower_prior_architecture_decision.md` |
| R44 | ABL-01 lower-prior matched lane | 单独 lower-prior focus-stack-only runner 的 matched 训练配置 | `2026-06-19_lower_prior_architecture_decision.md` |
| R51 | Seed repeat for current matched candidate | 至少 2 个 seeds 的 stability comparison | D71-D73 |
| R52 | Gated auxiliary-signal fusion design | focal-difference / glare cue gating design note and runner plan | D71-D73 |
| R53 | DFV repository inventory and P10 prediction contract | `tmp/external_baseline_results/DFV/logs/<date>_inventory.md`, P10 `.npy` prediction contract | DFV environment preflight |
| R54 | Glare cue quality audit | glare/risk cue 分布、与 error map 的相关性和可视化诊断 | D76-D78 |

## 4. Needs Setup

| ID | 任务 | 需要准备 | 风险 |
|---|---|---|---|
| S1 | DFV 复现 | 外部仓库、环境、1 个样本 adapter | 网络/依赖/GPU |
| S1a | 完整 synthetic split 外部导出 | 将 24 个样本按统一格式导出到 `tmp/external_baseline_data/` | 需确认导出体积和后续是否长期保留 |
| S2 | DDFFNet 复现 | HDF5 或官方数据格式转换 | 老代码依赖 |
| S3 | Depth Anything V2 qualitative figure | 模型权重、真实样本最佳聚焦帧或 AiF 图 | 单帧输出与显微形貌不直接可比，只能作为 auxiliary qualitative comparison |
| S4 | w/o prior 消融 | 训练脚本或模型配置调整 | 需要确认现有代码是否支持 |
| S5 | w/o focal difference 消融 | 输入通道配置调整 | 需要重新训练 |
| S6 | w/o glare cue 消融 | high-risk / glare cue 开关 | high-risk mask 需一致 |

## 5. Waiting

| ID | 任务 | 等待内容 |
|---|---|---|
| W1 | 真实 calibrated GT 验证 | step-height / profilometer / confocal / white-light interferometry 数据 |
| W2 | DDFS 数值复现 | 完整或等效相机参数 |
| W3 | 目标期刊选择 | 决定 Sensors / Applied Sciences / IEEE Access / Measurement 等投稿路径 |

## 6. Deferred

| ID | 任务 | 推迟原因 |
|---|---|---|
| F1 | 大规模真实无标签 self-training | 工作量大，适合下一版 |
| F2 | Minimal focal stack 实验 | 需要重新设计采集帧数和训练设置 |
| F3 | FAD / DualFocus 复现 | 当前代码可用性不明确 |
| F4 | 完整系统部署评估 | 当前论文重点在方法和 sim-to-real 论证 |

## 7. 下一轮建议

若继续写作优先：

1. 先完成 `latex_manuscript_assembly_plan.md`；
2. 再把 Abstract、Introduction、Method、Experiments、Discussion 大纲扩展为 LaTeX 正文；
3. 然后统一引用和图表占位。

若继续实验优先：

1. 先按 `baseline_adapter_spec.md` 做 DFV 单样本 smoke test；
2. 再评估是否进入完整 7-sample synthetic test；
3. 然后考虑 DDFFNet adapter。

## 8. 当前最高优先级

| 排名 | 任务 | 原因 |
|---|---|---|
| 1 | DFV baseline adapter / gap closure | 最影响 SOTA 对比可信度 |
| 2 | Gated auxiliary fusion / cue audit | longer repeat 后 full model 仍未占优，重点转向辅助信号质量和融合机制 |
| 3 | Failure analysis | 最影响审稿风险控制 |
| 4 | Figure/Table update after new metrics | 最影响论文呈现效率 |
| 5 | Journal-strengthened real validation plan | 最影响真实域说服力 |

## 9. 最新恢复断点

当前恢复入口为 `research_log_2026-06-19_ablation_longer_repeat.md` 和 `recovery_breakpoint_2026-06-19_after_longer_repeat.md`。当前 longer-budget repeat 已完成，结果显示训练预算增加后 full model 仍未占优。下一步应根据 supervisor 反馈选择 gated auxiliary-signal fusion、glare cue quality audit、seed repeat 或 DFV/DDFFNet 外部 SOTA。
