# Research Package Integrity Audit

Updated: 2026-06-18

Purpose: define a non-destructive audit for the autonomous research package. The audit checks whether the current planning, SOTA, external-baseline, manuscript, and temporary-output structure is internally consistent and does not imply unsupported results.

## 1. Audit Scope

The audit covers:

1. required planning documents under `submission_planning/autonomous_research/`;
2. required helper scripts under `submission_planning/tools/`;
3. external baseline temporary workspaces under `tmp/external_baseline_results/`;
4. external data package under `tmp/external_baseline_data/`;
5. foundation-depth auxiliary workspace under `tmp/foundation_depth_auxiliary/`;
6. claim-safety text checks in planning documents.

The audit does not inspect model performance, run external repositories, download dependencies, or edit original project assets.

## 2. Required Core Documents

| Category | Required files |
|---|---|
| index and task control | `research_index.md`, `research_task_board.md`, `experiment_roadmap.md`, `research_log_2026-06-18_ablation_training_entry_preflight.md`, `research_log_2026-06-19_minimal_ablation_runner_smoke.md`, `research_log_2026-06-19_ablation_training_runner_preflight.md`, `research_log_2026-06-19_ablation_controlled_pilot.md`, `research_log_2026-06-19_ablation_full_split_debug_eval.md`, `research_log_2026-06-19_ablation_matched_training_preflight.md`, `research_log_2026-06-19_ablation_matched_training_smoke.md`, `research_log_2026-06-19_ablation_full_matched_configuration.md`, `research_log_2026-06-19_ablation_matched_evaluator_smoke.md`, `research_log_2026-06-19_ablation_full_candidate_results.md`, `research_log_2026-06-19_ablation_longer_repeat.md`, `recovery_breakpoint_2026-06-19_matched_training_preflight_to_smoke.md`, `recovery_breakpoint_2026-06-19_matched_smoke_to_full_config.md`, `recovery_breakpoint_2026-06-19_full_config_to_matched_evaluator.md`, `recovery_breakpoint_2026-06-19_matched_evaluator_to_full_training.md`, `recovery_breakpoint_2026-06-19_after_full_candidate_eval.md`, `recovery_breakpoint_2026-06-19_after_longer_repeat.md` |
| SOTA and claim safety | `external_baseline_feasibility.md`, `claim_evidence_matrix.md`, `external_sota_next_decision_log.md`, `external_sota_eligibility_audit.md`, `depth_anything_v2_auxiliary_protocol.md`, `submission_gap_closure_plan.md`, `manuscript_claim_safety_audit.md`, `research_package_integrity_audit.md` |
| external execution | `baseline_adapter_spec.md`, `external_baseline_data_preflight.md`, `dfv_ddffnet_integration_protocol.md`, `external_baseline_workspace_scaffold.md`, `dfv_environment_preflight.md` |
| manuscript planning | `manuscript_blueprint.md`, `latex_manuscript_assembly_plan.md`, `abstract_draft_en.md`, `introduction_outline_en.md`, `method_outline_en.md`, `experiments_outline_en.md`, `discussion_outline_en.md` |
| experiment design | `ablation_design.md`, `ablation_execution_protocol.md`, `ablation_feature_schema_audit.md`, `abl03_focal_difference_implementation_audit.md`, `ablation_mask_smoke_test.md`, `ablation_training_entry_preflight.md`, `minimal_ablation_runner_smoke.md`, `ablation_training_runner_preflight.md`, `ablation_training_runner_dry_run.md`, `ablation_small_training_debug.md`, `ablation_controlled_pilot.md`, `ablation_full_split_debug_evaluation.md`, `ablation_matched_training_preflight.md`, `ablation_matched_training_smoke.md`, `ablation_full_matched_training_configuration.md`, `ablation_matched_full_split_evaluator_smoke.md`, `ablation_matched_full_candidate_results.md`, `ablation_matched_longer_repeat_results.md`, `supervisor_update_2026-06-19.md`, `supervisor_experiment_report_2026-06-19.md`, `failure_analysis_plan.md`, `figure_table_plan.md`, `submission_readiness_checklist.md` |

## 3. Required Tools

| Tool | Role |
|---|---|
| `preflight_external_baseline_data.py` | inspect whether local data can support external baselines |
| `export_one_external_baseline_sample.py` | create temporary focus-stack package |
| `preflight_dfv_environment.py` | record local DFV readiness before repository download or model execution |
| `smoke_test_external_baseline_package.py` | validate exported package tensor layout |
| `evaluate_external_prediction.py` | evaluate one external `.npy` prediction |
| `evaluate_external_prediction_batch.py` | summarize multiple predictions |
| `audit_external_sota_eligibility.py` | decide whether an external method can enter the main table |
| `scaffold_external_baseline_workspace.py` | create temporary DFV/DDFFNet result workspaces |
| `audit_manuscript_claim_safety.py` | audit the LaTeX manuscript for unsupported claim patterns |
| `scaffold_ablation_workspace.py` | create temporary ablation run directories, configs, logs, and metric templates |
| `audit_ablation_feature_schema.py` | verify current feature-channel schema and ablation channel-mask definitions |
| `audit_abl03_focal_difference_implementation.py` | verify the upgraded Focus-ResUNet focal-difference channel implementation |
| `smoke_test_ablation_masks.py` | verify ABL-01/02/03/04 input masks zero intended channels and preserve non-target channels |
| `preflight_ablation_training_entry.py` | verify ABL-00/01/02/03/04 training-entry assumptions before any ablation run |
| `run_ablation_variant_smoke.py` | verify minimal ABL-00/02/03/04 final-method runner shape and ABL-01 lower-prior design boundary |
| `preflight_ablation_training_runner.py` | verify the future training runner redirects outputs under `tmp/ablation_results/` and gates ABL-01 |
| `run_ablation_variant_training.py` | protected ablation dry-run, small-training debug, and controlled-pilot runner |
| `audit_ablation_pilot_eligibility.py` | verify controlled pilot outputs remain debug-only and claim-ineligible |
| `evaluate_ablation_full_split_metrics.py` | evaluate controlled-pilot checkpoints on the fixed synthetic test split under `tmp/` |
| `audit_ablation_full_split_eligibility.py` | verify full-split diagnostic metrics remain claim-ineligible |
| `preflight_ablation_matched_training.py` | verify matched training split, masks, forward/loss checks, and temporary output paths before training |
| `audit_ablation_matched_smoke_eligibility.py` | verify matched smoke outputs remain smoke-only and claim-ineligible |
| `preflight_ablation_full_matched_configuration.py` | verify full matched candidate training budget, split coverage, evaluator plan, and eligibility-gate requirements before training |
| `evaluate_ablation_matched_full_split_metrics.py` | evaluate matched checkpoints selected by checkpoint tag on the fixed synthetic test split |
| `audit_ablation_matched_training_eligibility.py` | audit matched full-candidate training, 7-sample metrics, checkpoints, logs, and current-stage evidence eligibility |
| `scaffold_depth_anything_v2_auxiliary_workspace.py` | create a scaffold-only Depth Anything V2 auxiliary workspace |

## 4. Safety Checks

| Check | Pass condition |
|---|---|
| no forbidden Chinese response pattern in planning files | no match for the project-level banned contrast sentence pattern |
| no unsupported external SOTA claim | no unconditional claim that the proposed method is stronger than DFV/DDFFNet before predictions exist |
| external method status | DFV/DDFFNet `run_config.json` has `main_table_eligible=false` until predictions and audits pass |
| DFV environment preflight | DFV preflight report exists and records `status=pass` with no errors |
| ablation training-entry preflight | ABL-00/01/02/03/04 entry assumptions are checked and corrected before training |
| minimal ablation runner smoke | ABL-00/02/03/04 pass 38-channel shape/loss smoke and ABL-01 remains a separate lower-prior design |
| ablation training-runner preflight | future training runner avoids original delivery-package outputs and keeps ABL-01 gated |
| ablation training-runner dry run | ABL-00/03 dry-run logs exist, no optimizer/backward/checkpoint/prediction/metric result is produced |
| ablation small-training debug | ABL-00/03 debug checkpoints and metrics exist only under `tmp/ablation_results/`, with claim eligibility disabled |
| ablation controlled pilot | ABL-00/02/03/04 controlled-pilot checkpoints and metrics exist only under `tmp/ablation_results/`, with claim eligibility disabled |
| ablation pilot eligibility audit | pilot audit exists and states debug-only, not main-table evidence |
| ablation full-split debug evaluation | ABL-00/02/03/04 per-sample and method-summary metrics exist for 7 synthetic test samples |
| ablation full-split eligibility audit | diagnostic full-split audit exists and states not manuscript ablation evidence |
| ablation matched training preflight | ABL-00/02/03/04 share train/validation/test split, finite train/validation loss diagnostics, and claim eligibility remains disabled |
| ablation matched training smoke | ABL-00/02/03/04 smoke checkpoints/history/logs exist under `tmp/ablation_results/`, split counts are 27/10/7, and eligibility remains disabled |
| ablation full matched configuration preflight | full candidate training config exists, uses split counts 27/10/7, keeps claim eligibility disabled, and records the matched evaluator requirement |
| ablation matched evaluator smoke | evaluator can load checkpoint tag `2026-06-19_matched_training_smoke` and write 1-sample smoke metrics under `tmp/` |
| ablation matched full candidate training | ABL-00/02/03/04 full-candidate checkpoints and histories exist under `tmp/ablation_results/` |
| ablation matched full candidate evaluation | 7-sample evaluator produces 28 per-sample rows and 4 method summary rows |
| ablation matched full candidate eligibility audit | audit passes and records current-stage ablation evidence with supervisor-review caveat |
| ablation matched longer repeat | 8-epoch repeat checkpoints, histories, and 7-sample evaluator outputs exist under `tmp/` |
| foundation-depth auxiliary status | Depth Anything V2 `run_config.json` has `auxiliary_only=true` and `main_table_eligible=false` |
| manuscript foundation-depth boundary | manuscript cites Depth Anything V2 as auxiliary and keeps it outside the main SOTA table |
| temporary outputs isolated | external baseline directories stay under `tmp/` |
| no cache artifacts in tools | no `__pycache__` under `submission_planning/tools/` |

## 5. Tool

Use:

```powershell
python -X utf8 submission_planning/tools/audit_research_package_integrity.py
```

The default report path is:

```text
tmp/research_package_audits/research_package_integrity_audit.md
tmp/research_package_audits/research_package_integrity_audit.json
```

## 6. Interpretation

Passing this audit means the research package is internally coherent and safe to continue. It does not mean the paper is submission-ready, because external deep DFF results, ablations, and real calibrated height validation may still be missing.
