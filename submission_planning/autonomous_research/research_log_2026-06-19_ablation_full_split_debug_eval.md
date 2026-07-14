# 本轮自主研究日志：ABL Full-Split Debug Evaluation

日期：2026-06-19  
范围：`submission_planning/tools/`、`submission_planning/autonomous_research/`、`tmp/ablation_results/`。  
边界：本轮只对 controlled-pilot checkpoints 做完整 test split 诊断评价；未改写 `src/`，未写入原始论文包、PPT 包或既有结果目录。

## 1. 本轮目标

上一轮已经完成 ABL-00/02/03/04 controlled pilot，但它只覆盖单个 P10 样本的小规模训练。本轮目标是建立 full-split metrics runner，确认相同 checkpoint 能否在固定 7 个 synthetic test samples 上完成 tiled inference、统一指标计算、per-run metrics 写入和 eligibility audit。

## 2. 采用的方式

新增 `submission_planning/tools/evaluate_ablation_full_split_metrics.py`。该脚本只复用以下只读部件：

```text
src/final_dataset_training.py::build_dataset
src/simulate_antiglare_highres_samples.py::generate_sample_arrays
src/simulate_antiglare_highres_samples.py::metrics
src/train_focus_resunet_loss_experiment.py::FocusResUNet
src/train_focus_resunet_loss_experiment.py::predict_tiled_upgraded
```

脚本不调用原训练脚本的 `main()`、`evaluate_split()`、`write_metric_plots()` 或 `write_report()`，因此不会写入原项目交付包。

## 3. 已完成任务

已完成 1-sample smoke test：

```text
python -X utf8 submission_planning/tools/evaluate_ablation_full_split_metrics.py --max-samples 1 --tag 2026-06-19_full_split_smoke_eval
```

已完成完整 7-sample diagnostic evaluation：

```text
python -X utf8 submission_planning/tools/evaluate_ablation_full_split_metrics.py --tag 2026-06-19_full_split_debug_eval
```

已新增 full-split eligibility audit：

```text
submission_planning/tools/audit_ablation_full_split_eligibility.py
tmp/ablation_results/eligibility_audits/ABL_full_split_eligibility.md
```

## 4. 结果摘要

| Run | Variant | Mean MAE um | Mean Edge MAE um | Mean High-Risk MAE um |
|---|---|---:|---:|---:|
| ABL-00 | Full S2R-FocusNet | 272.7972 | 193.5823 | 334.8069 |
| ABL-02 | w/o DFF/GADFF prior | 336.3451 | 224.5025 | 400.5218 |
| ABL-03 | w/o focal difference | 305.3061 | 206.9209 | 353.8419 |
| ABL-04 | w/o glare cue | 282.7030 | 193.0186 | 333.3346 |

该结果显示 full-split evaluation runner 已经具备计算主消融指标的能力，但训练来源仍是 P10 tiny pilot checkpoint，因此只能作为下一步正式消融训练的规划依据。

## 5. 对研究计划的更新

上一轮断点 R30 和 R31 的诊断版本已完成：full-split metrics runner 已可运行，full-split eligibility audit 已生成。研究计划需要进一步细分为：

```text
R33: Full-split ablation training runner
R34: ABL-00/02/03/04 matched training runs
R35: Claim-eligible ablation audit
R36: ABL-01 lower-prior training design
```

## 6. 当前结论

本轮把消融线从“能在 P10 上训练”推进到“能在完整 synthetic test split 上评价”。这是进入正式消融实验前的必要基础设施，但论文主张仍需要 matched training runs 和正式 claim eligibility audit 支撑。
