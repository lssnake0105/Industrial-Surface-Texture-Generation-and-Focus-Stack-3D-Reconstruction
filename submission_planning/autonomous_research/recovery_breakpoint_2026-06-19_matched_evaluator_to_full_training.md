# 恢复断点：Matched Evaluator 到 Full Candidate Training

日期：2026-06-19  
断点名称：`matched full-split evaluator smoke -> matched full candidate training`  
恢复目标：运行 ABL-00/02/03/04 的 `matched_full_candidate` 训练，并在训练后执行 7-sample matched evaluator。

## 1. 当前最后稳定状态

当前最后稳定产物是 matched evaluator smoke：

```text
submission_planning/tools/evaluate_ablation_matched_full_split_metrics.py
submission_planning/autonomous_research/ablation_matched_full_split_evaluator_smoke.md
submission_planning/autonomous_research/research_log_2026-06-19_ablation_matched_evaluator_smoke.md
tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_evaluator_smoke/2026-06-19_matched_evaluator_smoke_summary.md
tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_evaluator_smoke/2026-06-19_matched_evaluator_smoke_summary.json
```

当前 smoke 结果：

```text
status = pass
checks = 23
errors = 0
warnings = 0
claim_eligible = false
main_table_eligible = false
```

## 2. 当前可用命令

full candidate training 的计划命令为：

```text
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/run_ablation_variant_training.py --execute-training --run-kind matched_full_candidate --tag 2026-06-19_matched_training_full_candidate --run-id ABL-00 --run-id ABL-02 --run-id ABL-03 --run-id ABL-04 --max-epochs 4 --train-patches 128 --val-patches 32 --batch-size 1 --learning-rate 0.0006 --max-train-samples 0 --max-val-samples 0 --device cpu
```

训练完成后的 full evaluator 命令应为：

```text
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/evaluate_ablation_matched_full_split_metrics.py --tag 2026-06-19_matched_full_candidate_eval --checkpoint-tag 2026-06-19_matched_training_full_candidate --training-scope matched_full_candidate_train_validation_split --evaluation-scope matched_full_candidate_test_split_eval --max-samples 0 --device cpu
```

## 3. 当前不能跨过的边界

1. full matched candidate training 尚未运行；
2. full candidate checkpoint 尚不存在；
3. full 7-sample matched evaluator 尚未运行；
4. full matched candidate eligibility audit 尚未实现；
5. 任何 smoke 或 candidate 结果在 audit 前都不能进入论文主表。

## 4. 恢复前先检查

恢复实验前先运行：

```text
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/audit_research_package_integrity.py
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/audit_manuscript_claim_safety.py
```

预期状态：

```text
Research package audit: pass
Manuscript claim audit: pass
```

## 5. 恢复时的一句话判断

如果从这里继续，第一步应运行 ABL-00/02/03/04 的 full candidate training；第二步运行 full 7-sample matched evaluator；第三步新增 full matched candidate eligibility audit。
