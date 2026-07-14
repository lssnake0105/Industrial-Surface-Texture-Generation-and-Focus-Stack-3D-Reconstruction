# Ablation Matched Full-Split Evaluator Smoke

日期：2026-06-19  
标签：`2026-06-19_matched_evaluator_smoke`  
checkpoint 来源：`2026-06-19_matched_training_smoke`  
范围：ABL-00、ABL-02、ABL-03、ABL-04。  
边界：本轮只验证 matched checkpoint evaluator 是否能读取指定 tag 并完成 1-sample test evaluation；该结果不进入论文主表，也不支持模块贡献结论。

## 1. 目标

上一轮 full matched configuration 已确认正式候选训练预算，但旧 evaluator 仍绑定 controlled-pilot checkpoint。本轮目标是新增独立 evaluator，验证它能按 `--checkpoint-tag` 读取 matched checkpoints，并将输出限定在 `tmp/ablation_results/` 下。

## 2. 执行命令

```text
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/evaluate_ablation_matched_full_split_metrics.py --tag 2026-06-19_matched_evaluator_smoke --checkpoint-tag 2026-06-19_matched_training_smoke --training-scope matched_smoke_training --evaluation-scope matched_full_split_evaluator_smoke --max-samples 1 --device cpu
```

## 3. 输出位置

```text
tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_evaluator_smoke/2026-06-19_matched_evaluator_smoke_summary.md
tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_evaluator_smoke/2026-06-19_matched_evaluator_smoke_summary.json
tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_evaluator_smoke/2026-06-19_matched_evaluator_smoke_per_sample_metrics.csv
tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_evaluator_smoke/2026-06-19_matched_evaluator_smoke_method_summary_metrics.csv
```

## 4. Smoke Result

```text
status = pass
samples = 1
checks = 23
errors = 0
warnings = 0
claim_eligible = false
main_table_eligible = false
```

## 5. Smoke Metrics

这些数值只证明 evaluator path 可运行，因为 checkpoint 来自 1-epoch matched-smoke training。

| Run | Variant | Samples | Mean MAE um | Mean Edge MAE um | Mean High-Risk MAE um |
|---|---|---:|---:|---:|---:|
| ABL-00 | Full S2R-FocusNet | 1 | 363.7042 | 288.5295 | 550.5137 |
| ABL-02 | w/o DFF/GADFF prior | 1 | 295.2028 | 264.9166 | 384.8727 |
| ABL-03 | w/o focal difference | 1 | 348.5844 | 269.7047 | 531.9326 |
| ABL-04 | w/o glare cue | 1 | 296.5715 | 269.6706 | 394.3848 |

## 6. 当前可支持的表述

当前可以安全表述：

1. matched evaluator 已能按 checkpoint tag 读取 ABL-00/02/03/04 checkpoints；
2. evaluator 可生成 per-sample metrics、method summary 和 run_config 记录；
3. 输出路径保持在 `tmp/ablation_results/matched_full_split_eval/`；
4. `claim_eligible=false` 与 `main_table_eligible=false` 继续保持。

当前不能安全表述：

1. full matched candidate training 已经完成；
2. ablation table 已经具备入稿证据；
3. smoke metrics 可解释模块贡献强弱；
4. matched evaluator 已完成 7-sample full evaluation。

## 7. 下一步断点

下一步可以运行 `matched_full_candidate` 训练，但训练后仍需执行 full 7-sample matched evaluator 和独立 eligibility audit。
