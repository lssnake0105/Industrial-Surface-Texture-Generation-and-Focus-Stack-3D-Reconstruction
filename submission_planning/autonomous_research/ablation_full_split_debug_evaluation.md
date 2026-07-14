# Ablation Full-Split Debug Evaluation

日期：2026-06-19  
标签：`2026-06-19_full_split_debug_eval`  
范围：ABL-00、ABL-02、ABL-03、ABL-04；7 个 synthetic test samples。  
边界：本轮只评估 controlled-pilot checkpoints 在完整 test split 上的诊断指标；由于 checkpoints 只来自 P10 tiny pilot training，结果不能作为论文消融结论。

## 1. 执行目标

上一轮 controlled pilot 证明 ABL-00/02/03/04 可以训练并写入临时产物。本轮进一步验证这些 checkpoint 是否能在固定 synthetic test split 上完成 tiled inference 和统一指标计算。该步骤的价值是检查 full-split evaluator、per-sample metrics 和 run_config 记录链路，仍不等同于正式消融实验。

## 2. 执行命令

```text
python -X utf8 submission_planning/tools/evaluate_ablation_full_split_metrics.py --tag 2026-06-19_full_split_debug_eval
```

## 3. 输出位置

```text
tmp/ablation_results/full_split_debug_eval/2026-06-19_full_split_debug_eval_summary.md
tmp/ablation_results/full_split_debug_eval/2026-06-19_full_split_debug_eval_per_sample_metrics.csv
tmp/ablation_results/full_split_debug_eval/2026-06-19_full_split_debug_eval_method_summary_metrics.csv
tmp/ablation_results/eligibility_audits/ABL_full_split_eligibility.md
```

每个 run 也写入：

```text
tmp/ablation_results/<run_id>/metrics/2026-06-19_full_split_debug_eval_metrics.csv
tmp/ablation_results/<run_id>/metrics/2026-06-19_full_split_debug_eval_metrics.json
```

## 4. Full-split 诊断指标

| Run | Variant | Samples | Mean MAE um | Mean Edge MAE um | Mean High-Risk MAE um |
|---|---|---:|---:|---:|---:|
| ABL-00 | Full S2R-FocusNet | 7 | 272.7972 | 193.5823 | 334.8069 |
| ABL-02 | w/o DFF/GADFF prior | 7 | 336.3451 | 224.5025 | 400.5218 |
| ABL-03 | w/o focal difference | 7 | 305.3061 | 206.9209 | 353.8419 |
| ABL-04 | w/o glare cue | 7 | 282.7030 | 193.0186 | 333.3346 |

这些数值说明评估链路已覆盖 7 个 test samples，且当前 pilot checkpoint 的 full model 在 mean MAE 上低于三个去模块变体。但由于训练只来自单样本小规模 pilot，不能把该现象写成模块有效性的论文证据。

## 5. Eligibility audit

`audit_ablation_full_split_eligibility.py` 结果为：

```text
Ablation full-split eligibility audit: pass
Eligibility: Diagnostic full-split metrics only; not manuscript ablation evidence
Checks: 40, errors: 0, warnings: 0
```

四个 run 均保持：

```text
claim_eligible = false
main_table_eligible = false
```

## 6. 当前可支持的表述

当前可以安全表述：

1. full-split evaluator 已能加载 controlled-pilot checkpoints 并覆盖 7 个 synthetic test samples。
2. per-sample MAE、edge MAE、high-risk MAE 和 method summary 均已写入 `tmp/ablation_results/`。
3. full-split eligibility audit 已确认这些指标仅作为 diagnostic planning evidence。

当前不能安全表述：

1. ABL-00 在正式消融中优于 ABL-02/03/04。
2. DFF/GADFF prior、focal difference 或 glare cue 的贡献已经被证明。
3. 本轮 full-split debug metrics 可进入论文消融表。

## 7. 下一步断点

下一步要把消融推进到 claim-eligible，需要训练层面也覆盖完整 split 或至少固定 train/validation/test 策略，而不只评估 P10 pilot checkpoint。建议下一步为：

1. 设计 `full_split_ablation_training_runner`；
2. 每个 run 使用同一 train/validation split 和同一 epoch/patch/seed 规则；
3. 保存训练历史、validation curve、test per-sample metrics；
4. 新增正式 claim eligibility audit；
5. 单独处理 ABL-01 lower-prior architecture。
