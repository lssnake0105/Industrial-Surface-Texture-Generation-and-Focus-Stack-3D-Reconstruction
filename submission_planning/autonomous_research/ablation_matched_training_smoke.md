# Ablation Matched Training Smoke

日期：2026-06-19  
标签：`2026-06-19_matched_training_smoke`  
范围：ABL-00、ABL-02、ABL-03、ABL-04。  
边界：本轮是 matched split smoke training，只验证受保护 runner 能在统一 split 边界下完成 optimizer/backward/checkpoint/history/log 链路；不产生 test prediction、test metrics 或 manuscript-eligible ablation result。

## 1. 目标

上一轮 matched training preflight 已验证四个核心变体可以共享 train/validation/test split，并能完成 forward/loss 检查。本轮进一步验证正式 matched training 前的最小训练链路：同一 split 计数、同一 tag、同一输出命名、同一 claim gate，以及 checkpoint/history/log 是否全部限制在 `tmp/ablation_results/<run_id>/`。

## 2. 执行命令

```text
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/run_ablation_variant_training.py --execute-training --run-kind matched_smoke --tag 2026-06-19_matched_training_smoke --run-id ABL-00 --run-id ABL-02 --run-id ABL-03 --run-id ABL-04 --max-epochs 1 --train-patches 2 --val-patches 1 --batch-size 1 --max-train-samples 2 --max-val-samples 1
```

## 3. 输出位置

```text
tmp/ablation_results/training_runner_matched_smoke/2026-06-19_matched_training_smoke_summary.md
tmp/ablation_results/training_runner_matched_smoke/2026-06-19_matched_training_smoke_summary.json
tmp/ablation_results/eligibility_audits/ABL_matched_smoke_eligibility.md
tmp/ablation_results/eligibility_audits/ABL_matched_smoke_eligibility.json
```

每个 run 的单独产物位于：

```text
tmp/ablation_results/<run_id>/checkpoints/2026-06-19_matched_training_smoke.pt
tmp/ablation_results/<run_id>/metrics/2026-06-19_matched_training_history.csv
tmp/ablation_results/<run_id>/metrics/2026-06-19_matched_training_history.json
tmp/ablation_results/<run_id>/logs/2026-06-19_matched_training_smoke.md
tmp/ablation_results/<run_id>/logs/2026-06-19_matched_training_smoke.json
```

## 4. Smoke 设置

| Field | Value |
|---|---|
| train split count | 27 |
| validation split count | 10 |
| test split count | 7 |
| prepared train samples | 2 |
| prepared validation samples | 1 |
| epochs | 1 |
| train patches | 2 |
| validation patches | 1 |
| batch size | 1 |
| learning rate | 0.0006 |
| patch size | 64 |
| output scope | `tmp/ablation_results/<run_id>/` |

该设置刻意保持极小，只用于验证 matched training runner，不用于比较模块贡献。

## 5. Smoke 结果

| Run | Variant | Zero channels | Last val MAE norm debug | Checkpoint |
|---|---|---|---:|---|
| ABL-00 | Full S2R-FocusNet | none | 0.10953103 | `tmp/ablation_results/ABL-00/checkpoints/2026-06-19_matched_training_smoke.pt` |
| ABL-02 | w/o DFF/GADFF prior | 34-37 | 0.28376535 | `tmp/ablation_results/ABL-02/checkpoints/2026-06-19_matched_training_smoke.pt` |
| ABL-03 | w/o focal difference | 17-32 | 0.15884387 | `tmp/ablation_results/ABL-03/checkpoints/2026-06-19_matched_training_smoke.pt` |
| ABL-04 | w/o glare cue | 33 | 0.24383156 | `tmp/ablation_results/ABL-04/checkpoints/2026-06-19_matched_training_smoke.pt` |

检查结果：

```text
Ablation matched training smoke: pass
Checks: 52, errors: 0, warnings: 0
```

eligibility audit：

```text
Ablation matched smoke eligibility audit: pass
Eligibility: Matched smoke only; not manuscript ablation evidence
Checks: 78, errors: 0, warnings: 0
```

## 6. 当前可支持的表述

当前可以安全表述：

1. ABL-00/02/03/04 已能在受保护 runner 中完成 matched split smoke training；
2. 四个变体均生成 checkpoint、history 和 run log，且全部位于 `tmp/ablation_results/`；
3. run config、summary 和 eligibility audit 均保持 `claim_eligible=false` 与 `main_table_eligible=false`。

当前不能安全表述：

1. matched full training 已完成；
2. ABL-00/02/03/04 已产生可入稿消融结果；
3. 当前 smoke 数值说明某个模块有效；
4. 当前 smoke checkpoint 可用于论文主表或最终图表。

## 7. 下一步断点

下一步应进入 full matched ablation training configuration，而非直接扩大命令。建议先补一个正式训练配置文件或 preflight：

1. 固定 full matched training 的 `max_epochs`、`train_patches`、`val_patches`、seed 和 batch size；
2. 明确是否使用全部 27 个 train samples 和 10 个 validation samples；
3. 明确 test split evaluation runner 是否复用 `evaluate_ablation_full_split_metrics.py` 或新增 matched checkpoint evaluator；
4. 新增 `ABL_matched_training_eligibility.py`，只在 full train、test metrics、run logs 和 claim boundary 全部齐全时允许后续进入论文表格。
