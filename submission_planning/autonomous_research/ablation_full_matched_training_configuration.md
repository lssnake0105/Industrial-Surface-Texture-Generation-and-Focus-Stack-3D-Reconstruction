# Ablation Full Matched Training Configuration

日期：2026-06-19  
标签：`2026-06-19_matched_training_full_candidate`  
范围：ABL-00、ABL-02、ABL-03、ABL-04。  
边界：本轮只完成 full matched training configuration preflight；没有运行 full training，没有生成 full checkpoint、prediction、test metrics 或 manuscript-eligible ablation result。

## 1. 目标

上一轮 matched smoke 已验证四个核心变体可以在 fixed split boundary 下完成最小训练链路。本轮目标是先固定正式 matched ablation training 的候选预算、输出命名、evaluator 需求和 eligibility gate，避免直接扩大训练命令后产生难以解释的半正式结果。

## 2. 执行命令

```text
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/preflight_ablation_full_matched_configuration.py
```

## 3. 输出位置

```text
tmp/ablation_results/matched_training_full_config/2026-06-19_matched_training_full_candidate_config_preflight.md
tmp/ablation_results/matched_training_full_config/2026-06-19_matched_training_full_candidate_config_preflight.json
```

## 4. Candidate Training Configuration

| Field | Value |
|---|---|
| tag | `2026-06-19_matched_training_full_candidate` |
| seed | 20260619 |
| patch size | 64 |
| epochs | 4 |
| train patches per epoch | 128 |
| validation patches per epoch | 32 |
| batch size | 1 |
| learning rate | 0.0006 |
| train samples | all 27 |
| validation samples | all 10 |
| test samples | 7 |
| run kind | `matched_full_candidate` |
| output scope | `tmp/ablation_results/<run_id>/` |
| claim eligible after training | false |
| main table eligible after training | false |

该配置是第一轮 full matched candidate，不等同于最终论文训练预算。它的作用是从 smoke 进入更完整的训练候选，同时继续把入稿资格交给 full-split evaluator 和 eligibility audit。

## 5. Variant Plans

| Run | Variant | Zero channels | Planned checkpoint |
|---|---|---|---|
| ABL-00 | Full S2R-FocusNet | none | `tmp/ablation_results/ABL-00/checkpoints/2026-06-19_matched_training_full_candidate.pt` |
| ABL-02 | w/o DFF/GADFF prior | 34-37 | `tmp/ablation_results/ABL-02/checkpoints/2026-06-19_matched_training_full_candidate.pt` |
| ABL-03 | w/o focal difference | 17-32 | `tmp/ablation_results/ABL-03/checkpoints/2026-06-19_matched_training_full_candidate.pt` |
| ABL-04 | w/o glare cue | 33 | `tmp/ablation_results/ABL-04/checkpoints/2026-06-19_matched_training_full_candidate.pt` |

## 6. Evaluator Plan

当前 `evaluate_ablation_full_split_metrics.py` 仍绑定 controlled-pilot checkpoint lookup 和 `controlled_pilot_p10_debug` training scope。因此 full matched candidate 训练后不能直接复用当前 evaluator 作为正式评估工具。

下一步应新增或改造 evaluator，使其支持：

1. `--checkpoint-tag 2026-06-19_matched_training_full_candidate`；
2. `--training-scope matched_full_candidate_train_validation_split`；
3. `--evaluation-scope matched_full_candidate_test_split_eval`；
4. 输出 28 行 per-sample metrics 和 4 行 method summary；
5. 指标至少包含 MAE、edge MAE、high-risk MAE 和 P90。

## 7. Eligibility Plan

full matched 结果进入论文前必须新增 `audit_ablation_matched_training_eligibility.py`，至少检查：

1. ABL-00/02/03/04 full candidate checkpoint 均存在；
2. 每个 run 都有 history CSV/JSON 和 run log；
3. training config 记录 train=27、validation=10、test=7；
4. full-split evaluator 产生 28 行 per-sample metrics 和 4 行 summary；
5. run config 在 audit 前仍保持 `claim_eligible=false`；
6. eligibility audit 明确输出是否允许后续进入论文表格。

## 8. 当前可支持的表述

当前可以安全表述：

1. full matched training 的候选预算和输出命名已经固定；
2. runner 已支持 `matched_full_candidate` 模式；
3. ABL-00/02/03/04 均通过 configuration preflight；
4. full matched training 仍需 evaluator 和 eligibility audit 后才能讨论入稿。

当前不能安全表述：

1. full matched training 已经完成；
2. full matched checkpoint 已经存在；
3. full matched metrics 已经可用；
4. 模块贡献已被验证。

## 9. 下一步断点

下一步应先实现 matched checkpoint full-split evaluator。建议任务顺序：

1. 新增 `evaluate_ablation_matched_full_split_metrics.py`，支持 `--checkpoint-tag` 和 `--training-scope`；
2. 先对 `2026-06-19_matched_training_smoke` checkpoint 做 1-sample evaluator smoke，验证评估器不绑定 controlled pilot；
3. 再运行 `matched_full_candidate` 训练；
4. 最后执行 full matched evaluator 和 eligibility audit。
