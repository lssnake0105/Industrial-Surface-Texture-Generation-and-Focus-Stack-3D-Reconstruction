# Ablation Matched Training Preflight

日期：2026-06-19  
标签：`2026-06-19_matched_training_preflight`  
范围：ABL-00、ABL-02、ABL-03、ABL-04。  
边界：本轮只验证 matched full-split training runner 的训练前条件；没有 optimizer step、checkpoint、test prediction 或 manuscript-eligible result。

## 1. 目标

上一轮 full-split diagnostic evaluation 证明 controlled-pilot checkpoints 可以在 7 个 synthetic test samples 上完成评价，但训练来源仍然不足。本轮目标是检查正式 matched training 的最低前置条件：同一 train/validation/test split、同一训练超参数、同一输出目录策略、同一 feature masking 逻辑，以及四个核心变体的 train/validation forward loss 是否可运行。

## 2. 执行命令

```text
python -X utf8 submission_planning/tools/preflight_ablation_matched_training.py
```

## 3. 输出位置

```text
tmp/ablation_results/matched_training_preflight/2026-06-19_matched_training_preflight.md
tmp/ablation_results/matched_training_preflight/2026-06-19_matched_training_preflight.json
```

## 4. Matched training plan

| Field | Value |
|---|---|
| train split count | 27 |
| validation split count | 10 |
| test split count | 7 |
| seed | 20260619 |
| patch size | 64 |
| default epochs | 2 |
| default train patches per epoch | 32 |
| default val patches per epoch | 8 |
| default batch size | 1 |
| default learning rate | 0.0006 |
| output scope | `tmp/ablation_results/<run_id>/` |

这些参数是下一步 smoke/matched training 的默认轻量设置，不等于最终论文训练配置。

## 5. Preflight 结果

| Run | Variant | Trainable | Zero channels | Train loss diagnostic | Val loss diagnostic |
|---|---|---|---|---:|---:|
| ABL-00 | Full S2R-FocusNet | true | none | 0.50354838 | 0.48970196 |
| ABL-02 | w/o DFF/GADFF prior | true | 34-37 | 0.56292027 | 0.57115078 |
| ABL-03 | w/o focal difference | true | 17-32 | 0.47684631 | 0.45476335 |
| ABL-04 | w/o glare cue | true | 33 | 0.40210730 | 0.34177950 |

检查结果：

```text
Ablation matched training preflight: pass
Checks: 50, errors: 0, warnings: 0
```

## 6. 当前可支持的表述

当前可以安全表述：

1. ABL-00/02/03/04 在同一 train/validation/test split 定义下具备 matched training 的前向和 loss 条件。
2. 四个变体的 upgraded 38-channel features 与 channel masking 均可在 train/validation 样本上运行。
3. 未来 matched training 的输出路径已经限定到 `tmp/ablation_results/<run_id>/`。

当前不能安全表述：

1. matched training 已经完成；
2. ABL-00/02/03/04 已经具备可入稿消融结果；
3. 任何模块贡献已经由本轮 preflight 证明。

## 7. 下一步断点

下一步应新增 matched training runner 的显式 smoke mode。建议先运行：

1. ABL-00/02/03/04；
2. 1 epoch；
3. 8-16 train patches；
4. 4 validation patches；
5. 只保存 smoke checkpoint、training history 和 run log；
6. `claim_eligible=false` 与 `main_table_eligible=false` 保持关闭。
