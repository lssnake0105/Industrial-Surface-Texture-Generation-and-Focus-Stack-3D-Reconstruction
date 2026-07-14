# 本轮自主研究日志：ABL Longer-Budget Repeat

日期：2026-06-19  
范围：`submission_planning/tools/`、`submission_planning/autonomous_research/`、`tmp/ablation_results/`。  
边界：本轮新增 longer-budget repeat lane，产物全部位于 `tmp/ablation_results/` 和 `submission_planning/`，未改写 `src/` 或原项目交付包。

## 1. 本轮目标

上一轮 full-candidate ablation 显示 full model 没有优于 w/o focal difference 和 w/o glare cue。为了判断这是否来自训练预算过短，本轮新增 `matched_longer_repeat` run kind，并使用更长训练预算复跑 ABL-00/02/03/04。

## 2. 已完成任务

已新增 runner mode：

```text
matched_longer_repeat
tag = 2026-06-19_matched_training_longer_repeat
```

已完成 longer-budget training：

```text
epochs = 8
train patches per epoch = 192
validation patches per epoch = 48
status = pass
checks = 52
errors = 0
warnings = 0
```

已完成 7-sample evaluator：

```text
tag = 2026-06-19_matched_longer_repeat_eval
samples = 7
per-sample rows = 28
summary rows = 4
status = pass
checks = 23
errors = 0
warnings = 0
```

## 3. 关键结果

| Variant | Candidate MAE um | Longer MAE um | Interpretation |
|---|---:|---:|---|
| Full S2R-FocusNet | 130.9028 | 109.2209 | longer training improves full model |
| w/o DFF/GADFF prior | 245.3440 | 133.4808 | prior removal still hurts |
| w/o focal difference | 113.1038 | 90.4542 | still better than full |
| w/o glare cue | 111.8795 | 75.4572 | still best in this repeat |

## 4. 计划修正

本轮结果显示 full model 未占优并不只是 4-epoch 训练预算不足。更合理的下一步是把问题定位到 auxiliary signal fusion：focal-difference 和 glare cue 可能需要 learnable gating、confidence weighting、denoising 或重新设计 loss。

## 5. 当前结论

DFF/GADFF prior 的贡献在 longer repeat 中仍然成立。focal-difference 和 glare cue 的当前融合方式不稳定，下一步应优先做 gated fusion 或 seed repeat，同时继续推进 DFV/DDFFNet 外部 SOTA。
