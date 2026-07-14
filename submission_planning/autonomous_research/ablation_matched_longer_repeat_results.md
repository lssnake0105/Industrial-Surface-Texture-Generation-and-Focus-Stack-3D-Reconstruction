# Ablation Matched Longer-Budget Repeat Results

日期：2026-06-19  
训练标签：`2026-06-19_matched_training_longer_repeat`  
评估标签：`2026-06-19_matched_longer_repeat_eval`  
范围：ABL-00、ABL-02、ABL-03、ABL-04。  
边界：本结果用于判断 4-epoch full candidate 中 full model 未占优是否主要来自训练预算不足；所有产物仍位于 `tmp/ablation_results/`。

## 1. Repeat Setup

| Item | Candidate | Longer Repeat |
|---|---:|---:|
| split | 27 / 10 / 7 | 27 / 10 / 7 |
| epochs | 4 | 8 |
| train patches per epoch | 128 | 192 |
| validation patches per epoch | 32 | 48 |
| batch size | 1 | 1 |
| learning rate | 0.0006 | 0.0006 |
| device | cuda | cuda |

## 2. Training Result

```text
status = pass
runs = ABL-00, ABL-02, ABL-03, ABL-04
checks = 52
errors = 0
warnings = 0
```

| Run | Variant | Last Val MAE Norm |
|---|---|---:|
| ABL-00 | Full S2R-FocusNet | 0.10933487 |
| ABL-02 | w/o DFF/GADFF prior | 0.12403641 |
| ABL-03 | w/o focal difference | 0.11301514 |
| ABL-04 | w/o glare cue | 0.09051167 |

## 3. Full Test-Split Evaluation

```text
status = pass
samples = 7
per-sample rows = 28
summary rows = 4
checks = 23
errors = 0
warnings = 0
```

| Run | Variant | Mean MAE um | Mean Edge MAE um | Mean High-Risk MAE um | Mean P90 Norm |
|---|---:|---:|---:|---:|---:|
| ABL-00 | Full S2R-FocusNet | 109.2209 | 153.0310 | 86.6455 | 0.1832 |
| ABL-02 | w/o DFF/GADFF prior | 133.4808 | 181.9107 | 121.3387 | 0.2511 |
| ABL-03 | w/o focal difference | 90.4542 | 158.5932 | 57.2526 | 0.1878 |
| ABL-04 | w/o glare cue | 75.4572 | 126.8816 | 60.7381 | 0.1486 |

## 4. Candidate vs Longer Repeat

| Variant | Candidate MAE um | Longer MAE um | Change |
|---|---:|---:|---:|
| Full S2R-FocusNet | 130.9028 | 109.2209 | -21.6819 |
| w/o DFF/GADFF prior | 245.3440 | 133.4808 | -111.8632 |
| w/o focal difference | 113.1038 | 90.4542 | -22.6496 |
| w/o glare cue | 111.8795 | 75.4572 | -36.4223 |

## 5. Interpretation

Longer-budget repeat 改善了所有变体，但没有改变核心排序：full model 仍未优于 w/o focal difference 或 w/o glare cue。这个结果说明 4-epoch candidate 中 full model 未占优并不只是训练预算不足，更可能来自辅助信号融合方式、glare cue 噪声、loss 权重或通道校准问题。

DFF/GADFF prior 的结论仍然稳健。即使在 longer repeat 下，移除 prior 后 Mean MAE 仍从 109.2209 um 上升到 133.4808 um，high-risk MAE 从 86.6455 um 上升到 121.3387 um。

## 6. Updated Supervisor Message

当前可以向 supervisor 更新：我已经完成 longer-budget repeat。结果显示更长训练确实降低了整体误差，但 full model 仍没有超过去掉 focal-difference 或 glare cue 的变体。因此下一步的技术重点应从单纯加训练预算，转向辅助信号融合机制，例如 channel gating、confidence weighting、glare cue denoising 或重新设计 high-risk loss。

## 7. Next Step

下一步建议优先选择：

1. 设计 gated full model，把 focal-difference 和 glare cue 从固定拼接改为可学习或 confidence-aware 融合；
2. 对 ABL-00、ABL-03、ABL-04 做 seed repeat，确认排序稳定性；
3. 同步推进 DFV external baseline，保证投稿对比不只依赖内部消融。
