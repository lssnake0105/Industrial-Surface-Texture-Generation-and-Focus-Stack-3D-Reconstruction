# Ablation Matched Full Candidate Results

日期：2026-06-19  
训练标签：`2026-06-19_matched_training_full_candidate`  
评估标签：`2026-06-19_matched_full_candidate_eval`  
范围：ABL-00、ABL-02、ABL-03、ABL-04。  
边界：本结果是当前阶段 full-candidate 消融证据，可用于 supervisor 讨论和下一步实验设计；进入论文主表前仍需结合外部 SOTA、训练预算复核和 supervisor 确认。

## 1. Experimental Setup

| Item | Value |
|---|---|
| split | train 27 / validation 10 / test 7 |
| epochs | 4 |
| train patches per epoch | 128 |
| validation patches per epoch | 32 |
| batch size | 1 |
| learning rate | 0.0006 |
| patch size | 64 |
| device | cuda |
| output root | `tmp/ablation_results/` |

## 2. Training Outcome

训练已完成并通过 summary check：

```text
status = pass
runs = ABL-00, ABL-02, ABL-03, ABL-04
checks = 52
errors = 0
warnings = 0
```

| Run | Variant | Last Val MAE Norm | Checkpoint |
|---|---|---:|---|
| ABL-00 | Full S2R-FocusNet | 0.15679009 | `tmp/ablation_results/ABL-00/checkpoints/2026-06-19_matched_training_full_candidate.pt` |
| ABL-02 | w/o DFF/GADFF prior | 0.15910939 | `tmp/ablation_results/ABL-02/checkpoints/2026-06-19_matched_training_full_candidate.pt` |
| ABL-03 | w/o focal difference | 0.13092036 | `tmp/ablation_results/ABL-03/checkpoints/2026-06-19_matched_training_full_candidate.pt` |
| ABL-04 | w/o glare cue | 0.11970254 | `tmp/ablation_results/ABL-04/checkpoints/2026-06-19_matched_training_full_candidate.pt` |

## 3. Full Test-Split Evaluation

7-sample evaluator 已完成并通过 check：

```text
status = pass
samples = 7
per-sample rows = 28
summary rows = 4
checks = 23
errors = 0
warnings = 0
```

| Run | Variant | Samples | Mean MAE um | Mean Edge MAE um | Mean High-Risk MAE um | Mean P90 Norm |
|---|---|---:|---:|---:|---:|---:|
| ABL-00 | Full S2R-FocusNet | 7 | 130.9028 | 183.7727 | 117.9743 | 0.2393 |
| ABL-02 | w/o DFF/GADFF prior | 7 | 245.3440 | 233.6410 | 261.3550 | 0.3846 |
| ABL-03 | w/o focal difference | 7 | 113.1038 | 161.9272 | 103.4093 | 0.1970 |
| ABL-04 | w/o glare cue | 7 | 111.8795 | 155.0823 | 110.4368 | 0.1937 |

## 4. Eligibility Audit

已新增并运行：

```text
submission_planning/tools/audit_ablation_matched_training_eligibility.py
```

审计结果：

```text
status = pass
eligibility = Current-stage matched ablation evidence; manuscript table candidate pending supervisor review and external-baseline context
checks = 104
errors = 0
warnings = 0
```

## 5. Main Interpretation

当前结果最明确支持的是：DFF/GADFF prior 对当前任务仍是关键锚点。移除 DFF/GADFF prior 后，Mean MAE 从 130.9028 um 上升到 245.3440 um，high-risk MAE 也从 117.9743 um 上升到 261.3550 um，说明显式 focus-derived prior 能显著降低网络在小数据训练下的学习负担。

更需要谨慎解释的是：在当前 4-epoch candidate budget 下，w/o focal difference 与 w/o glare cue 的 test MAE 低于 full variant。这提示 full model 可能存在辅助通道噪声、短训练预算下的优化不稳定、loss 对 high-risk 区域约束不足，或 channel fusion 未充分校准。该结果应作为模型设计反馈，而不应直接写成 focal-difference 或 glare cue 无效。

## 6. Supervisor Discussion Points

1. 当前 full model 可能需要更长训练、更稳定 learning-rate schedule 或 channel attention / gating 机制，才能把 focal-difference 和 glare cue 转化为稳定收益。
2. DFF/GADFF prior 的消融结果很强，可作为论文方法主线中的核心实验支撑。
3. focal-difference 和 glare cue 当前更适合写成 pending refinement：它们是合理的物理信号，但需要重新设计融合层级、loss 权重或 confidence gate。
4. 下一步应优先做重复 seed / longer budget，以确认 ABL-03/04 的优势是否来自真实泛化改善，还是短训练预算下的随机性和正则化效应。

## 7. Next Prospect

| Direction | Purpose | Concrete Action |
|---|---|---|
| longer-budget repeat | 验证 full model 是否训练不足 | 8-12 epochs、相同 split、至少 2 seeds |
| channel-gated full model | 降低辅助通道噪声 | 对 focal-difference / glare cue 引入 learnable gate 或 confidence weighting |
| loss refinement | 提升 high-risk 区域稳定性 | 对 edge / high-risk mask 增加 loss weight 或 consistency term |
| external SOTA | 支撑投稿公平比较 | 跑 DFV 或 DDFFNet 的 fixed synthetic split 预测 |
| real validation | 强化 sim-to-real 说服力 | 保持 no-reference 指标，争取 calibrated real subset |
