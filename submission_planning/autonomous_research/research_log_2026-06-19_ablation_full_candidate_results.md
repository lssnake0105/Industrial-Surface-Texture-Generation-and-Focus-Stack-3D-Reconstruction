# 本轮自主研究日志：ABL Full Candidate Training and Supervisor Update

日期：2026-06-19  
范围：`submission_planning/tools/`、`submission_planning/autonomous_research/`、`tmp/ablation_results/`。  
边界：本轮运行 ABL-00/02/03/04 matched full-candidate training、7-sample evaluator 和 eligibility audit；产物全部位于 `tmp/ablation_results/` 与 `submission_planning/`，未改写 `src/` 或原项目交付包。

## 1. 本轮目标

上一轮 matched evaluator smoke 已通过，说明评估器可以读取指定 checkpoint tag。本轮目标是跑完当前 full-candidate 消融训练，执行完整 7-sample evaluator，并把结果整理为 supervisor update。

## 2. 已完成任务

已完成 ABL-00/02/03/04 `matched_full_candidate` 训练：

```text
status = pass
checks = 52
errors = 0
warnings = 0
```

已完成 7-sample matched evaluator：

```text
status = pass
samples = 7
per-sample rows = 28
summary rows = 4
checks = 23
errors = 0
warnings = 0
```

已完成 matched training eligibility audit：

```text
status = pass
checks = 104
errors = 0
warnings = 0
```

## 3. 关键结果

| Variant | Mean MAE um | Mean Edge MAE um | Mean High-Risk MAE um |
|---|---:|---:|---:|
| Full S2R-FocusNet | 130.9028 | 183.7727 | 117.9743 |
| w/o DFF/GADFF prior | 245.3440 | 233.6410 | 261.3550 |
| w/o focal difference | 113.1038 | 161.9272 | 103.4093 |
| w/o glare cue | 111.8795 | 155.0823 | 110.4368 |

## 4. 计划修正

本轮结果强化了 DFF/GADFF prior 的必要性，但也显示 full model 当前没有稳定优于去掉 focal-difference 或 glare cue 的变体。下一步不应直接把 full model 写成模块全部有效，而应将其解释为：prior-guided route 成立，辅助信号融合方式需要二次设计和更长训练预算验证。

## 5. 产出文件

```text
submission_planning/autonomous_research/ablation_matched_full_candidate_results.md
submission_planning/autonomous_research/supervisor_update_2026-06-19.md
submission_planning/autonomous_research/recovery_breakpoint_2026-06-19_after_full_candidate_eval.md
tmp/ablation_results/eligibility_audits/ABL_matched_training_eligibility.md
```

## 6. 当前结论

当前最适合向 supervisor 汇报的结论是：项目已经具备从原型到投稿研究的初步实验闭环；DFF/GADFF prior 是当前最稳定的贡献；focal-difference 和 glare cue 需要通过更好的融合策略、loss 设计和 longer-budget repeat 继续验证。
