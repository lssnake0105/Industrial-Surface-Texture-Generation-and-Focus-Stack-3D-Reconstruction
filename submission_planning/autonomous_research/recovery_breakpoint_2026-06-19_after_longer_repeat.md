# 恢复断点：Longer Repeat 后的下一步

日期：2026-06-19  
断点名称：`matched longer repeat -> gated fusion / seed repeat / external SOTA`  
恢复目标：基于 longer-budget repeat 的结果，决定下一步是做 gated auxiliary fusion、seed repeat，还是转向 DFV/DDFFNet 外部 SOTA。

## 1. 当前最后稳定状态

当前最后稳定产物：

```text
tmp/ablation_results/training_runner_matched_longer_repeat/2026-06-19_matched_training_longer_repeat_summary.md
tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_longer_repeat_eval/2026-06-19_matched_longer_repeat_eval_summary.md
submission_planning/autonomous_research/ablation_matched_longer_repeat_results.md
submission_planning/autonomous_research/research_log_2026-06-19_ablation_longer_repeat.md
submission_planning/autonomous_research/supervisor_update_2026-06-19.md
```

## 2. 当前结果摘要

| Variant | Mean MAE um | Mean Edge MAE um | Mean High-Risk MAE um |
|---|---:|---:|---:|
| Full S2R-FocusNet | 109.2209 | 153.0310 | 86.6455 |
| w/o DFF/GADFF prior | 133.4808 | 181.9107 | 121.3387 |
| w/o focal difference | 90.4542 | 158.5932 | 57.2526 |
| w/o glare cue | 75.4572 | 126.8816 | 60.7381 |

## 3. 当前判断

1. 更长训练预算改善了所有变体；
2. DFF/GADFF prior 的贡献仍然稳定；
3. full model 仍未优于 w/o focal difference 和 w/o glare cue；
4. 问题更可能来自 auxiliary cue fusion 或 cue noise，而非单纯训练不足。

## 4. 下一步建议

```text
R51: Seed repeat for ABL-00/03/04
R52: Gated auxiliary-signal fusion design
R53: DFV repository inventory and P10 prediction contract
R54: Glare cue quality audit
```

## 5. 恢复前先检查

```text
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/audit_research_package_integrity.py
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/audit_manuscript_claim_safety.py
```

## 6. 恢复时的一句话判断

如果下一轮继续模型实验，优先设计 gated auxiliary-signal fusion；如果下一轮继续投稿闭环，优先推进 DFV/DDFFNet 外部 SOTA。
