# 恢复断点：Full Candidate Eval 后的下一步

日期：2026-06-19  
断点名称：`matched full-candidate eval -> longer-budget repeat / external SOTA`  
恢复目标：基于当前 full-candidate 消融结果，选择 longer-budget repeat、模型融合改进或 DFV/DDFFNet 外部 SOTA 路线。

## 1. 当前最后稳定状态

当前最后稳定产物：

```text
tmp/ablation_results/training_runner_matched_full_candidate/2026-06-19_matched_training_full_candidate_summary.md
tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_full_candidate_eval/2026-06-19_matched_full_candidate_eval_summary.md
tmp/ablation_results/eligibility_audits/ABL_matched_training_eligibility.md
submission_planning/autonomous_research/ablation_matched_full_candidate_results.md
submission_planning/autonomous_research/supervisor_update_2026-06-19.md
```

当前审计结果：

```text
training status = pass
evaluation status = pass
eligibility audit status = pass
research package audit = pending rerun after documentation update
```

## 2. 当前结果摘要

| Variant | Mean MAE um | Mean Edge MAE um | Mean High-Risk MAE um |
|---|---:|---:|---:|
| Full S2R-FocusNet | 130.9028 | 183.7727 | 117.9743 |
| w/o DFF/GADFF prior | 245.3440 | 233.6410 | 261.3550 |
| w/o focal difference | 113.1038 | 161.9272 | 103.4093 |
| w/o glare cue | 111.8795 | 155.0823 | 110.4368 |

## 3. 当前判断

1. DFF/GADFF prior 的贡献最清楚，移除后整体误差和 high-risk 误差显著升高；
2. full model 在当前 4-epoch candidate budget 下没有最优；
3. focal-difference 和 glare cue 的作用需要用 longer-budget repeat、seed repeat 或 gated fusion 继续判断；
4. 当前结果可用于 supervisor update，但论文主表使用前应先完成更稳健的复核。

## 4. 下一步建议

优先路线：

```text
R50: Longer-budget matched repeat for ABL-00/02/03/04
R51: Seed repeat for current matched candidate
R52: Gated auxiliary-signal fusion design
R53: DFV repository inventory and P10 prediction contract
```

## 5. 恢复前先检查

```text
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/audit_research_package_integrity.py
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/audit_manuscript_claim_safety.py
```

## 6. 恢复时的一句话判断

如果 supervisor 更关心模型贡献，下一步优先做 longer-budget repeat 和 gated fusion；如果 supervisor 更关心投稿完整性，下一步优先做 DFV/DDFFNet 外部 SOTA。
