# 恢复断点：Matched Smoke 到 Full Matched Configuration

日期：2026-06-19  
断点名称：`ABL matched smoke -> full matched configuration`  
恢复目标：在保持 claim gate 关闭的前提下，继续推进正式 matched ablation training 的配置、评估和 eligibility gate。

## 1. 当前最后稳定状态

当前最后稳定产物是 ABL-00/02/03/04 matched training smoke：

```text
submission_planning/autonomous_research/ablation_matched_training_smoke.md
submission_planning/autonomous_research/research_log_2026-06-19_ablation_matched_training_smoke.md
tmp/ablation_results/training_runner_matched_smoke/2026-06-19_matched_training_smoke_summary.md
tmp/ablation_results/training_runner_matched_smoke/2026-06-19_matched_training_smoke_summary.json
tmp/ablation_results/eligibility_audits/ABL_matched_smoke_eligibility.md
tmp/ablation_results/eligibility_audits/ABL_matched_smoke_eligibility.json
```

当前 smoke 结果：

```text
status = pass
checks = 52
errors = 0
warnings = 0
claim_eligible = false
main_table_eligible = false
```

当前 eligibility audit：

```text
status = pass
eligibility_level = Matched smoke only; not manuscript ablation evidence
checks = 78
errors = 0
warnings = 0
```

## 2. 当前 split 与 smoke 设置

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
| patch size | 64 |

当前 smoke checkpoint：

```text
tmp/ablation_results/ABL-00/checkpoints/2026-06-19_matched_training_smoke.pt
tmp/ablation_results/ABL-02/checkpoints/2026-06-19_matched_training_smoke.pt
tmp/ablation_results/ABL-03/checkpoints/2026-06-19_matched_training_smoke.pt
tmp/ablation_results/ABL-04/checkpoints/2026-06-19_matched_training_smoke.pt
```

## 3. 当前结果解释

当前只允许支持以下判断：

1. ABL-00/02/03/04 可以在同一 split 计数边界下完成最小训练链路；
2. 每个变体的 checkpoint、history 和 log 都位于 `tmp/ablation_results/`；
3. `claim_eligible=false` 和 `main_table_eligible=false` 已由 matched smoke eligibility audit 保护。

当前不能支持以下判断：

1. matched full training 已完成；
2. 任何模块贡献已被验证；
3. smoke 数值可以进入论文主表；
4. smoke checkpoint 可以作为 final ablation checkpoint。

## 4. 恢复前先检查

恢复实验前先运行：

```text
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/audit_research_package_integrity.py
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/audit_manuscript_claim_safety.py
```

预期状态：

```text
Research package audit: pass
Manuscript claim audit: pass
```

## 5. 下一步应先做的设计项

下一步不要直接扩大训练命令。应先生成 full matched ablation training configuration，至少明确：

1. 是否使用全部 27 个 train samples 和 10 个 validation samples；
2. `max_epochs`、`train_patches`、`val_patches`、`batch_size`、learning rate 和 seed；
3. checkpoint tag 是否使用 `2026-06-19_matched_training_full` 或后续日期；
4. test evaluator 是否复用现有 full-split evaluator，或新增 matched checkpoint evaluator；
5. full matched run 的 eligibility audit 需要哪些字段才允许后续进入论文表格。

## 6. 下一组建议任务

```text
R41: Full matched ablation training configuration
R42: Matched checkpoint full-split evaluator
R43: ABL-00/02/03/04 full matched runs
R44: Full matched ablation eligibility audit
```

## 7. 不能跨过的边界

1. 不把 matched smoke validation MAE norm 写入论文主表；
2. 不把 smoke checkpoint 用作正式消融 checkpoint；
3. 不在 `src/`、`论文与PPT制作项目包/` 或原始结果目录写入新实验产物；
4. 不覆盖 `2026-06-19_controlled_pilot`、`2026-06-19_small_training_debug` 或 `2026-06-19_matched_training_smoke` 的已保存证据；
5. 不把 ABL-01 放入当前 38-channel masking runner；
6. 不把 DFV/DDFFNet 写成已完成外部 SOTA 对比。

## 8. 恢复时的一句话判断

如果从这里继续，第一步应写 full matched ablation training configuration；第二步定义 matched checkpoint evaluator；第三步再运行正式 matched runs 和 eligibility audit。
