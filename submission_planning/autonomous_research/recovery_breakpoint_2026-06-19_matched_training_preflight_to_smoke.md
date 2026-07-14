# 恢复断点：Matched Training Preflight 到 Smoke Runner

日期：2026-06-19  
断点名称：`ABL matched training preflight -> matched smoke runner`  
恢复目标：在不改写原项目资源的前提下，继续推进 ABL-00/02/03/04 的 matched training smoke runner。

## 1. 当前最后稳定状态

当前最后稳定产物是 matched training preflight：

```text
submission_planning/autonomous_research/ablation_matched_training_preflight.md
submission_planning/autonomous_research/research_log_2026-06-19_ablation_matched_training_preflight.md
tmp/ablation_results/matched_training_preflight/2026-06-19_matched_training_preflight.md
tmp/ablation_results/matched_training_preflight/2026-06-19_matched_training_preflight.json
```

当前 preflight 结果：

```text
status = pass
checks = 50
errors = 0
warnings = 0
claim_eligible = false
main_table_eligible = false
```

当前 split 计数：

| Split | Count |
|---|---:|
| train | 27 |
| validation | 10 |
| test | 7 |

当前四个核心变体均已通过 train/validation forward loss 检查：

| Run | Variant | Zero channels | Train loss diagnostic | Val loss diagnostic |
|---|---|---|---:|---:|
| ABL-00 | Full S2R-FocusNet | none | 0.50354838 | 0.48970196 |
| ABL-02 | w/o DFF/GADFF prior | 34-37 | 0.56292027 | 0.57115078 |
| ABL-03 | w/o focal difference | 17-32 | 0.47684631 | 0.45476335 |
| ABL-04 | w/o glare cue | 33 | 0.40210730 | 0.34177950 |

## 2. 当前安全边界

本轮 preflight 没有生成以下产物：

```text
optimizer step
backward/update result
matched smoke checkpoint
test prediction
test metrics
manuscript-eligible ablation result
```

当前只允许支持以下判断：

1. ABL-00/02/03/04 可以共享同一 train/validation/test split；
2. ABL-02、ABL-03、ABL-04 的通道 mask 能在 upgraded 38-channel feature space 中运行；
3. 未来输出路径已经限定到 `tmp/ablation_results/<run_id>/`；
4. 当前仍不能写成正式训练结果、模块贡献结果或论文主表结果。

## 3. 恢复前先检查

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

若审计失败，先修复路径、日志、claim boundary 或缓存问题，再继续实验。

## 4. 下一步应先做的代码级小改动

下一步应在 `submission_planning/tools/run_ablation_variant_training.py` 或新 runner 中新增 matched smoke mode，建议接口如下：

```text
--run-kind matched_smoke
--tag 2026-06-19_matched_training_smoke
--run-id ABL-00 --run-id ABL-02 --run-id ABL-03 --run-id ABL-04
--max-epochs 1
--train-patches 8 或 16
--val-patches 4
--batch-size 1
```

建议输出：

```text
tmp/ablation_results/training_runner_matched_smoke/2026-06-19_matched_training_smoke_summary.md
tmp/ablation_results/training_runner_matched_smoke/2026-06-19_matched_training_smoke_summary.json
tmp/ablation_results/<run_id>/checkpoints/2026-06-19_matched_training_smoke.pt
tmp/ablation_results/<run_id>/metrics/2026-06-19_matched_training_history.csv
tmp/ablation_results/<run_id>/logs/2026-06-19_matched_training_smoke.md
```

`run_config.json` 建议保持：

```text
claim_eligible = false
main_table_eligible = false
```

并新增或更新：

```text
status = matched_training_smoke_run
matched_smoke_training.debug_only = true
matched_smoke_training.claim_eligible = false
matched_smoke_training.main_table_eligible = false
```

## 5. 下一条安全实验命令

runner 完成后，建议先运行轻量 smoke：

```text
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/run_ablation_variant_training.py --execute-training --run-kind matched_smoke --tag 2026-06-19_matched_training_smoke --run-id ABL-00 --run-id ABL-02 --run-id ABL-03 --run-id ABL-04 --max-epochs 1 --train-patches 8 --val-patches 4 --batch-size 1
```

若该命令耗时可接受，再提高到：

```text
--max-epochs 1 --train-patches 16 --val-patches 4 --batch-size 1
```

## 6. Matched smoke 完成后的审计项

matched smoke 完成后需要新增或更新：

```text
submission_planning/autonomous_research/ablation_matched_training_smoke.md
submission_planning/autonomous_research/research_log_2026-06-19_ablation_matched_training_smoke.md
submission_planning/tools/audit_ablation_matched_smoke_eligibility.py
tmp/ablation_results/eligibility_audits/ABL_matched_smoke_eligibility.md
tmp/ablation_results/eligibility_audits/ABL_matched_smoke_eligibility.json
```

还应同步更新：

```text
submission_planning/autonomous_research/research_index.md
submission_planning/autonomous_research/research_task_board.md
submission_planning/autonomous_research/experiment_roadmap.md
submission_planning/autonomous_research/submission_gap_closure_plan.md
submission_planning/autonomous_research/research_package_integrity_audit.md
submission_planning/tools/audit_research_package_integrity.py
```

audit 中至少检查：

1. ABL-00/02/03/04 均出现在 matched smoke summary；
2. 每个 run 的 checkpoint、history、log 均位于 `tmp/ablation_results/<run_id>/`；
3. 每个 run 使用同一 split、seed、patch plan 和 batch size；
4. `claim_eligible=false`；
5. `main_table_eligible=false`；
6. ABL-01 继续由 lower-prior focus-stack-only runner 单独处理。

## 7. 不能跨过的边界

1. 不把 matched preflight loss 写入论文主表；
2. 不把 matched smoke 结果写成模块贡献结论；
3. 不在 `src/`、`论文与PPT制作项目包/` 或原始结果目录写入新实验产物；
4. 不覆盖 `2026-06-19_controlled_pilot` 或 `2026-06-19_small_training_debug` 产物；
5. 不把 ABL-01 放入当前 38-channel masking runner；
6. 不把 DFV/DDFFNet 写成已完成外部 SOTA 对比。

## 8. 恢复时的一句话判断

如果从这里继续，第一步应新增 matched training smoke runner；第二步运行 ABL-00/02/03/04 smoke；第三步执行 matched smoke eligibility audit，并继续保持 claim gate 关闭。
