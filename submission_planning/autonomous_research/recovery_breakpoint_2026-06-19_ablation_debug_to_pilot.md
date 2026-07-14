# 恢复断点：ABL Small-Training Debug 到 Controlled Pilot

日期：2026-06-19  
断点名称：`ABL small-training debug -> controlled pilot`  
恢复目标：在不覆盖既有 debug 产物的前提下，继续推进 ABL-00/02/03/04 controlled pilot。

## 1. 当前最后稳定状态

当前最后稳定产物是 ABL-00/03 small-training debug：

```text
tmp/ablation_results/training_runner_small_train/ablation_training_runner_small_train_summary.md
tmp/ablation_results/training_runner_small_train/ablation_training_runner_small_train_summary.json
```

单 run 产物：

```text
tmp/ablation_results/ABL-00/checkpoints/2026-06-19_small_training_debug.pt
tmp/ablation_results/ABL-00/metrics/small_training_debug_metrics.csv
tmp/ablation_results/ABL-00/logs/2026-06-19_small_training_debug.md

tmp/ablation_results/ABL-03/checkpoints/2026-06-19_small_training_debug.pt
tmp/ablation_results/ABL-03/metrics/small_training_debug_metrics.csv
tmp/ablation_results/ABL-03/logs/2026-06-19_small_training_debug.md
```

当前 debug 指标：

| Run | Val MAE norm debug | 当前解释 |
|---|---:|---|
| ABL-00 | 0.18825971 | 小训练链路可运行 |
| ABL-03 | 0.27409419 | zero focal-difference 链路可运行 |

当前 `run_config.json` 安全边界：

```text
status = small_training_debug_run
claim_eligible = false
main_table_eligible = false
```

## 2. 恢复前先检查

恢复实验前先运行：

```text
python -X utf8 submission_planning/tools/audit_research_package_integrity.py
python -X utf8 submission_planning/tools/audit_manuscript_claim_safety.py
```

预期状态：

```text
Research package audit: pass
Manuscript claim safety audit: pass
```

若 audit 失败，先修复日志、claim boundary 或路径一致性，再继续实验。

## 3. 当前 runner 能力

当前训练入口：

```text
submission_planning/tools/run_ablation_variant_training.py
```

当前可用能力：

1. 默认 dry-run：检查数据、mask、模型、loss，不训练。
2. `--execute-training`：执行受保护小规模训练。
3. 目前训练产物名称固定为 `2026-06-19_small_training_debug` 和 `small_training_debug_metrics`。

当前可重跑命令如下，但会刷新同名 debug 产物，恢复时谨慎使用：

```text
python -X utf8 submission_planning/tools/run_ablation_variant_training.py --execute-training --run-id ABL-00 --run-id ABL-03 --max-epochs 1 --train-patches 8 --val-patches 4 --batch-size 1
```

## 4. 下一步应先做的代码级小改动

下一步先在 `run_ablation_variant_training.py` 中增加独立输出标签，建议接口如下：

```text
--run-kind small_debug|controlled_pilot
--tag 2026-06-19_controlled_pilot
```

建议保持默认行为兼容：

```text
run_kind = small_debug
tag = 2026-06-19_small_training_debug
```

controlled pilot 的输出应写入：

```text
tmp/ablation_results/training_runner_controlled_pilot/
tmp/ablation_results/<run_id>/checkpoints/2026-06-19_controlled_pilot.pt
tmp/ablation_results/<run_id>/metrics/2026-06-19_controlled_pilot_metrics.csv
tmp/ablation_results/<run_id>/logs/2026-06-19_controlled_pilot.md
```

`run_config.json` 建议新增：

```text
status = controlled_pilot_debug_run
claim_eligible = false
main_table_eligible = false
pilot_training = {
  "date": "2026-06-19",
  "debug_only": true,
  "claim_eligible": false,
  "main_table_eligible": false
}
```

## 5. 下一条安全实验命令

完成标签机制后，再运行 controlled pilot：

```text
python -X utf8 submission_planning/tools/run_ablation_variant_training.py --execute-training --run-kind controlled_pilot --tag 2026-06-19_controlled_pilot --run-id ABL-00 --run-id ABL-02 --run-id ABL-03 --run-id ABL-04 --max-epochs 2 --train-patches 16 --val-patches 8 --batch-size 1
```

初始 pilot 建议保持 CPU 友好设置。若运行时间可接受，再提高到：

```text
--max-epochs 3 --train-patches 64 --val-patches 16 --batch-size 1
```

## 6. Controlled pilot 完成后的审计项

pilot 完成后需要新增或更新：

```text
submission_planning/autonomous_research/ablation_controlled_pilot.md
submission_planning/autonomous_research/research_log_2026-06-19_ablation_controlled_pilot.md
tmp/ablation_results/training_runner_controlled_pilot/2026-06-19_controlled_pilot_summary.md
tmp/ablation_results/training_runner_controlled_pilot/2026-06-19_controlled_pilot_summary.json
```

还应更新：

```text
submission_planning/autonomous_research/research_index.md
submission_planning/autonomous_research/research_task_board.md
submission_planning/autonomous_research/experiment_roadmap.md
submission_planning/autonomous_research/submission_gap_closure_plan.md
submission_planning/autonomous_research/research_package_integrity_audit.md
submission_planning/tools/audit_research_package_integrity.py
```

audit 中至少检查：

1. ABL-00/02/03/04 均出现在 pilot summary。
2. 每个 run 的 checkpoint、metrics、log 均位于 `tmp/ablation_results/<run_id>/`。
3. `claim_eligible=false`。
4. `main_table_eligible=false`。
5. ABL-01 仍由 lower-prior 架构单独处理。

## 7. 不能跨过的边界

1. 不把 `small_training_debug` 或 `controlled_pilot` 数值写入论文主结果表。
2. 不把 ABL-00/03 debug 差异解释为模块贡献结论。
3. 不在 `src/`、`论文与PPT制作项目包/` 或原始结果目录写入新实验产物。
4. 不把 ABL-01 放入当前 38-channel masking runner。
5. 不把 DFV/DDFFNet 写成已完成外部 SOTA 对比。

## 8. 恢复时的一句话判断

如果从这里继续，第一步应修改 `run_ablation_variant_training.py` 的 tag/run-kind 输出机制；第二步运行 ABL-00/02/03/04 controlled pilot；第三步执行 pilot eligibility audit，并继续保持 claim gate 关闭。
