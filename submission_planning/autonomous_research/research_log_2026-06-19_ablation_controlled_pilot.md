# 本轮自主研究日志：ABL Controlled Pilot

日期：2026-06-19  
范围：`submission_planning/tools/`、`submission_planning/autonomous_research/`、`tmp/ablation_results/`。  
边界：本轮新增受保护 runner 的 pilot 输出标签机制，并运行 ABL-00/02/03/04 的极小 controlled pilot；未改写 `src/`，未写入论文包、PPT 包或原结果目录。

## 1. 本轮目标

上一轮断点要求先避免覆盖 ABL-00/03 small-training debug 产物，再扩展到 ABL-00/02/03/04 controlled pilot。本轮目标是完成这个断点，并为 pilot 结果增加独立 eligibility audit，防止 debug 数值被误写成论文消融结论。

## 2. 采用的方式

本轮修改 `submission_planning/tools/run_ablation_variant_training.py`，新增：

```text
--run-kind small_debug|controlled_pilot
--tag <safe_tag>
```

默认 `small_debug` 保留旧路径和旧 summary 名称。`controlled_pilot` 写入独立路径：

```text
tmp/ablation_results/training_runner_controlled_pilot/
tmp/ablation_results/<run_id>/checkpoints/2026-06-19_controlled_pilot.pt
tmp/ablation_results/<run_id>/metrics/2026-06-19_controlled_pilot_metrics.csv
tmp/ablation_results/<run_id>/logs/2026-06-19_controlled_pilot.md
```

随后运行四个核心变体：

```text
ABL-00: full model
ABL-02: w/o DFF/GADFF prior
ABL-03: w/o focal difference
ABL-04: w/o glare cue
```

## 3. 已完成任务

已完成 controlled pilot：

```text
tmp/ablation_results/training_runner_controlled_pilot/2026-06-19_controlled_pilot_summary.md
tmp/ablation_results/training_runner_controlled_pilot/2026-06-19_controlled_pilot_summary.json
```

已新增并运行 pilot eligibility audit：

```text
submission_planning/tools/audit_ablation_pilot_eligibility.py
tmp/ablation_results/eligibility_audits/ABL_pilot_eligibility.md
tmp/ablation_results/eligibility_audits/ABL_pilot_eligibility.json
```

## 4. Pilot 结果摘要

| Run | Variant | Last val MAE norm debug | 当前解释 |
|---|---|---:|---|
| ABL-00 | Full S2R-FocusNet | 0.32677382 | full runner 可训练 |
| ABL-02 | w/o DFF/GADFF prior | 0.26719564 | prior mask 路径可训练 |
| ABL-03 | w/o focal difference | 0.22327405 | focal-difference mask 路径可训练 |
| ABL-04 | w/o glare cue | 0.28974518 | glare cue mask 路径可训练 |

结果中的相对大小不能解释为模块贡献，因为本轮只使用单个 P10 样本、2 epoch、16 train patches、8 validation patches。

## 5. 安全边界

pilot eligibility audit 已通过：

```text
Status: pass
Eligibility: Debug-only; not manuscript main-table evidence
Checks: 53
Errors: 0
Warnings: 0
```

四个 run 的 `run_config.json` 均保持：

```text
claim_eligible = false
main_table_eligible = false
```

## 6. 对研究计划的更新

上一轮计划中的 R27、R28、R29 已完成：runner 具备 tag/run-kind 输出机制，controlled pilot 已跑通，pilot eligibility audit 已生成。研究计划的下一阶段应从“能不能训练”转向“能不能形成可入稿证据”。

新的断点为：

```text
R30: Full-split ablation metrics runner
R31: Full-split ablation eligibility audit
R32: DFV repository inventory and P10 prediction contract
```

## 7. 当前结论

本轮把消融线从 ABL-00/03 小训练 debug 推进到 ABL-00/02/03/04 controlled pilot。当前成果证明受保护训练链路已经覆盖三个核心模块消融路径，但论文主张仍需完整 split、per-sample metrics、重复 seed 或至少更充分的配置审计。
