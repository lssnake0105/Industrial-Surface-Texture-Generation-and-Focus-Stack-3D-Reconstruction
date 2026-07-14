# Ablation Controlled Pilot

日期：2026-06-19  
范围：ABL-00、ABL-02、ABL-03、ABL-04。  
标签：`2026-06-19_controlled_pilot`。  
边界：本轮是 controlled pilot/debug training，只验证受保护 runner 可扩展到四个核心变体；不构成论文消融结论。

## 1. 实验目标

上一阶段 ABL-00/03 small-training debug 只证明 full model 和 w/o focal-difference 两条路径可以完成最小 optimizer/backward/update 链路。本轮目标是扩大到 ABL-00/02/03/04，确认同一个 upgraded 38-channel Focus-ResUNet runner 可以在不同 channel masking 设置下完成训练、checkpoint、metrics 和日志写入。

## 2. 执行命令

```text
python -X utf8 submission_planning/tools/run_ablation_variant_training.py --execute-training --run-kind controlled_pilot --tag 2026-06-19_controlled_pilot --run-id ABL-00 --run-id ABL-02 --run-id ABL-03 --run-id ABL-04 --max-epochs 2 --train-patches 16 --val-patches 8 --batch-size 1
```

## 3. 输出位置

总览报告：

```text
tmp/ablation_results/training_runner_controlled_pilot/2026-06-19_controlled_pilot_summary.md
tmp/ablation_results/training_runner_controlled_pilot/2026-06-19_controlled_pilot_summary.json
```

单 run 输出：

```text
tmp/ablation_results/<run_id>/checkpoints/2026-06-19_controlled_pilot.pt
tmp/ablation_results/<run_id>/metrics/2026-06-19_controlled_pilot_metrics.csv
tmp/ablation_results/<run_id>/logs/2026-06-19_controlled_pilot.md
```

eligibility audit：

```text
tmp/ablation_results/eligibility_audits/ABL_pilot_eligibility.md
tmp/ablation_results/eligibility_audits/ABL_pilot_eligibility.json
```

## 4. Pilot 结果

| Run | Variant | Zeroed channels | Last val MAE norm debug | Status |
|---|---|---|---:|---|
| ABL-00 | Full S2R-FocusNet | none | 0.32677382 | controlled_pilot_debug_completed |
| ABL-02 | w/o DFF/GADFF prior | 34-37 | 0.26719564 | controlled_pilot_debug_completed |
| ABL-03 | w/o focal difference | 17-32 | 0.22327405 | controlled_pilot_debug_completed |
| ABL-04 | w/o glare cue | 33 | 0.28974518 | controlled_pilot_debug_completed |

这些数值来自单个 P10 synthetic sample 的极小 patch-sampling pilot。它们只说明四个变体都能在受保护 runner 中完成训练链路，不说明模块贡献方向。

## 5. Eligibility audit 结果

`audit_ablation_pilot_eligibility.py` 已生成独立审计，结果为：

```text
Ablation pilot eligibility audit: pass
Eligibility: Debug-only; not manuscript main-table evidence
Checks: 53, errors: 0, warnings: 0
```

四个 run 的 `run_config.json` 均保持：

```text
claim_eligible = false
main_table_eligible = false
```

## 6. 当前可支持的表述

当前可以安全表述：

1. 受保护 ABL runner 已支持 ABL-00/02/03/04 四个核心变体。
2. ABL-02、ABL-03、ABL-04 的 channel masking 路径可以完成训练、checkpoint 和 metrics 写入。
3. 所有 pilot 产物都位于 `tmp/ablation_results/` 下，没有写入原项目交付包。
4. pilot eligibility audit 已确认这些结果仍为 debug-only evidence。

当前不能安全表述：

1. full model 优于任一 ablation variant。
2. DFF/GADFF prior、focal difference 或 glare cue 的贡献已被证明。
3. pilot val MAE 可以进入论文消融表。
4. ABL-01 lower-prior baseline 已完成。

## 7. 下一步断点

下一步应从 pilot 进入 claim-eligible 消融设计，最小要求包括：

1. 固定完整 synthetic split；
2. 为 ABL-00/02/03/04 生成 per-sample MAE、edge MAE 和 high-risk MAE；
3. 至少记录 seed、split、训练配置和代码状态；
4. 完成 full-split ablation eligibility audit；
5. 仍将 ABL-01 作为单独 lower-prior 架构任务处理。
