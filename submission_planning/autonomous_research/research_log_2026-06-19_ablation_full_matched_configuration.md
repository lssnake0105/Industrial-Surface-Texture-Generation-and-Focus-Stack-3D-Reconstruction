# 本轮自主研究日志：ABL Full Matched Configuration

日期：2026-06-19  
范围：`submission_planning/tools/`、`submission_planning/autonomous_research/`、`tmp/ablation_results/`。  
边界：本轮只生成 full matched training configuration preflight；未运行正式 full training，未生成新的 full candidate checkpoint，未改写 `src/` 或原项目交付包。

## 1. 本轮目标

上一轮 matched smoke 已证明 ABL-00/02/03/04 可以完成最小训练链路，但 smoke 预算过小，不能作为模块贡献证据。本轮目标是进入正式训练前的配置层，把训练预算、run kind、checkpoint tag、evaluator 需求和 eligibility gate 先固定下来。

## 2. 采用的方式

在 `submission_planning/tools/run_ablation_variant_training.py` 中新增 `matched_full_candidate` run kind。该模式沿用受保护 runner，只允许把产物写到 `tmp/ablation_results/<run_id>/`，并继续保持 `claim_eligible=false` 与 `main_table_eligible=false`。

新增 `submission_planning/tools/preflight_ablation_full_matched_configuration.py`。该工具只检查配置和前置条件，不训练模型。它读取 `build_dataset()` 的 split 计数，检查 ABL-00/02/03/04 的 run_config、matched smoke 证据、runner 支持状态，并生成 full candidate training/evaluator/eligibility 计划。

## 3. 已完成任务

已运行：

```text
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/preflight_ablation_full_matched_configuration.py
```

结果：

```text
Ablation full matched configuration preflight: pass
Checks: 31, errors: 0, warnings: 0
```

输出：

```text
tmp/ablation_results/matched_training_full_config/2026-06-19_matched_training_full_candidate_config_preflight.md
tmp/ablation_results/matched_training_full_config/2026-06-19_matched_training_full_candidate_config_preflight.json
```

## 4. 配置结论

当前 full matched candidate 配置为：

| Field | Value |
|---|---|
| epochs | 4 |
| train patches per epoch | 128 |
| validation patches per epoch | 32 |
| batch size | 1 |
| learning rate | 0.0006 |
| train split | all 27 samples |
| validation split | all 10 samples |
| test split | 7 samples |
| checkpoint tag | `2026-06-19_matched_training_full_candidate` |

该配置用于第一轮 full matched candidate。它仍是候选训练预算，后续需要通过 full-split evaluator 和 eligibility audit 才能决定是否进入论文表格。

## 5. 计划修正

本轮明确发现当前 `evaluate_ablation_full_split_metrics.py` 仍绑定 controlled-pilot checkpoint lookup 和 `controlled_pilot_p10_debug` training scope。因此下一步不应直接运行 full candidate training，否则训练完成后仍缺少匹配的 evaluator 和 claim gate。

新的断点为：

```text
R45: Matched full-split evaluator implementation
R46: Matched evaluator smoke on matched-smoke checkpoints
R47: ABL-00/02/03/04 matched full candidate training
R48: Full matched candidate eligibility audit
```

## 6. 当前结论

本轮把消融线从 matched smoke 推进到 full matched training configuration。当前仍没有正式训练结果，但已经固定了第一轮 full candidate 训练预算、输出标签、evaluator 改造需求和 eligibility audit 条件。
