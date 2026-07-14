# 恢复断点：Full Matched Configuration 到 Matched Evaluator

日期：2026-06-19  
断点名称：`ABL full matched configuration -> matched full-split evaluator`  
恢复目标：在运行 full matched candidate training 前，先实现能读取指定 checkpoint tag 的 matched full-split evaluator。

## 1. 当前最后稳定状态

当前最后稳定产物是 full matched training configuration preflight：

```text
submission_planning/autonomous_research/ablation_full_matched_training_configuration.md
submission_planning/autonomous_research/research_log_2026-06-19_ablation_full_matched_configuration.md
tmp/ablation_results/matched_training_full_config/2026-06-19_matched_training_full_candidate_config_preflight.md
tmp/ablation_results/matched_training_full_config/2026-06-19_matched_training_full_candidate_config_preflight.json
```

当前 preflight 结果：

```text
status = pass
checks = 31
errors = 0
warnings = 0
claim_eligible = false
main_table_eligible = false
```

## 2. 当前 full candidate 配置

| Field | Value |
|---|---|
| tag | `2026-06-19_matched_training_full_candidate` |
| seed | 20260619 |
| patch size | 64 |
| epochs | 4 |
| train patches per epoch | 128 |
| validation patches per epoch | 32 |
| batch size | 1 |
| learning rate | 0.0006 |
| train split | all 27 samples |
| validation split | all 10 samples |
| test split | 7 samples |
| run kind | `matched_full_candidate` |
| output scope | `tmp/ablation_results/<run_id>/` |

## 3. 当前不能跨过的边界

1. full matched training 尚未运行；
2. full candidate checkpoint 尚不存在；
3. full matched test metrics 尚不存在；
4. 当前 full-split evaluator 仍绑定 controlled-pilot checkpoint lookup；
5. 任何 ablation result 仍不能进入论文主表。

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

## 5. 下一步应做的代码级任务

下一步先新增 evaluator：

```text
submission_planning/tools/evaluate_ablation_matched_full_split_metrics.py
```

最低要求：

1. 支持 `--checkpoint-tag`；
2. 支持 `--training-scope` 和 `--evaluation-scope`；
3. 默认可对 `2026-06-19_matched_training_smoke` checkpoint 做 `--max-samples 1` smoke；
4. full run 时输出 28 行 per-sample metrics 和 4 行 method summary；
5. 所有输出写入 `tmp/ablation_results/matched_full_split_eval/` 或等效 `tmp/` 子目录；
6. 写入 run_config 时继续保持 `claim_eligible=false` 和 `main_table_eligible=false`。

## 6. 下一组建议任务

```text
R45: Matched full-split evaluator implementation
R46: Matched evaluator smoke on matched-smoke checkpoints
R47: ABL-00/02/03/04 matched full candidate training
R48: Full matched candidate eligibility audit
```

## 7. 恢复时的一句话判断

如果从这里继续，第一步应实现 matched checkpoint evaluator；第二步用 matched smoke checkpoint 做 1-sample evaluator smoke；第三步再运行 full matched candidate training。
