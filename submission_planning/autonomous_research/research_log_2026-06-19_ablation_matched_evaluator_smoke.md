# 本轮自主研究日志：ABL Matched Evaluator Smoke

日期：2026-06-19  
范围：`submission_planning/tools/`、`submission_planning/autonomous_research/`、`tmp/ablation_results/`。  
边界：本轮只实现 matched checkpoint evaluator 并完成 1-sample smoke；未运行 full matched candidate training，未生成新的 full candidate checkpoint，未改写 `src/` 或原项目交付包。

## 1. 本轮目标

上一轮断点要求在 full matched candidate training 前先补齐 evaluator。原因是旧 `evaluate_ablation_full_split_metrics.py` 绑定 controlled-pilot checkpoint lookup 和 diagnostic scope，无法直接作为 matched full-candidate 的评估工具。

## 2. 采用的方式

新增 `submission_planning/tools/evaluate_ablation_matched_full_split_metrics.py`。该工具支持 `--checkpoint-tag`、`--training-scope`、`--evaluation-scope` 和 `--max-samples`，默认读取 `2026-06-19_matched_training_smoke` checkpoint 做 1-sample smoke。它复用现有模型、mask、tiled prediction 和 metrics 口径，但输出只写入 `tmp/ablation_results/matched_full_split_eval/<tag>/`。

## 3. 已完成任务

已运行 matched evaluator smoke：

```text
status = pass
runs = ABL-00, ABL-02, ABL-03, ABL-04
samples = 1
checks = 23
errors = 0
warnings = 0
```

输出记录：

```text
tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_evaluator_smoke/2026-06-19_matched_evaluator_smoke_summary.md
tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_evaluator_smoke/2026-06-19_matched_evaluator_smoke_summary.json
```

## 4. 计划修正

本轮将原计划中的 R45 与 R46 合并推进到完成：matched evaluator 已实现，并已经用 matched-smoke checkpoint 做最小 smoke。下一步不再需要先改 evaluator，重点转向 R47：运行 ABL-00/02/03/04 的 `matched_full_candidate` 训练。

新的任务顺序为：

```text
R47: ABL-00/02/03/04 matched full candidate training
R49: Matched full-candidate 7-sample evaluator run
R48: Full matched candidate eligibility audit
```

## 5. 当前结论

本轮把消融线从“full candidate 配置已固定”推进到“matched evaluator 已通过 smoke”。当前仍没有正式消融训练结果；已有 smoke metrics 只证明评估管线可运行。
