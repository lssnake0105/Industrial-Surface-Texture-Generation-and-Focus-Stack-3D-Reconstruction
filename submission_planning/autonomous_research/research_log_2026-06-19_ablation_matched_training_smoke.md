# 本轮自主研究日志：ABL Matched Training Smoke

日期：2026-06-19  
范围：`submission_planning/tools/`、`submission_planning/autonomous_research/`、`tmp/ablation_results/`。  
边界：本轮只推进 matched split smoke training；未改写 `src/`，未写入原论文包、PPT 包或既有交付结果目录。

## 1. 本轮目标

上一轮 matched training preflight 只证明 forward/loss 条件成立，还没有执行 optimizer、backward、checkpoint 和 history 写入。本轮目标是把 ABL-00/02/03/04 从 preflight 推进到受保护的 matched smoke training，同时继续保持所有 claim gate 关闭。

## 2. 采用的方式

更新 `submission_planning/tools/run_ablation_variant_training.py`，新增 `matched_smoke` 模式。该模式复用 `build_dataset()` 的 train/validation/test split 计数，按 run-specific channel mask 准备 upgraded 38-channel features，并将 checkpoint、metrics、log 和 summary 全部写入 `tmp/ablation_results/`。

本轮还新增 `submission_planning/tools/audit_ablation_matched_smoke_eligibility.py`，用于检查四个核心变体是否都完成 smoke、是否保留统一 split 计数、是否存在 checkpoint/history/log，以及 `claim_eligible=false` 和 `main_table_eligible=false` 是否保持。

## 3. 已完成任务

已运行：

```text
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/run_ablation_variant_training.py --execute-training --run-kind matched_smoke --tag 2026-06-19_matched_training_smoke --run-id ABL-00 --run-id ABL-02 --run-id ABL-03 --run-id ABL-04 --max-epochs 1 --train-patches 2 --val-patches 1 --batch-size 1 --max-train-samples 2 --max-val-samples 1
```

结果：

```text
Ablation matched training smoke: pass
Checks: 52, errors: 0, warnings: 0
```

已运行：

```text
$env:PYTHONDONTWRITEBYTECODE='1'; python -B -X utf8 submission_planning/tools/audit_ablation_matched_smoke_eligibility.py
```

结果：

```text
Ablation matched smoke eligibility audit: pass
Eligibility: Matched smoke only; not manuscript ablation evidence
Checks: 78, errors: 0, warnings: 0
```

## 4. 关键观察

四个核心变体均完成 1 epoch、2 train patches、1 validation patch 的 smoke training。该 smoke 使用 fixed split boundary：train 27、validation 10、test 7，并为每个 run 生成独立 checkpoint、history CSV/JSON 和日志。

当前 debug validation MAE norm 分别为：

| Run | Val MAE norm debug |
|---|---:|
| ABL-00 | 0.10953103 |
| ABL-02 | 0.28376535 |
| ABL-03 | 0.15884387 |
| ABL-04 | 0.24383156 |

这些数值只说明 runner 可运行，不说明模块贡献强弱。

## 5. 计划修正

原先的下一步可以被理解为直接进入 full matched training。经过本轮 smoke 后，下一阶段应先补 full matched training configuration，而不是直接扩大 patch 数。原因是 full run 会产生可被误读的 metrics 和 checkpoints，必须先定义正式训练预算、test evaluator、claim eligibility gate 和结果入表条件。

新的断点为：

```text
R41: Full matched ablation training configuration
R42: Matched checkpoint full-split evaluator
R43: ABL-00/02/03/04 full matched runs
R44: Full matched ablation eligibility audit
```

## 6. 当前结论

本轮把消融线从“matched training 条件可预检”推进到“matched split smoke training 可运行”。当前仍未产生正式消融结果，但已经确认四个核心变体能在统一 split 边界和受保护输出目录中完成最小训练链路。
