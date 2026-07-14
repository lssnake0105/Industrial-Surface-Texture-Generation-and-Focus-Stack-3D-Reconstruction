# 本轮自主研究日志：ABL Matched Training Preflight

日期：2026-06-19  
范围：`submission_planning/tools/`、`submission_planning/autonomous_research/`、`tmp/ablation_results/`。  
边界：本轮只做 matched training 前置检查；未改写 `src/`，未写入原论文包、PPT 包或既有结果目录。

## 1. 本轮目标

上一轮完成了 full-split diagnostic evaluation，但训练来源仍是 P10 tiny controlled pilot checkpoints。本轮目标是继续向正式消融训练推进，验证 ABL-00/02/03/04 能否共享同一 train/validation/test split、同一超参数计划和同一临时输出边界。

## 2. 采用的方式

新增 `submission_planning/tools/preflight_ablation_matched_training.py`。脚本只复用模型、loss、特征增强和数据生成函数，执行 train/validation forward 与 loss 检查，不创建 optimizer，不执行 backward，不保存 checkpoint。

脚本验证的核心条件包括：

1. split 计数为 train 27、validation 10、test 7；
2. ABL-00/02/03/04 均在 upgraded 38-channel feature space 中可运行；
3. ABL-02、ABL-03、ABL-04 的目标通道可正确置零；
4. train/validation 64 x 64 patch forward 输出为 `[1, 1, 64, 64]`；
5. 所有 loss 为有限值；
6. 所有 planned outputs 均位于 `tmp/ablation_results/<run_id>/`。

## 3. 已完成任务

已运行：

```text
python -X utf8 submission_planning/tools/preflight_ablation_matched_training.py
```

结果：

```text
Ablation matched training preflight: pass
Checks: 50, errors: 0, warnings: 0
```

输出：

```text
tmp/ablation_results/matched_training_preflight/2026-06-19_matched_training_preflight.md
tmp/ablation_results/matched_training_preflight/2026-06-19_matched_training_preflight.json
```

## 4. 计划修正

原来的下一步可以被理解为直接运行 matched training。本轮检查后，计划应再细分一层：先做 matched training smoke mode，再做更完整的 matched runs。原因是正式训练会产生多个 checkpoint、history 和 test metrics，必须先确认 runner 不会调用原脚本输出函数，也不会写入交付包。

新的断点为：

```text
R37: Matched training smoke runner
R38: ABL-00/02/03/04 matched smoke runs
R39: Matched smoke eligibility audit
R40: Full matched ablation training configuration
```

## 5. 当前结论

本轮把消融线从“完整 test split 可评价”推进到“matched training 条件可预检”。当前仍未产生正式训练结果，但已经确认四个核心消融变体可以在统一 split 和统一临时输出边界下进入下一阶段 smoke training。
