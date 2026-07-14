# 本轮自主研究日志：ABL Small-Training Debug

日期：2026-06-19  
范围：`submission_planning/` 与 `tmp/ablation_results/`。  
边界：本轮未修改 `src/`，未写入原项目交付包。训练产物只保存在 `tmp/ablation_results/`，且仍标记为 debug-only。

## 1. 本轮研究目标

上一轮已经完成 ABL training runner dry-run，证明 ABL-00 和 ABL-03 能到达数据、mask、模型和 loss 接口。本轮目标是在同一个受保护 runner 中打开显式小规模训练模式，验证 optimizer、backward、weight update、checkpoint 写入和 debug metrics 写入是否能在 `tmp/ablation_results/` 内闭环。

该目标仍属于 runner debugging，不属于正式消融实验。它解决的是“训练链路是否可运行且不污染项目资源”的问题。

## 2. 采用的方法

本轮修改 `submission_planning/tools/run_ablation_variant_training.py`，保持默认 dry-run 行为不变，只有显式传入 `--execute-training` 才进入小规模训练。

执行命令为：

```text
python -X utf8 submission_planning/tools/run_ablation_variant_training.py --execute-training --run-id ABL-00 --run-id ABL-03 --max-epochs 1 --train-patches 8 --val-patches 4 --batch-size 1
```

训练输入仍使用 P10 synthetic scenario。ABL-00 使用完整 upgraded 38-channel features，ABL-03 在同一 feature space 中 zero channels 17-32，用于移除 focal-difference input signal。

## 3. 已完成的任务

已完成 ABL-00 与 ABL-03 的 1 epoch 小规模训练调试。

总览报告：

```text
tmp/ablation_results/training_runner_small_train/ablation_training_runner_small_train_summary.md
tmp/ablation_results/training_runner_small_train/ablation_training_runner_small_train_summary.json
```

ABL-00 产物：

```text
tmp/ablation_results/ABL-00/checkpoints/2026-06-19_small_training_debug.pt
tmp/ablation_results/ABL-00/metrics/small_training_debug_metrics.csv
tmp/ablation_results/ABL-00/logs/2026-06-19_small_training_debug.md
```

ABL-03 产物：

```text
tmp/ablation_results/ABL-03/checkpoints/2026-06-19_small_training_debug.pt
tmp/ablation_results/ABL-03/metrics/small_training_debug_metrics.csv
tmp/ablation_results/ABL-03/logs/2026-06-19_small_training_debug.md
```

## 4. 当前调试结果

| Run | Train loss debug | Val loss debug | Val MAE norm debug |
|---|---:|---:|---:|
| ABL-00 | 0.42624453 | 0.26109813 | 0.18825971 |
| ABL-03 | 0.36811332 | 0.39810403 | 0.27409419 |

这些数值只来自单个 P10 patch-sampling 调试设置。它们说明训练链路可运行，但不能说明模块贡献，也不能进入论文表格。

## 5. 关键判断

ABL-00 和 ABL-03 的最小训练链路均已打通。checkpoint 和 debug metrics 已写入 `tmp/ablation_results/<run_id>/`，没有写入 `src/` 或原项目交付包。

两个 run 的 `run_config.json` 均已更新为：

```text
status = small_training_debug_run
main_table_eligible = false
claim_eligible = false
```

这保证调试结果不会被误当成论文证据。

## 6. 对研究计划的更新

上一轮断点是 R27：Ablation explicit small-training mode。该断点已经推进完成。

新的断点是：

```text
R27: Controlled ABL-00/02/03/04 pilot training
```

下一步应扩展到 ABL-00、ABL-02、ABL-03、ABL-04，并使用 2-3 epoch、64-128 train patches、16-32 validation patches 的 controlled pilot。pilot 仍然应保持 `claim_eligible=false`，直到完整 split、per-sample metrics 和 eligibility audit 完成。

## 7. 当前证据边界

本轮可以支持的表述是：受保护 ABL runner 已能对 ABL-00 和 ABL-03 完成最小 optimizer/backward/update 链路，并将 debug checkpoint 和 metrics 写入 `tmp/ablation_results/`。

本轮不能支持的表述包括：focal-difference 已被证明有效、ABL-00 优于 ABL-03、debug val MAE 可作为论文指标、ABL 消融已完成。

## 8. 下一轮建议

如果继续消融线，优先运行 ABL-00/02/03/04 controlled pilot，并新增 pilot eligibility audit。若继续外部 SOTA 线，优先推进 DFV repository inventory 与 P10 prediction contract。
