# 本轮自主研究日志：ABL Training Runner Dry Run

日期：2026-06-19  
范围：`submission_planning/` 与 `tmp/ablation_results/`。  
边界：本轮未修改 `src/`，未创建 optimizer，未执行 backward，未保存 checkpoint、预测数组、图像或真实 metrics。

## 1. 本轮研究目标

上一轮已经确认 future ABL training runner 不能直接调用原 final-method 实验脚本的输出型函数。本轮目标是把这个边界落实成一个可恢复的 runner 骨架，并先对 ABL-00 和 ABL-03 做默认 dry-run。

选择 ABL-00 和 ABL-03 的原因是：ABL-00 对应 full final method，ABL-03 对应 w/o focal-difference，是当前最直接影响方法贡献论证的一组核心对照。

## 2. 采用的方法

本轮新增 `submission_planning/tools/run_ablation_variant_training.py`。工具默认只运行 dry-run，支持 ABL-00 和 ABL-03 的 P10 64 x 64 center patch forward/loss interface check。

工具会读取 ABL workspace 的 `run_config.json`，确认 `claim_eligible=false` 和 `scaffold_only_no_training_run`，再生成 base 22-channel features 和 upgraded 38-channel features。ABL-00 保持完整 38 通道，ABL-03 zero channels 17-32，对应 focal-difference input signal。

工具当前显式拒绝 `--execute-training`。这保证它不会意外进入训练流程，也不会写出 checkpoint、prediction 或 figure。

## 3. 已完成的任务

已新增默认 dry-run runner：

```text
submission_planning/tools/run_ablation_variant_training.py
```

已生成 dry-run 总览：

```text
tmp/ablation_results/training_runner_dry_run/ablation_training_runner_dry_run_summary.md
tmp/ablation_results/training_runner_dry_run/ablation_training_runner_dry_run_summary.json
```

已生成 ABL-00 与 ABL-03 的 dry-run 日志：

```text
tmp/ablation_results/ABL-00/logs/2026-06-19_training_runner_dry_run.md
tmp/ablation_results/ABL-03/logs/2026-06-19_training_runner_dry_run.md
```

当前结果为 `pass`：共 20 项检查，0 个 error，0 个 warning。

## 4. 关键发现

ABL-00 dry-run 通过，输入输出形状为 `[1, 38, 64, 64] -> [1, 1, 64, 64]`，diagnostic loss 有限，且无梯度累积。

ABL-03 dry-run 通过，zero focal-difference channels 17-32 后仍能完成 `[1, 38, 64, 64] -> [1, 1, 64, 64]` 的 forward/loss interface check。

两个 run 均保持 `claim_eligible=false`，没有出现 `.pt`、`.pth`、`.ckpt`、`.npy`、`.png` 或 `.jpg` 等训练/预测/图像产物。

## 5. 对研究计划的更新

上一轮的下一步是实现 training runner。现在该 runner 的默认 dry-run 骨架已经完成。下一步应在同一个工具中增加受保护的小规模训练模式，要求必须显式传入 `--execute-training`，并允许限制 `--max-epochs`、`--train-patches` 和 `--val-patches`。

真正训练前，建议先跑：

```text
ABL-00: 1 epoch, 8 train patches, 4 validation patches
ABL-03: 1 epoch, 8 train patches, 4 validation patches
```

这些结果仍只能作为 runner debugging evidence，不能直接写入论文消融表。进入论文前需要完整 split、重复性记录和 claim eligibility audit。

## 6. 已同步更新的研究包内容

已新增 `submission_planning/autonomous_research/ablation_training_runner_dry_run.md`。

已更新 `audit_research_package_integrity.py` 与 `research_package_integrity_audit.md`，使完整性审计检查 dry-run summary、ABL-00/03 dry-run logs 和 runner 工具。

已更新 `research_task_board.md`、`research_index.md`、`experiment_roadmap.md`、`ablation_execution_protocol.md` 和 `submission_gap_closure_plan.md`，统一记录 dry-run 已完成，下一步是 explicit small-training mode。

## 7. 当前断点

当前断点是：

```text
R27: Ablation explicit small-training mode
```

推荐恢复点：

1. 打开 `submission_planning/tools/run_ablation_variant_training.py`；
2. 增加受保护的小规模训练分支；
3. 保持默认 dry-run 不变；
4. 将所有输出继续限制在 `tmp/ablation_results/<run_id>/`；
5. 先运行 ABL-00 与 ABL-03 的 1-epoch 小规模训练；
6. 训练后运行 research package audit 和 manuscript claim audit。

## 8. 当前证据边界

本轮可以支持的表述是：ABL training runner 的默认 dry-run 已能到达 ABL-00/03 的数据、mask、模型和 loss 接口。

本轮不能支持的表述包括：ABL-00 或 ABL-03 已经训练、focal-difference 已被消融证明有效、任何 diagnostic loss 可作为论文指标、任何 checkpoint 或预测结果已经生成。
