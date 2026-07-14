# 本轮自主研究日志：ABL 消融训练入口预检与计划修正

日期：2026-06-18  
范围：`submission_planning/` 与 `tmp/` 下的研究规划、审计工具和临时 preflight 报告。  
边界：本轮未修改 `src/`，未启动训练，未下载外部仓库，未生成 checkpoint、预测图或模型输出。

## 1. 本轮研究目标

本轮工作承接前一阶段的 DFV environment preflight。DFV 侧已经确认 P10 临时数据包、PyTorch/CUDA 环境和输出目录可用，但外部仓库尚未下载，仍不能进入预测或主表结果。因此，本轮将研究重点转向另一个投稿关键缺口：核心消融实验能否从已有脚手架推进到可执行训练。

核心问题是：ABL-00/01/02/03/04 的 run matrix 虽然已经存在，但早期脚手架将 `src/final_dataset_training.py` 作为统一训练入口，可能无法准确代表最终论文方法。若直接基于旧假设训练，消融结果很容易与最终模型结构不一致，进而削弱方法贡献的可信度。

## 2. 采用的方法

本轮采用静态代码审计、脚手架一致性检查和既有 preflight 证据交叉验证三类方式完成判断。

首先，对核心训练相关文件进行 AST 级语法检查和符号检查。重点核对 `src/final_dataset_training.py`、`src/train_focus_resunet_loss_experiment.py` 和 `src/simulate_antiglare_highres_samples.py` 中的训练函数、数据构造函数、特征增强函数、模型类和指标函数是否存在。

其次，对 `tmp/ablation_results/ABL-00` 至 `ABL-04` 的 run config、synthetic metrics 模板、real metrics 模板和日志目录进行检查，确认这些目录仍处于 `scaffold_only_no_training_run` 状态，且 `claim_eligible=false`。

最后，交叉读取此前生成的三项证据：feature schema audit、ABL-03 focal-difference implementation audit 和 mask smoke test。这样可以判断通道 mask 是否正确，也可以判断这些 mask 是否真正对应最终 Focus-ResUNet 的输入空间。

## 3. 已完成的任务

已新增 `submission_planning/autonomous_research/ablation_training_entry_preflight.md`，作为 ABL-00/01/02/03/04 训练前入口核验说明。该文档明确记录当前可执行边界、检查对象、风险点和下一步 runner 设计要求。

已新增 `submission_planning/tools/preflight_ablation_training_entry.py`。该工具只做只读检查，不导入重型训练模块，不训练模型，不写 checkpoint。运行后生成 `tmp/ablation_results/preflight/ablation_training_entry_preflight.json` 和 `.md` 报告。

本轮 preflight 结果为 `pass`，共 55 项检查，0 个 error，0 个 warning。该结果说明消融脚手架、核心源文件符号、既有 schema audit、ABL-03 implementation audit 和 mask smoke test 在当前状态下是自洽的。

## 4. 对原始研究计划的修正

### 修正一：最终方法消融入口需要锚定 upgraded Focus-ResUNet

原计划默认 `src/final_dataset_training.py` 可以作为 ABL-00/01/02/03/04 的统一训练入口。preflight 后判断应修正为：最终方法相关消融应优先锚定 `src/train_focus_resunet_loss_experiment.py`，因为该脚本包含 `FocusResUNet`、`HybridDFFLoss`、`augment_features()`、`predict_tiled_upgraded()` 等最终方法路径所需部件。

`src/final_dataset_training.py` 仍然有价值，适合作为基础数据读取、base training workflow 和指标写入参考。它不再作为 final-method ablation 的唯一入口假设。

### 修正二：ABL-03 的消融空间应从 base 22-channel 更新为 upgraded 38-channel

此前的 feature schema audit 证明 base feature 是 `[22, 540, 960]`。后续 ABL-03 implementation audit 进一步确认最终 Focus-ResUNet 会通过 `augment_features()` 构造 `[38, 540, 960]` upgraded feature，其中 17-32 通道是相邻焦平面差分。

因此，ABL-03 应定义为在 upgraded 38-channel feature 上 zero channels 17-32，用来检验 focal-difference input signal 的贡献。若只在 base 22-channel 上讨论 ABL-03，会漏掉最终模型真实使用的焦向差分表示。

### 修正三：ABL-01 不能简单等同于通道置零

ABL-01 的目标是 direct image-to-depth 或 lower-prior baseline。若只是把 final method 的 prior 通道置零，它仍可能保留最终模型结构、损失设计或残差边界带来的能力。preflight 后建议将 ABL-01 作为单独 runner 设计问题处理，先明确它使用哪些 raw stack 输入、是否保留 residual bound、是否保留 HybridDFFLoss。

### 修正四：ABL-02 和 ABL-04 需要在 upgraded 路径上重新定义

ABL-02 的 “w/o DFF/GADFF prior” 在 base 22-channel 上可以对应 18-21 通道，但在 upgraded 38-channel 路径中，需要进一步确认 `augment_features()` 后 prior channel 的对应位置与是否存在 derived prior signal。

ABL-04 的 “w/o glare cue” 在 base feature 中可以先 zero risk channel 17，但如果 GADFF 或 confidence 分支也隐含 glare-aware 信息，后续 runner 需要决定只移除 explicit risk cue，还是扩展为更严格的 glare-related cue removal。

### 修正五：下一步优先级从直接训练调整为 minimal runner 设计

本轮之后，消融任务不应直接进入训练。更稳妥的下一步是新增 minimal ablation runner，先完成 shape check、channel mask check、config logging、metrics path check 和 dry-run 级别验证。只有 runner 自身可审计后，ABL-00/01/02/03/04 的训练结果才适合进入论文证据链。

## 5. 已同步更新的研究包内容

已将 ABL training-entry preflight 接入 `research_task_board.md`，新增 D42，并将下一步 Ready 任务更新为 minimal ablation runner design。

已将该 preflight 接入 `research_index.md`，新增外部 baseline 工具链表中的“消融训练入口 preflight”记录，并新增 WP10 minimal ablation runner 工作包。

已更新 `experiment_roadmap.md`，将消融状态改为“入口 preflight 已完成，训练不足”，并把 Week 2 的消融任务从表格设计推进为 runner 设计。

已更新 `submission_gap_closure_plan.md`，将核心消融闭环路径调整为先完成 minimal ablation runner，再进入 ABL-00/01/02/03/04 训练和 claim eligibility 审计。

已更新 `ablation_execution_protocol.md`，新增 training-entry preflight result 小节，给出 ABL-00 至 ABL-04 的修正入口、feature space 和下一步实现动作。

已更新 `research_package_integrity_audit.md` 和 `audit_research_package_integrity.py`，使研究包完整性审计能够检查新 preflight 文档、工具和报告。

## 6. 当前证据边界

本轮完成的是训练前的研究核验，不是消融实验本身。当前可支持的表述是：ABL-00/01/02/03/04 的训练入口、feature space 和脚手架状态已经被检查，且已修正 final-method 消融路径。

当前不能支持的表述包括：某个模块已经被消融证明有效、ABL-00 优于 ABL-01/02/03/04、glare cue 已提升 high-risk MAE、focal-difference 已在实测中提升边缘误差。这些结论仍需要训练日志、metrics CSV、预测文件和 claim eligibility audit。

## 7. 下一轮建议

下一轮若继续消融线，建议先新增 `run_ablation_variant_smoke.py` 或等效工具。该 runner 应只在 `tmp/ablation_results/` 写入日志和临时结果，先支持 ABL-00 与 ABL-03 的 shape/config smoke，再逐步扩展到 ABL-01/02/04。

下一轮若继续 SOTA 线，建议在网络或外部代码条件允许时把 DFV 仓库放入 `tmp/external_repos/DFV/`，先做 code inventory 和 dataloader contract check，再生成 P10 prediction contract。

综合投稿风险来看，DFV baseline 和 minimal ablation runner 是当前两个最高优先级。Depth Anything V2 已完成 auxiliary 定位和稿件边界同步，短期内不应占用主实验优先级。
