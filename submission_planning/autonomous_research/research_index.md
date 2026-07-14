# SRTP 自主研究索引

更新日期：2026-06-18  
保存位置：`submission_planning/autonomous_research/`  
边界：本目录只保存投稿研究笔记、文献判断、实验规划和风险清单，不修改现有源码、数据、图像、论文 PDF 或 README。

## 1. 当前研究命题

**建议论文主张：** 面向反光工业表面缺陷形貌，构建一个 simulation-to-real 的 focus-stack 3D reconstruction 框架，以合成数据提供可控 ground truth，以 DFF/GADFF、focal-difference volume 和 glare-aware cues 引导深度模型完成形貌校正。

**当前证据边界：**

| 证据类型 | 当前状态 | 可支持的论文表述 | 不能支持的表述 |
|---|---|---|---|
| 合成样本 | 有 height ground truth，可计算 MAE / edge MAE / high-risk MAE | controlled synthetic quantitative evaluation | 真实样本 absolute metrology accuracy |
| 真实样本 | 有 focus stack 与无参考形貌指标 | no-reference morphology stability, spike suppression, visual plausibility | calibrated real height error |
| 方法结果 | Focus-ResUNet 在现有 7 个合成测试样本 mean MAE 最低 | prior-guided learning can improve synthetic reconstruction accuracy | 泛化到全部工业材料或全部显微系统 |
| 对比方法 | 已有传统 DFF、post-processing、Lee2013、Li2019、GADFF、TinyDepthNet、Focus-ResUNet | 可形成内部 baseline 与传统 SFF/DFF baseline | 与最新 deep DFF SOTA 的完整公平比较 |

## 2. 当前项目结果摘要

### 2.1 Synthetic quantitative ranking

来源：`论文与PPT制作项目包/03_Data/algorithm_comparison/paper_algorithm_comparison_metrics.csv`

| Method | Mean MAE (um) | Mean Edge MAE (um) | Mean High-Risk MAE (um) | 样本数 |
|---|---:|---:|---:|---:|
| Focus-ResUNet | 53.22 | 86.68 | 40.14 | 7 |
| TinyDepthNet | 57.31 | 89.10 | 42.63 | 7 |
| Lee2013 adaptive window | 62.35 | 146.83 | 30.52 | 7 |
| Residual Focus-ResUNet | 62.52 | 126.91 | 30.08 | 7 |
| Li2019 adaptive iteration | 62.95 | 145.33 | 31.24 | 7 |
| Original DFF + post | 63.81 | 149.04 | 29.72 | 7 |
| Original DFF | 100.55 | 206.61 | 46.32 | 7 |
| GADFF | 105.83 | 210.60 | 46.38 | 7 |

**解读：** Focus-ResUNet 的整体 MAE 和 edge MAE 最强，但 high-risk MAE 仍由部分传统/残差方法更低。论文中应把主模型表述为整体形貌恢复与边界稳定性更好，同时承认 glare/high-risk 区域仍需要更细的先验设计或 loss weighting。

### 2.2 Real no-reference morphology metrics

来源：`论文与PPT制作项目包/03_Data/real_sample_comparison/real_midterm_method_summary.csv`

| Method | Roughness | Edge Retention to Frame | Relative Dynamic Range | Low-Conf Spike Count |
|---|---:|---:|---:|---:|
| Focus-ResUNet | 0.0078 | 0.0009 | 0.5317 | 2.0 |
| TinyDepthNet | 0.0067 | -0.0243 | 0.3623 | 49.0 |
| Original DFF + post | 0.0130 | -0.0296 | 0.3944 | 4091.3 |
| GADFF | 0.0316 | -0.0295 | 0.5406 | 3351.4 |
| Lee2013 adaptive window | 0.0295 | -0.0209 | 0.6325 | 5018.7 |
| Li2019 adaptive iteration | 0.0318 | -0.0233 | 0.6515 | 5433.6 |
| Original DFF | 0.0977 | -0.0394 | 0.6964 | 9179.7 |
| Residual Focus-ResUNet | 0.0594 | -0.0343 | 0.5203 | 9262.1 |

**解读：** 真实样本上，Focus-ResUNet 的 spike count 最低且 edge retention 接近正相关，说明其输出更稳定。该结论属于无参考形貌稳定性，不应写成真实高度误差更低。

## 3. SOTA 纳入判断

| 方法 | 当前层级 | 纳入理由 | 实验动作 |
|---|---|---|---|
| DDFFNet | P0 | 第一代端到端 learning-based DFF，可作为深度学习下限基线 | 优先复现或用本项目 synthetic split 重新训练 |
| DFV | P0 | differential focus volume 与本项目 focal-difference 思路最接近 | 必须优先尝试，作为最关键外部 SOTA |
| DfF in the Wild | P1 | 强化真实相机和 simulation-to-real 叙事 | 若适配成本可控，做补充对比 |
| DDFS | P1 | 相机参数和 defocus model 显式进入学习框架 | 相机参数可整理时复现，参数不足时写入 Related Work |
| HybridDepth | P1 | 2025 年 focal stack 与 single-image prior 融合路线 | 可作为较新的强外部对比候选 |
| Depth Anything V2 | P3 | arXiv:2406.09414v1 强调精确 synthetic labels、真实无标签 pseudo-label、teacher-student bridge、反光/透明场景鲁棒性 | 只做辅助可视化或方法论论据，不进入 focus-stack 主数值表 |
| DDL-Recurrent SFF | P2 | 2026 年 IJCV 学习型 SFF，和焦堆序列建模相关 | 加入 Related Work，等待代码/实现条件 |
| FAD | P2 | 频域增强 DFF，关联周期纹理和弱纹理 | 引用为最新趋势，代码可用时再复现 |
| DualFocus | P2 | spatio-focal constraints 关联边界和焦向歧义 | 引用为最新趋势 |
| Minimal Focal Stack | P2 | 减少焦堆帧数，关联采集成本 | 放入 Discussion / Future Work |

## 4. 研究假设

### H1: DFF prior 仍是显微信息中的强物理锚点

DFF 在强反光和弱纹理处容易失效，但其焦点响应仍保留了可解释的轴向线索。深度网络直接从图像到高度会承担过高学习负担，prior-guided correction 更适合当前数据规模。

### H2: Focal-difference volume 是连接传统 DFF 与 deep DFF 的关键表征

DFV 的 differential focus volume 和本项目的 focal-difference 设计都在强调焦平面维度的变化率。该方向应成为模型贡献中最核心的对外解释点。

### H3: High-risk 区域需要独立建模

现有结果显示 Focus-ResUNet 的整体 MAE 最优，但 high-risk MAE 并非最优。glare、边缘、低纹理和深度突变区域应有单独 mask、loss weight 或 confidence calibration。

### H4: Simulation-to-real 的核心难点是标签可信度与域差异

Depth Anything V2 对本项目的启发在于：高质量合成标签和真实无标签数据可以通过 teacher-student 方式连接。该文还提示真实采集深度标签可能存在传感器噪声、匹配误差或细节缺失，因此本项目使用可控仿真构造显微形貌 ground truth 具有合理性。下一阶段可以探索用当前模型、传统 DFF ensemble 或外部单帧 prior 为真实焦堆生成 pseudo depth / pseudo confidence，再进行自训练或一致性训练。

## 5. 可执行实验路线

### Route A: 最小投稿补强

时间：1-2 周  
目标：让稿件具备最低限度的外部 SOTA 对比可信度。

1. 复现或适配 DDFFNet。
2. 复现或适配 DFV。
3. 在同一 synthetic test set 上报告 MAE、edge MAE、high-risk MAE。
4. 在真实样本上只做 no-reference metrics 和可视化。
5. 更新 Related Work，覆盖 FAD、DDL-Recurrent SFF、DualFocus、HybridDepth、Depth Anything V2。

### Route B: Simulation-to-real 强化

时间：3-5 周  
目标：让论文故事从“一个模型更好”升级为“数据构造与训练策略可迁移”。

1. 增加 domain randomization ablation。
2. 加入真实无标签焦堆的 pseudo-label / consistency training 原型。
3. 把 DFF/GADFF confidence 作为 mask 或 loss weight。
4. 用 Depth Anything V2 作为单帧 qualitative auxiliary prior，展示其在显微反光表面上的优势和不足。

### Route C: 更高质量投稿

时间：6-10 周  
目标：补强真实验证和审稿抗风险能力。

1. 采集至少一个 step-height、标准粗糙度或可由 profilometer / confocal / white-light interferometry 标定的真实样本。
2. 在真实标定子集上报告 calibrated MAE 或 profile error。
3. 对 DFV / DDFS / HybridDepth 至少完成两个外部方法的公平适配。
4. 添加 failure analysis，明确哪些材料、纹理或反光条件仍不稳定。

## 5.1 外部基线工具链现状

当前已经完成的是外部 SOTA 复现前的临时数据、读取和评估基础设施；尚未完成 DFV/DDFFNet 的真实模型推理，因此不能把现有工具链写成外部 SOTA 数值结果。

| 模块 | 文件或位置 | 当前状态 | 可支持内容 | 不能支持内容 |
|---|---|---|---|---|
| 数据预检 | `external_baseline_data_preflight.md`, `../tools/preflight_external_baseline_data.py` | 已完成 | 明确 presentation package 中可直接定位的样本资产和缺口 | 完整外部 baseline 结果 |
| P10 临时导出 | `../tools/export_one_external_baseline_sample.py`, `tmp/external_baseline_data/` | 已完成单样本 | 生成 17 帧焦堆、GT、risk mask 和 priors，用于 adapter smoke test | 代表完整 7-sample synthetic test split |
| Dataloader smoke test | `../tools/smoke_test_external_baseline_package.py` | 已通过 P10 | 证明 P10 包可被整理为 `[1, 17, C, 540, 960]` 输入 | 证明 DFV/DDFFNet 已可运行 |
| 单预测评估器 | `external_prediction_evaluator.md`, `../tools/evaluate_external_prediction.py` | 已用 P10 DFF/GADFF priors 对齐 | 证明 MAE、P90、edge MAE、high-risk MAE 口径与项目原指标一致 | 产生外部模型预测 |
| 批量评估器 | `batch_external_evaluation.md`, `../tools/evaluate_external_prediction_batch.py` | 已用 P10 manifest smoke test | 后续可汇总 DFV/DDFFNet `.npy` 预测 | 替代外部模型复现 |
| 接入规程 | `dfv_ddffnet_integration_protocol.md` | 已完成 | 定义 DFV/DDFFNet 接入、导出、评估和入表门槛 | 保证外部仓库依赖一定可运行 |
| 入表审计 gate | `external_sota_eligibility_audit.md`, `../tools/audit_external_sota_eligibility.py` | 已完成并用 P10 prior smoke 负例验证 | 防止单样本 smoke 或缺日志结果误入主表 | 证明某外部方法优于当前方法 |
| 临时 workspace scaffold | `external_baseline_workspace_scaffold.md`, `../tools/scaffold_external_baseline_workspace.py` | 已为 DFV/DDFFNet 生成 `tmp/external_baseline_results/<method>/` | 固定 run_config、prediction manifest、logs、predictions、evaluation 路径 | 证明外部模型已经运行 |
| 研究包完整性审计 | `research_package_integrity_audit.md`, `../tools/audit_research_package_integrity.py` | 已通过，报告位于 `tmp/research_package_audits/` | 证明当前研究包文档、工具、临时目录和 claim safety 自洽 | 证明论文已经投稿就绪 |
| 稿件 claim 安全审计 | `manuscript_claim_safety_audit.md`, `../tools/audit_manuscript_claim_safety.py` | 已通过，报告位于 `tmp/manuscript_audits/` | 证明当前 LaTeX 草稿没有明显越过证据边界的主张 | 证明稿件不需要补实验 |
| 消融执行协议 | `ablation_execution_protocol.md`, `templates/ablation_run_matrix_template.csv` | 已完成 ABL-00 至 ABL-06 的 run matrix | 固定消融变体、证据要求、入稿规则和执行顺序 | 证明模块贡献已经验证 |
| 消融 workspace scaffold | `../tools/scaffold_ablation_workspace.py`, `tmp/ablation_results/` | 已生成 7 个消融运行目录和模板 | 为后续训练/评估保留日志、配置和指标路径 | 证明消融训练已经运行 |
| 消融特征 schema 审计 | `ablation_feature_schema_audit.md`, `../tools/audit_ablation_feature_schema.py` | 已通过，报告位于 `tmp/ablation_results/schema_audit/`，保留 ABL-05/06 实现 warning | 明确 22 通道 base input 和 ABL-01/02/04 的 mask 定义 | 证明消融训练已经运行 |
| ABL-03 焦向差分实现审计 | `abl03_focal_difference_implementation_audit.md`, `../tools/audit_abl03_focal_difference_implementation.py` | 已通过，确认 upgraded features 为 `[38, 540, 960]` 且 17-32 通道等于 `np.diff(stack)` | 明确 ABL-03 应 zero 17-32 通道来测试 focal-difference input signal | 证明 ABL-03 结果已经存在 |
| 消融 mask smoke test | `ablation_mask_smoke_test.md`, `../tools/smoke_test_ablation_masks.py` | 已通过，报告位于 `tmp/ablation_results/mask_smoke/` | 证明 ABL-01/02/03/04 输入 mask 可正确置零且非目标通道不变 | 证明消融训练结果 |
| 消融训练入口 preflight | `ablation_training_entry_preflight.md`, `../tools/preflight_ablation_training_entry.py` | 已通过，报告位于 `tmp/ablation_results/preflight/` | 修正 ABL-00/01/02/03/04 的训练入口、feature space 和 runner 设计边界 | 证明消融训练已经运行 |
| Minimal ablation runner smoke | `minimal_ablation_runner_smoke.md`, `../tools/run_ablation_variant_smoke.py` | 已通过，报告位于 `tmp/ablation_results/runner_smoke/` | 证明 ABL-00/02/03/04 可共享 upgraded 38-channel Focus-ResUNet 输入路径，ABL-01 需要单独 lower-prior runner | 证明任何消融性能结论 |
| Ablation training-runner preflight | `ablation_training_runner_preflight.md`, `../tools/preflight_ablation_training_runner.py` | 已通过，报告位于 `tmp/ablation_results/training_runner_preflight/` | 证明未来训练 runner 必须绕开原脚本输出路径，并把所有产物限制到 `tmp/ablation_results/<run_id>/` | 证明训练已经开始 |
| ABL-01 lower-prior decision | `tmp/ablation_results/ABL-01/logs/2026-06-19_lower_prior_architecture_decision.md` | 已完成 | 将 ABL-01 定义为 17-channel focus-stack-only lower-prior baseline，训练前还需选定架构 | 证明 ABL-01 性能 |
| Ablation training-runner dry run | `ablation_training_runner_dry_run.md`, `../tools/run_ablation_variant_training.py` | 已通过，报告位于 `tmp/ablation_results/training_runner_dry_run/`，并写入 ABL-00/03 logs | 证明 future runner 的默认 dry-run 能到达 ABL-00/03 的数据、mask、模型和 loss 接口 | 证明任何训练结果或性能 |
| Ablation small-training debug | `ablation_small_training_debug.md`, `../tools/run_ablation_variant_training.py` | 已通过，报告位于 `tmp/ablation_results/training_runner_small_train/`，ABL-00/03 均有 debug checkpoint 和 metrics | 证明受保护 runner 可完成最小 optimizer/backward/update 链路并保持 claim eligibility 关闭 | 证明可入稿消融结论 |
| 本轮开发日志与恢复断点 | `research_log_2026-06-19_development_resume_checkpoint.md`, `recovery_breakpoint_2026-06-19_ablation_debug_to_pilot.md` | 已完成 | 固化当前实验边界、可恢复命令、下一步 controlled pilot 改动和安全约束 | 替代训练结果或 eligibility audit |
| Ablation controlled pilot | `ablation_controlled_pilot.md`, `../tools/run_ablation_variant_training.py` | 已通过，报告位于 `tmp/ablation_results/training_runner_controlled_pilot/`，ABL-00/02/03/04 均有 pilot checkpoint 和 metrics | 证明受保护 runner 已覆盖四个核心消融路径，且 tag/run-kind 机制避免覆盖旧 debug 产物 | 证明模块贡献或可入稿消融结论 |
| Ablation pilot eligibility audit | `../tools/audit_ablation_pilot_eligibility.py`, `tmp/ablation_results/eligibility_audits/ABL_pilot_eligibility.md` | 已通过，eligibility 为 debug-only | 证明 pilot 结果仍保持 `claim_eligible=false` 和 `main_table_eligible=false` | 证明 pilot 数值可进入论文主表 |
| Controlled pilot 研究日志 | `research_log_2026-06-19_ablation_controlled_pilot.md` | 已完成 | 记录 runner 改动、pilot 结果、eligibility gate 和下一断点 | 替代 full-split 实验日志 |
| Ablation full-split debug evaluation | `ablation_full_split_debug_evaluation.md`, `../tools/evaluate_ablation_full_split_metrics.py` | 已通过，报告位于 `tmp/ablation_results/full_split_debug_eval/`，包含 7 个 test samples 的 per-sample 与 method summary metrics | 证明 full-split evaluator 可加载 pilot checkpoints 并计算 MAE、edge MAE、high-risk MAE | 证明正式消融结论 |
| Ablation full-split eligibility audit | `../tools/audit_ablation_full_split_eligibility.py`, `tmp/ablation_results/eligibility_audits/ABL_full_split_eligibility.md` | 已通过，eligibility 为 diagnostic-only | 证明 full-split metrics 仍不能入论文主表，因为训练来源只是 P10 tiny pilot checkpoints | 证明模块贡献可入稿 |
| Full-split debug evaluation 研究日志 | `research_log_2026-06-19_ablation_full_split_debug_eval.md` | 已完成 | 记录 full-split evaluator、结果摘要、eligibility gate 和 matched training 下一断点 | 替代正式训练日志 |
| Ablation matched training preflight | `ablation_matched_training_preflight.md`, `../tools/preflight_ablation_matched_training.py` | 已通过，报告位于 `tmp/ablation_results/matched_training_preflight/` | 证明 ABL-00/02/03/04 可在同一 train/validation/test split、统一 masking 和临时输出边界下完成 train/validation forward loss 检查 | 证明 matched training 已完成或可入稿 |
| Matched training preflight 研究日志 | `research_log_2026-06-19_ablation_matched_training_preflight.md` | 已完成 | 记录本轮 preflight、训练计划修正和下一步 smoke runner 断点 | 替代训练结果或 eligibility audit |
| Ablation matched training smoke | `ablation_matched_training_smoke.md`, `../tools/run_ablation_variant_training.py` | 已通过，报告位于 `tmp/ablation_results/training_runner_matched_smoke/` | 证明 ABL-00/02/03/04 可在统一 split 边界下完成受保护 smoke training，并生成 tmp 内 checkpoint、history 和 run log | 证明正式 matched training 或模块贡献 |
| Ablation matched smoke eligibility audit | `../tools/audit_ablation_matched_smoke_eligibility.py`, `tmp/ablation_results/eligibility_audits/ABL_matched_smoke_eligibility.md` | 已通过，eligibility 为 matched-smoke-only | 证明 matched smoke 结果仍保持 `claim_eligible=false` 和 `main_table_eligible=false` | 证明 smoke 数值可进入论文主表 |
| Matched smoke 研究日志与恢复断点 | `research_log_2026-06-19_ablation_matched_training_smoke.md`, `recovery_breakpoint_2026-06-19_matched_smoke_to_full_config.md` | 已完成 | 记录 runner 改动、smoke 结果、eligibility gate 和 full matched configuration 下一断点 | 替代正式训练配置 |
| Ablation full matched training configuration | `ablation_full_matched_training_configuration.md`, `../tools/preflight_ablation_full_matched_configuration.py` | 已通过，报告位于 `tmp/ablation_results/matched_training_full_config/` | 证明正式候选训练预算、checkpoint tag、split 覆盖、evaluator 缺口和 eligibility gate 已被固定 | 证明 full training、test metrics 或模块贡献 |
| Full matched configuration 研究日志与恢复断点 | `research_log_2026-06-19_ablation_full_matched_configuration.md`, `recovery_breakpoint_2026-06-19_full_config_to_matched_evaluator.md` | 已完成 | 记录本轮从 full configuration 转入 matched evaluator 的计划修正和恢复入口 | 替代 evaluator、训练或 eligibility audit |
| Ablation matched full-split evaluator smoke | `ablation_matched_full_split_evaluator_smoke.md`, `../tools/evaluate_ablation_matched_full_split_metrics.py` | 已通过，报告位于 `tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_evaluator_smoke/` | 证明 matched evaluator 可按 checkpoint tag 读取模型并输出 metrics | 证明正式 full-candidate evaluation |
| Ablation matched full candidate results | `ablation_matched_full_candidate_results.md`, `tmp/ablation_results/training_runner_matched_full_candidate/2026-06-19_matched_training_full_candidate_summary.md`, `tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_full_candidate_eval/2026-06-19_matched_full_candidate_eval_summary.md` | 已通过，eligibility audit 为 current-stage evidence | 证明当前 full-candidate ablation 已覆盖 27/10/7 split、4 个核心变体和 7 个 test samples | 证明 full model 已最终定型或优于所有消融变体 |
| Supervisor update | `supervisor_update_2026-06-19.md`, `research_log_2026-06-19_ablation_full_candidate_results.md`, `recovery_breakpoint_2026-06-19_after_full_candidate_eval.md` | 已完成 | 汇总原型到投稿研究的改进点、当前实验验证和 prospect | 替代论文最终结果或外部 SOTA |
| Ablation matched longer-budget repeat | `ablation_matched_longer_repeat_results.md`, `tmp/ablation_results/training_runner_matched_longer_repeat/2026-06-19_matched_training_longer_repeat_summary.md`, `tmp/ablation_results/matched_full_split_eval/2026-06-19_matched_longer_repeat_eval/2026-06-19_matched_longer_repeat_eval_summary.md` | 已通过 | 证明增加训练预算后所有变体误差下降，但 full model 仍未占优 | 证明辅助信号无价值或最终模型已确定 |
| Supervisor experiment report | `supervisor_experiment_report_2026-06-19.md`, `report_figures_2026-06-19/` | 已完成 | 以论文式中文结构汇总实验流程、样品条件、结果图、研究判断和 prospect | 替代正式论文正文或外部 SOTA |
| Depth Anything V2 auxiliary lane | `depth_anything_v2_auxiliary_protocol.md`, `../tools/scaffold_depth_anything_v2_auxiliary_workspace.py` | 已生成 scaffold，位于 `tmp/foundation_depth_auxiliary/DepthAnythingV2/` | 固定单目基础深度模型的 auxiliary-only 用法、输入清单和 claim 边界 | 证明 Depth Anything V2 已运行或可进入主表 |
| Manuscript Depth Anything V2 同步 | `../manuscript_draft/s2r_focus_stack_manuscript.tex`, `../manuscript_draft/references.bib` | 已同步 auxiliary protocol 边界并清理重复 BibTeX 条目 | 让正文把 Depth Anything V2 写成训练策略和单帧辅助参考 | 证明 auxiliary 结果已经存在 |
| 投稿差距闭环计划 | `submission_gap_closure_plan.md` | 已完成 | 将外部 baseline、消融、真实 GT 边界和入稿门槛绑定成可执行任务 | 证明投稿缺口已完成 |
| DFV environment preflight | `dfv_environment_preflight.md`, `../tools/preflight_dfv_environment.py` | 已通过，报告位于 `tmp/external_baseline_results/DFV/preflight/` | 证明本地 P10 数据包、DFV 输出目录、Python/PyTorch/CUDA 环境已准备好，下一步可下载仓库到 `tmp/external_repos/DFV/` | 证明 DFV 已运行或有预测结果 |
| 本轮自主研究日志 | `research_log_2026-06-18_ablation_training_entry_preflight.md` | 已完成 | 记录本轮完成内容、采用方式和对原始研究计划的修正 | 替代实验指标或模型运行日志 |

下一步的关键分界线有两条：外部 SOTA 方面，只有当 DFV 或 DDFFNet 在固定 synthetic split 上输出可复现 `.npy` 预测，并通过 batch evaluator 生成 method summary 后，才允许进入论文主结果表；消融方面，只有当 matched full-candidate checkpoints 经过 7-sample full-split evaluator 和 eligibility audit 后，才允许写入 ablation table。

## 6. 下一步工作包

| 工作包 | 产出 | 优先级 |
|---|---|---|
| WP1 外部基线适配调查 | `external_baseline_feasibility.md` | 已完成 |
| WP2 数据接口规范 | `dataset_interface_notes.md`, `baseline_adapter_spec.md`, `dfv_ddffnet_integration_protocol.md` | 已完成 |
| WP3 Ablation 表设计 | full / w-o prior / w-o focal difference / w-o glare prior / w-o domain randomization | 高 |
| WP4 Related Work 草稿 | 按 Classical SFF, Deep DFF, Sim-to-real, Foundation depth, Industrial morphology 组织 | 中 |
| WP5 真实样本评估补强 | profile curve、edge continuity、spike map、confidence map | 中 |
| WP6 DFV 外部仓库 smoke test | `tmp/external_repos/` 与 `tmp/external_baseline_results/DFV/` 下的运行日志 | 高 |
| WP7 DDFFNet 数据格式适配 | HDF5 或官方 loader 兼容记录 | 中 |
| WP8 投稿缺口闭环 | 按 `submission_gap_closure_plan.md` 推进 DFV 和核心消融证据门槛 | 高 |
| WP9 DFV repository inventory | 下载/放置 DFV 到 `tmp/external_repos/DFV/` 后做代码入口、依赖和 loader 调查 | 高 |
| WP10 Minimal ablation runner | 基于 `ablation_training_entry_preflight.md` 设计 ABL-00/01/02/03/04 的最小可审计训练入口 | 高 |
| WP11 Ablation training runner | 基于 `minimal_ablation_runner_smoke.md` 写入真实训练日志、metrics 和 claim eligibility gate | 高 |
| WP12 ABL-00/03 small-scale dry run | 在 future training runner 可用后，先验证 full 与 w/o focal-difference 两个关键变体 | 高 |
| WP13 Explicit small-training mode | 在 `run_ablation_variant_training.py` 中增加受保护的小规模训练模式，先跑 ABL-00/03 1 epoch | 高 |
| WP14 Controlled ablation pilot | 扩展到 ABL-00/02/03/04 的 2-3 epoch pilot，并准备后续 claim eligibility audit | 已完成 |
| WP15 Full-split diagnostic ablation metrics | 为 ABL-00/02/03/04 pilot checkpoints 生成 per-sample MAE、edge MAE、high-risk MAE 和 full-split eligibility audit | 已完成 |
| WP16 Matched full-split ablation training | 用同一 train/validation split、epoch、patch、seed 规则训练 ABL-00/02/03/04，并生成 claim-eligible audit | 高 |
| WP17 Matched training smoke mode | 在正式 matched runs 前完成 ABL-00/02/03/04 的 1-epoch smoke runner、临时 checkpoint、history 和 eligibility audit | 已完成 |
| WP18 Full matched ablation configuration | 定义正式 matched training 预算、checkpoint tag、full-split evaluator 和 claim eligibility gate | 已完成 |
| WP19 Matched full-split evaluator | 实现可选择 checkpoint tag 与 training/evaluation scope 的 7-sample evaluator，并先在 matched-smoke checkpoints 上 smoke test | 已完成 |
| WP20 Matched full-candidate ablation | 运行 ABL-00/02/03/04 full candidate training、7-sample evaluator 和 eligibility audit | 已完成 |
| WP21 Longer-budget / seed repeat | 复核 full model 当前未占优是否来自训练预算、随机性或辅助信号融合问题 | 部分完成：longer-budget repeat 已完成，seed repeat 未完成 |
| WP22 Supervisor-facing project update | 整理原型改进点、实验验证成果、风险解释和下一步 prospect | 已完成 |
| WP23 Gated auxiliary fusion / cue audit | 针对 focal-difference 和 glare cue 设计 gating、confidence weighting 或 cue quality audit | 高 |

## 7. 不污染项目资源的规则

1. 新研究笔记只放入 `submission_planning/autonomous_research/`。
2. 外部仓库若后续需要下载，应放入未跟踪的 `tmp/` 或单独 `external_repos/`，并在复现前确认是否加入 `.gitignore`。
3. 不覆盖 `src/`、`data/`、`results/`、`assets/`、`论文与PPT制作项目包/` 中的已有文件。
4. 只在明确需要更新投稿材料时修改 `submission_planning/` 下的 Markdown 文件。
5. 每次修改后用 `git status --short` 确认只出现预期新增或更新文件。

## 8. 最新研究状态更新

已新增消融训练入口 preflight。当前结果为 pass：ABL-00/01/02/03/04 的脚手架、既有 schema audit、ABL-03 implementation audit 和 mask smoke test 均可追溯；同时确认 final-method 消融应以 `src/train_focus_resunet_loss_experiment.py` 的 upgraded Focus-ResUNet 路径为主，早期 `src/final_dataset_training.py` 更适合作为基础数据/训练参考入口。下一步应设计 minimal ablation runner，再进入训练和 metrics 写入。

已新增本轮自主研究日志。日志记录了本轮从 DFV 环境可用性检查转入消融入口核验的研究决策，并把原计划中的“直接训练消融”更新为“先完成 minimal ablation runner 与形状/配置验证”。

2026-06-19 已新增 minimal ablation runner smoke。当前结果为 pass：ABL-00/02/03/04 在 64 x 64 P10 patch 上均可通过 `[1, 38, 64, 64] -> [1, 1, 64, 64]` 的 final Focus-ResUNet 前向形状检查，ABL-01 被明确记录为需要单独 lower-prior runner。该结果只证明 runner 接口和 mask 设计可进入下一阶段，不构成训练或消融性能证据。

2026-06-19 已新增 ablation training-runner preflight。当前结果为 pass：未来训练 runner 不应调用原 `main()`、`evaluate_split()`、`write_metric_plots()` 或 `write_report()`，因为这些函数会写入原项目交付包；训练 runner 应只复用模型、损失、特征构造和指标部件，并将所有输出写入 `tmp/ablation_results/<run_id>/`。ABL-01 已单独记录为 lower-prior focus-stack-only 架构决策项。

2026-06-19 已新增 ablation training-runner dry run。当前结果为 pass：默认 dry-run 支持 ABL-00 和 ABL-03，均完成 `[1, 38, 64, 64] -> [1, 1, 64, 64]` 的 forward/loss interface check，并确认没有 optimizer、backward、checkpoint、prediction、figure、metric result 或 claim update。下一步才是显式受保护的小规模训练模式。

2026-06-19 已新增 ABL-00/03 small-training debug。当前结果为 pass：两条路径均完成 1 epoch、8 train patches、4 validation patches 的最小训练调试，并在 `tmp/ablation_results/<run_id>/` 写入 checkpoint 与 debug metrics。`claim_eligible=false` 与 `main_table_eligible=false` 保持不变，因此这些数值只证明 runner 链路可运行，不能作为论文消融结论。

2026-06-19 已新增本轮开发日志与恢复断点。当前断点明确为：先给 `run_ablation_variant_training.py` 增加 tag/run-kind 输出机制，再运行 ABL-00/02/03/04 controlled pilot，并新增 pilot eligibility audit。已有 debug checkpoint 与 metrics 继续保留在 `tmp/ablation_results/`，不得进入论文主表。

2026-06-19 已新增 ABL-00/02/03/04 controlled pilot。当前结果为 pass：四个核心变体均完成 2 epoch、16 train patches、8 validation patches 的受保护 pilot，并写入独立 `2026-06-19_controlled_pilot` 标签产物。`audit_ablation_pilot_eligibility.py` 已确认该结果为 debug-only evidence，不能作为论文主表或模块贡献结论。下一步断点为 full-split ablation metrics runner、full-split eligibility audit 和 DFV repository inventory。

2026-06-19 已新增 ABL-00/02/03/04 full-split debug evaluation。当前结果为 pass：controlled-pilot checkpoints 已在 7 个 synthetic test samples 上完成 tiled inference 和 MAE、edge MAE、high-risk MAE 计算。`audit_ablation_full_split_eligibility.py` 已确认该结果为 diagnostic-only evidence，因为训练来源仍是 P10 tiny pilot checkpoints。下一步断点为 matched full-split ablation training runner、claim-eligible audit 和 DFV repository inventory。

2026-06-19 已新增 ABL-00/02/03/04 matched training preflight。当前结果为 pass：固定 split 计数为 train 27、validation 10、test 7；四个核心变体均能完成 64 x 64 train/validation patch 的 forward/loss 检查，并保持 planned outputs 位于 `tmp/ablation_results/<run_id>/`。该结果只证明训练前条件成立，不生成 optimizer step、checkpoint、test prediction 或入稿消融证据。下一步断点为 matched training smoke runner、matched smoke eligibility audit 和 DFV repository inventory。

2026-06-19 已新增 ABL-00/02/03/04 matched training smoke。当前结果为 pass：四个核心变体均在 fixed split boundary 27/10/7 下完成 1 epoch、2 train patches、1 validation patch 的受保护 smoke training，并生成 `2026-06-19_matched_training_smoke` 标签的 checkpoint、history 和 log。`audit_ablation_matched_smoke_eligibility.py` 已确认 eligibility 为 matched-smoke-only，因此这些数值只能证明 runner continuity，不能进入论文主表或支持模块贡献。下一步断点为 full matched ablation training configuration、matched checkpoint full-split evaluator 和 DFV repository inventory。

2026-06-19 已新增 full matched ablation training configuration。当前结果为 pass：正式候选训练配置固定为 tag `2026-06-19_matched_training_full_candidate`、4 epochs、128 train patches、32 validation patches、batch size 1、learning rate 0.0006、train/validation/test split 27/10/7，并继续保持 `claim_eligible=false` 与 `main_table_eligible=false`。下一步断点为 matched full-split evaluator implementation，并先用 matched-smoke checkpoints 做 evaluator smoke。

2026-06-19 已新增 matched evaluator smoke。当前结果为 pass：`evaluate_ablation_matched_full_split_metrics.py` 可读取 `2026-06-19_matched_training_smoke` checkpoint，并在 1 个 test sample 上生成 per-sample metrics、method summary 和 run_config 记录。该结果只证明 evaluator path 可运行，不构成消融性能证据。

2026-06-19 已完成 ABL-00/02/03/04 matched full candidate training、7-sample evaluator 和 eligibility audit。当前结果显示 DFF/GADFF prior 是最稳定的贡献：移除 prior 后 Mean MAE 从 130.9028 um 上升到 245.3440 um，high-risk MAE 从 117.9743 um 上升到 261.3550 um。同时，w/o focal difference 与 w/o glare cue 在当前 4-epoch candidate budget 下优于 full model，提示辅助信号融合和训练策略需要复核。当前 supervisor update 已整理为 `supervisor_update_2026-06-19.md`。

2026-06-19 已完成 matched longer-budget repeat。8-epoch repeat 将 full model Mean MAE 从 130.9028 um 降到 109.2209 um，但 w/o focal difference 和 w/o glare cue 仍分别达到 90.4542 um 与 75.4572 um。该结果说明 full model 未占优并非单纯训练预算不足，下一步应聚焦 gated auxiliary fusion、glare cue quality audit 或 seed repeat，同时继续推进 DFV/DDFFNet 外部 SOTA。

2026-06-19 已新增 supervisor-facing 实验汇报 `supervisor_experiment_report_2026-06-19.md`。该文档用中文论文式结构整理研究背景、实验流程、样品条件、candidate vs longer repeat、per-sample heatmap、主要研究判断和 prospect；其中纯流程内容采用文字说明，图片只保留实验预算、样品条件和结果对比。
