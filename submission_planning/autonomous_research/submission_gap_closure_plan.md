# 投稿差距闭环计划

更新日期：2026-06-18

用途：把当前研究包中的投稿缺口转化为可执行、可审计、可入稿的闭环任务。本文档只记录计划、证据门槛和写作边界，不代表实验已经完成。

## 1. 当前投稿状态判断

当前稿件已经具备一个可投稿雏形：任务定义、仿真数据、内部 baseline、真实无参考验证、SOTA 文献定位和 LaTeX 草稿均已形成。主要短板集中在三类证据：外部 deep DFF 对比、核心模块消融、真实域验证边界。投稿策略应先让主张与证据对齐，再决定是否追求更强的期刊版本。

| 维度 | 当前状态 | 投稿风险 | 闭环目标 |
|---|---|---|---|
| 论文故事 | 基本成型 | simulation-to-real 叙事仍需用实验表支撑 | 把仿真 GT、真实无 GT、prior-guided correction 串成一条主线 |
| 外部 SOTA | 文献和接口规划已完成，实测未完成 | 审稿人可能质疑对比不充分 | 至少完成 DFV 或 DDFFNet 的可复现实测，理想状态加入二者 |
| 消融实验 | matched full-candidate training、7-sample evaluator、eligibility audit 和 longer-budget repeat 已完成 | full model 在 longer repeat 后仍未占优，模块贡献需要更谨慎解释 | 将 DFF/GADFF prior 写成当前稳定贡献；focal-difference/glare cue 进入 gated fusion、cue audit 或 exploratory component |
| 真实验证 | no-reference metrics 已有，calibrated GT 缺失 | 不能宣称真实绝对精度 | 明确 no-reference 边界，若条件允许补一个标定子集 |
| Foundation depth | Depth Anything V2 已定位为辅助参考 | 容易被误写成不公平主基线 | 只作为训练策略和单帧 qualitative auxiliary |

## 2. 必须闭环的投稿缺口

| ID | 缺口 | 最小闭环证据 | 入稿位置 | 未闭环时写法 |
|---|---|---|---|---|
| G1 | 外部 deep DFF 对比 | DFV 或 DDFFNet 在固定 synthetic split 上输出 `.npy` 预测，并通过 batch evaluator | Experiments main table | 写成 planned external baseline，不写 superiority |
| G2 | 核心消融 | ABL-00/02/03/04 已生成 matched full-candidate training log、7-sample synthetic metrics 和 eligibility audit | Ablation / Discussion | 当前可写 prior contribution；focal-difference/glare cue 需要复核后再写强贡献 |
| G3 | 真实 GT 边界 | Abstract、Experiments、Discussion 均明确 calibrated real height GT unavailable | 全文 claim safety | 只报告 no-reference morphology stability |
| G4 | High-risk 区域解释 | high-risk MAE、risk mask 可视化、失败样本描述 | Results / Discussion | 写成 limitation 和 future modeling target |
| G5 | 公平比较说明 | frame count、training setting、scale alignment、metric function 均有记录 | Experiments setup | 不比较不同输入假设下的绝对优劣 |
| G6 | 单一最终方法 | 全文只把 `S2R-FocusNet` 作为 final method | Title / Abstract / Method | TinyDepthNet、Residual variants 只作为内部 baseline |

## 3. 优先级路线

### Route 1: 最小投稿闭环

目标：让稿件具备基本审稿可信度。

| 顺序 | 动作 | 成功标准 | 失败时降级 |
|---|---|---|---|
| 1 | 跑通 DFV P10 single-sample smoke test | 生成 DFV `.npy` 预测和单样本 evaluator 报告 | 记录依赖阻碍，DFV 保持 Related Work |
| 2 | 扩展到 7-sample synthetic test split | batch evaluator 输出 `method_summary_metrics.csv` | 只保留 P10 qualitative / protocol，不入主表 |
| 3 | 基于 longer repeat 做 gated fusion / glare cue quality audit | full model 的辅助信号贡献更稳定 | 当前先作为 supervisor discussion 和 model-design feedback |
| 4 | 更新 LaTeX 主结果表和 ablation 表 | 表格数值来自 evaluator / run logs | 只写当前 internal baseline |
| 5 | rerun claim safety 和 research package audit | 两项均 pass | 修正文稿边界 |

### Route 2: 稳健期刊版本

目标：降低真实域和 SOTA 对比相关审稿风险。

| 顺序 | 动作 | 成功标准 |
|---|---|---|
| 1 | 同时完成 DFV 与 DDFFNet | 两个外部方法进入主结果表 |
| 2 | 尝试 HybridDepth 或 DfF in the Wild | 至少一个 2022 以后强相关方法进入补充表 |
| 3 | 采集一个 calibrated real subset | step-height / profilometer / confocal / interferometry 任一可解释参考 |
| 4 | 增加 synthetic-real gap figure | 展示真实样本和合成样本在 glare、texture、focus response 上的差异 |
| 5 | 增加 failure analysis panel | P10、periodic stripe、real glare 样本均有 profile 或 mask 可视化 |

## 4. 入稿门槛

| 内容 | 允许进入 Abstract | 允许进入 Results | 允许进入 Discussion | 当前状态 |
|---|---|---|---|---|
| synthetic GT 数据构造 | 是 | 是 | 是 | 已满足 |
| Focus-ResUNet internal mean MAE 最低 | 是，限定 internal baselines | 是 | 是 | 已满足 |
| real spike suppression | 是，限定 no-reference | 是 | 是 | 已满足 |
| DFV / DDFFNet 数值对比 | 否 | 需要实测通过 | 可讨论为 planned | 未满足 |
| module contribution | 否 | 需要消融通过 | 可写为设计动机 | 未满足 |
| Depth Anything V2 | 否，除非作为背景一句 | 不进主表 | 是，training strategy / auxiliary prior | 已满足边界 |
| real absolute height accuracy | 否 | 需要 calibrated real GT | 可作为 future work | 未满足 |

## 5. 实验证据门槛

### 5.1 外部 baseline 入表门槛

外部方法进入主表前必须同时满足：

1. 预测文件存在于 `tmp/external_baseline_results/<method>/predictions/`；
2. `prediction_manifest.csv` 覆盖目标 synthetic split；
3. batch evaluator 生成 per-sample 和 method summary；
4. run log 记录代码来源、训练设置、输入帧数、输出尺度；
5. eligibility audit 通过；
6. `main_table_eligible=true` 只在审计通过后更新。

### 5.2 消融入稿门槛

消融结果进入论文前必须同时满足：

1. ABL-00 full model 与每个 ablation variant 使用同一 split；
2. 每个 variant 有 run config、run log、metrics CSV；
3. 输入 mask 或配置变更经过 smoke test；
4. synthetic metrics 至少包含 MAE、edge MAE、high-risk MAE；
5. run log 明确 seed、训练入口和代码状态；
6. `claim_eligible=true` 只在 metrics 和日志齐全后更新。

### 5.3 真实样本入稿门槛

真实样本只允许支持以下主张：

| 可支持 | 不可支持 |
|---|---|
| no-reference morphology stability | real absolute height accuracy |
| spike suppression | calibrated metrology |
| relative dynamic range | universal industrial generalization |
| edge retention trend | superiority over external SOTA on real GT |

## 6. 下一步任务映射

| 下一步 | 对应任务板 ID | 产出 | 影响 |
|---|---|---|---|
| DFV repo / environment inventory | D41 / R26 | `tmp/external_baseline_results/DFV/logs/<date>_inventory.md` | 环境 preflight 已通过，下一步决定外部仓库是否可执行 |
| DFV P10 prediction contract | R18 | P10 `.npy` + evaluator report | 验证外部模型输出接口 |
| DFV eligibility audit | R20 | `DFV_eligibility_audit.md` | 决定是否进入主表 |
| Matched training smoke runner | D61/D62 | ABL-00/02/03/04 matched smoke histories、checkpoints、logs | 已完成，eligibility 为 matched-smoke-only |
| Matched smoke eligibility audit | D63 | `tmp/ablation_results/eligibility_audits/ABL_matched_smoke_eligibility.md` | 已完成 |
| Full matched ablation training configuration | D66 | 正式 matched training config、run log 和 claim-eligible metrics 计划 | 已完成，仍未生成 training checkpoint |
| Matched checkpoint full-split evaluator | D68/D69 | matched checkpoint 的 7-sample test split evaluator 和 matched-smoke smoke report | 已完成 |
| Matched full-candidate training/evaluation | D71/D72/D73 | full-candidate histories、checkpoints、7-sample metrics 和 eligibility audit | 已完成，结果需谨慎解释 |
| Supervisor update | D74/D75 | `supervisor_update_2026-06-19.md` 和 full-candidate 结果记录 | 已完成 |
| Longer-budget / seed repeat | R50/R51 | repeat histories、metrics、stability comparison | 当前 full model 未占优，需要复核 |
| Matched longer-budget repeat | D76/D77/D78/D79 | 8-epoch repeat histories、7-sample metrics、更新后的 supervisor update | 已完成，说明 full 未占优并非单纯训练不足 |
| Gated fusion / cue audit | R52/R54 | fusion design note、glare cue quality audit | 下一步模型改进重点 |
| ABL-00 config verification | R23 | config verification log | 确认 full model 复现实验入口 |
| ABL-03 zero-diff smoke runner | R24 | zero-difference smoke log | 确认 focal-difference 消融路径 |
| Depth Anything V2 preprocessing decision | R25 | input selection note | 准备 auxiliary qualitative figure |

## 7. 写作更新顺序

| 阶段 | 更新内容 | 触发条件 |
|---|---|---|
| Draft-safe | 保持当前 LaTeX 草稿，只写 internal baselines 和 planned external work | 当前状态 |
| Baseline-added | 增加 DFV/DDFFNet 数值行和公平比较说明 | 外部 baseline 审计通过 |
| Ablation-added | 增加 ablation 表和模块贡献段落 | 消融 metrics 与 logs 齐全 |
| Journal-strengthened | 增加 calibrated real subset 或更完整 failure analysis | 真实标定 / 扩展实验完成 |

## 8. 当前结论

当前最短路径是把已完成的消融结果转化为更稳健的投稿证据，同时推进 DFV 外部 baseline。Depth Anything V2 已完成文献定位、辅助协议和稿件边界同步；DFV environment preflight 已确认本地数据包、PyTorch 和 CUDA 可用；ABL 消融链路已完成 matched full-candidate training、7-sample evaluator、eligibility audit 和 longer-budget repeat。当前最清楚的实验结论是 DFF/GADFF prior 有显著贡献；longer repeat 后 full model 仍未优于 w/o focal difference 和 w/o glare cue，因此下一轮优先级应是 gated auxiliary fusion、glare cue quality audit、seed repeat 或 DFV repository download / code inventory。
