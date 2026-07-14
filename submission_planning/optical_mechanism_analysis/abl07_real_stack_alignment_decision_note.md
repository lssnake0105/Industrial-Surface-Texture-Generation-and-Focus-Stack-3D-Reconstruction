# ABL-07 Real-Stack Alignment Decision Note

- Date: 2026-06-22
- Scope: no-reference diagnostics on real focus stacks
- Status: audit-passed internal evidence
- Claim boundary: no calibrated real height ground truth is used here; this note supports real-stack diagnostic alignment, not absolute real-height accuracy.

## Decision

ABL-07 can be kept as the main internal candidate for the current Simulation-to-Real manuscript story. The strongest real-stack evidence is a diagnostic-consistency result: on seven real focus stacks, both ABL-07 checkpoints substantially suppress DFF local-deviation spikes in regions flagged as unreliable by focus-margin, spike, saturation, and focus-curve morphology diagnostics.

中英文对照：

ABL-07 可以继续作为当前 Simulation-to-Real 论文故事线的主候选模型。当前真实样本证据的核心是诊断一致性：在 7 组真实焦栈上，两组 ABL-07 checkpoint 都能在 focus-margin、spike、saturation 与 focus-curve morphology 指标判定为不可靠的区域显著抑制 DFF 的局部波动。

## Evidence Summary

| Checkpoint | Real stacks | Low-confidence dev reduction | Spike-top10 dev reduction | Saturated dev reduction | Confident-region corr. | Confident std ratio |
|---|---:|---:|---:|---:|---:|---:|
| 2026-06-22_confidence_gated_prior_full_candidate | 7 | 94.85% | 96.25% | 83.16% | 0.5073 | 0.6299 |
| 2026-06-22_confidence_gated_prior_seed_repeat | 7 | 95.32% | 96.93% | 81.40% | 0.4213 | 0.4727 |

Across all per-stack rows, the minimum low-confidence reduction is 87.56%, the minimum spike-top10 reduction is 90.93%, and the minimum finite saturated-region reduction is 77.71%. These values show that the smoothing behavior is not concentrated in only one sample.

中英文对照：

逐栈结果中，低置信区域局部波动下降的最小值为 87.56%，spike-top10 区域下降的最小值为 90.93%，存在饱和掩码的样本中饱和区域下降的最小值为 77.71%。这说明该现象具有跨真实焦栈的一致性，并非由单一样本驱动。

## Interpretation for the Paper Story

The real-stack result strengthens the Simulation-to-Real story in a narrow but useful way. It supports the claim that the confidence-gated DFF/GADFF prior behaves coherently on real focus stacks: it trusts DFF-like structure more in confident regions and suppresses unstable DFF behavior in diagnostically risky regions. This matches the synthetic finding that low-confidence focus regions are the clearest source of ABL-07's gain.

中英文对照：

真实焦栈结果以较窄但有价值的方式加强了 Simulation-to-Real 叙事。它支持这样的表述：confidence-gated DFF/GADFF prior 在真实焦栈上具有一致行为，即在较可信区域保留 DFF 结构，在诊断风险较高的区域抑制 DFF 的不稳定波动。这与合成实验中“低置信焦点区域是 ABL-07 最清晰收益来源”的结论相互对应。

## What This Does Not Prove

This evidence does not prove real-height MAE, calibrated micron-level accuracy, external SOTA superiority, or final paper-table readiness. The real images do not include calibrated height ground truth, so all real-stack claims must be written as no-reference diagnostic alignment or qualitative/diagnostic validation.

中英文对照：

这组证据不能证明真实高度 MAE、经过标定的微米级精度、相对于外部 SOTA 的优越性，也不能直接作为最终论文主表结论。由于真实图像缺少标定高度真值，所有真实样本相关表述都应限制在无参考诊断一致性或定性/诊断验证范围内。

## Recommended Manuscript Wording

Safe English wording:

> On real focus stacks without calibrated height ground truth, ABL-07 consistently reduced local DFF fluctuations in diagnostically unreliable regions, including low-confidence focus regions, spike-prone regions, and saturated regions. This result supports the proposed confidence-gated prior as a simulation-to-real alignment mechanism, while calibrated real-height accuracy remains unverified.

Safe Chinese wording:

> 在缺少标定高度真值的真实焦栈上，ABL-07 在低置信焦点区域、spike-prone 区域和饱和区域中一致降低了 DFF 的局部波动。该结果支持 confidence-gated prior 作为 Simulation-to-Real 对齐机制的合理性，但真实高度精度仍需后续标定实验验证。

## Artifact Pointers

- Main diagnostic report: `submission_planning/optical_mechanism_analysis/abl07_real_stack_alignment/abl07_real_stack_alignment_report.md`
- Aggregate metrics: `submission_planning/optical_mechanism_analysis/abl07_real_stack_alignment/abl07_real_stack_alignment_aggregate.csv`
- Per-stack metrics: `submission_planning/optical_mechanism_analysis/abl07_real_stack_alignment/abl07_real_stack_alignment_stack_metrics.csv`
- Class metrics: `submission_planning/optical_mechanism_analysis/abl07_real_stack_alignment/abl07_real_stack_alignment_class_metrics.csv`
- Summary JSON: `submission_planning/optical_mechanism_analysis/abl07_real_stack_alignment/abl07_real_stack_alignment_summary.json`
- Visual panels: `submission_planning/optical_mechanism_analysis/abl07_real_stack_alignment/*_alignment.png`
- Morphology maps: `submission_planning/optical_mechanism_analysis/abl07_real_stack_alignment/*_alignment_class_map.png`
- Audit report: `tmp/ablation_results/eligibility_audits/ABL07_real_stack_alignment_audit.md`
- Audit JSON: `tmp/ablation_results/eligibility_audits/ABL07_real_stack_alignment_audit.json`

## Next Research Step

The next useful step is manuscript-level consolidation: compress the model outputs into one method name and one evidence chain, then run a final claim-safety pass over the paper draft. If time allows, the strongest additional experiment would be a small calibrated real-height subset or a controlled real reference sample, because that is the missing evidence needed to move from diagnostic alignment to real-height accuracy.
