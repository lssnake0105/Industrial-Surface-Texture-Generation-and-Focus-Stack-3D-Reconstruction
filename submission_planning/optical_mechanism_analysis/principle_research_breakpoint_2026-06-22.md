# Principle Research Breakpoint

- Date: 2026-06-22
- Current focus: ABL-07 confidence-gated DFF/GADFF prior consistency
- Current status: synthetic quantitative evidence passed; real-stack no-reference alignment passed; manuscript claim remains guarded.

## Current Evidence Chain

1. Synthetic GT evaluation supports ABL-07 as the current main internal candidate.
   - Full candidate: mean MAE 55.9286 um, gain vs DFF 44.38%.
   - Seed repeat: mean MAE 66.0954 um, gain vs DFF 34.27%.
   - Both remain below the previous ABL-04 reference threshold of 75.4572 um.

2. Low-confidence strata analysis gives the clearest mechanism.
   - Full candidate low-confidence gain: 55.64%.
   - Seed repeat low-confidence gain: 49.74%.
   - This supports writing the model around confidence-gated prior consistency rather than direct glare weighting.

3. Real-stack no-reference diagnostics support Simulation-to-Real alignment.
   - Seven real focus stacks were evaluated.
   - Two ABL-07 checkpoints were tested.
   - Mean low-confidence local-deviation reduction: 94.85% and 95.32%.
   - Mean spike-top10 local-deviation reduction: 96.25% and 96.93%.
   - Mean saturated-region local-deviation reduction: 83.16% and 81.40%.
   - Real-stack alignment audit: pass, 34 checks, 0 errors, 0 warnings.

4. Claim safety remains explicit.
   - Extended claim-safety audit: pass, 15 checks, 0 errors, 0 warnings.
   - Real-height accuracy claim remains unsupported without calibrated real height ground truth.
   - External SOTA superiority remains unsupported until updated baselines are run under compatible conditions.

## Current Paper Story

Recommended story line:

> The project should be framed as a Simulation-to-Real depth-from-focus study for glare-prone industrial surfaces. The central method is a confidence-gated DFF/GADFF prior consistency model. Synthetic GT experiments provide quantitative height-error evidence, low-confidence strata explain the mechanism, and real-stack no-reference diagnostics show that the same mechanism suppresses DFF instability on real samples.

中文叙事：

> 当前论文应围绕“面向强反光工业表面的 Simulation-to-Real DFF 深度恢复”展开。核心方法收束为 confidence-gated DFF/GADFF prior consistency model。合成 GT 实验提供定量高度误差证据，低置信分层分析解释收益机制，真实焦栈无参考诊断说明该机制在真实样本上同样能抑制 DFF 不稳定性。

## Claim Boundary

Safe claim:

> ABL-07 improves synthetic height prediction and aligns with no-reference real-stack diagnostics by suppressing DFF instability in low-confidence and spike-prone regions.

Unsafe claims for now:

- calibrated real-height accuracy on real samples;
- superiority over updated external SOTA;
- final manuscript-table readiness without paper-level claim review;
- causal proof that every real-sample smoothing effect is physically correct surface recovery.

## Next Best Actions

1. Collapse method naming into one model name.
   - Avoid presenting too many intermediate model products.
   - Suggested name: Confidence-Gated Prior FocusNet, or CGP-FocusNet.

2. Rewrite the manuscript experiment section around three evidence layers.
   - Synthetic GT quantitative table.
   - Low-confidence mechanism/strata table.
   - Real-stack no-reference diagnostic table and visual panel.

3. Update Related Work and external baselines.
   - Keep Depth Anything and modern monocular depth as context, but avoid direct SOTA comparison unless the task setting is compatible.
   - Re-check recent DFF, focus-stack depth, reflection/glare-robust reconstruction, and industrial surface metrology baselines.

4. Prepare final claim-safety review on the actual paper draft.
   - Search for real-accuracy, SOTA, best, final, and outperform wording.
   - Make every real-stack claim diagnostic or qualitative unless calibrated GT is added.

## Key Files

- Real-stack decision note: `submission_planning/optical_mechanism_analysis/abl07_real_stack_alignment_decision_note.md`
- Real-stack report: `submission_planning/optical_mechanism_analysis/abl07_real_stack_alignment/abl07_real_stack_alignment_report.md`
- Real-stack audit: `tmp/ablation_results/eligibility_audits/ABL07_real_stack_alignment_audit.md`
- Extended claim-safety audit: `tmp/ablation_results/eligibility_audits/ABL07_claim_safety_audit.md`
- ABL-07 closure note: `submission_planning/optical_mechanism_analysis/abl07_audit_closure_note.md`

