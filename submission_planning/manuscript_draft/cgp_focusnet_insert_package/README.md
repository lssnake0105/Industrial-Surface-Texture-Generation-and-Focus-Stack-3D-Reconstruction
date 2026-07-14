# CGP-FocusNet Manuscript Insert Package

- Date: 2026-06-22
- Purpose: merge the latest ABL-07 mechanism evidence into the paper without overwriting the older manuscript draft.
- Main insert file: `cgp_focusnet_manuscript_insert.tex`

## Recommended Paper-Level Update

Use `CGP-FocusNet` as the unified method name:

> Confidence-Gated Prior FocusNet

This name keeps the contribution centered on the verified mechanism: confidence-gated DFF/GADFF prior consistency. Older names such as Focus-ResUNet and S2R-FocusNet can remain as implementation/backbone history or internal variants.

## Evidence Used

| Evidence layer | Source | Status |
|---|---|---|
| Synthetic full-split metrics | `tmp/ablation_results/confidence_gated_full_split_eval/*_method_summary_metrics.csv` | audit-passed internal evidence |
| Stratified mechanism metrics | `tmp/ablation_results/confidence_gated_full_split_eval/*_stratum_summary_metrics.csv` | audit-passed internal evidence |
| Real-stack diagnostic alignment | `submission_planning/optical_mechanism_analysis/abl07_real_stack_alignment/*` | audit-passed real-stack diagnostic evidence |
| Claim-safety audit | `tmp/ablation_results/eligibility_audits/ABL07_claim_safety_audit.md` | pass, 15 checks |
| Real-stack alignment audit | `tmp/ablation_results/eligibility_audits/ABL07_real_stack_alignment_audit.md` | pass, 34 checks |

## How to Merge Into the Existing Draft

Recommended replacement targets in `submission_planning/manuscript_draft/s2r_focus_stack_manuscript.tex`:

1. Replace `\newcommand{\method}{S2R-FocusNet}` with `\newcommand{\method}{CGP-FocusNet}`.
2. Replace the abstract with the `Abstract Replacement Draft`.
3. Replace the contribution list with the `Introduction Contribution Replacement`.
4. Update `Method` to include the 38-channel input representation and confidence-gated prior objective.
5. Replace or supplement the old synthetic table with `tab:cgp_synthetic`.
6. Add the stratified mechanism table `tab:cgp_strata`.
7. Replace the old real no-reference table with `tab:cgp_real_alignment` or add it as a separate real-stack diagnostic table.
8. Keep the claim-boundary paragraph in Discussion.

## Figure Suggestions

Safe figure candidates:

- Pipeline figure: `output/imagegen/focus_stack_dff_pipeline_v2.png`
- Network figure: `output/imagegen/focus-resunet-industrial-network-diagram-final-2048x1152.png`
- Real-stack diagnostic visual panels: `submission_planning/optical_mechanism_analysis/abl07_real_stack_alignment/*_alignment.png`

For the paper, prefer one representative real-stack diagnostic panel rather than all generated panels. A good candidate is `钥匙纹路100um` because its confident-region correlation is high and the defect texture is visually interpretable.

## Claim Boundary

Safe claim:

> CGP-FocusNet improves synthetic height prediction and aligns with no-reference real-stack diagnostics by suppressing unstable DFF behavior in low-confidence and spike-prone regions.

Current unsafe claims:

- calibrated real-height accuracy on real samples;
- superiority over updated external SOTA;
- final manuscript-table readiness before paper-level claim review;
- physical correctness of every smoothed real-stack region.

