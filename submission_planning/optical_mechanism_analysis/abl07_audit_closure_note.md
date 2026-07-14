# ABL-07 Audit Closure Note

- Date: 2026-06-22
- Scope: confidence-gated DFF/GADFF prior evidence audit and claim-safety audit
- Status: internal synthetic evidence passed; real-stack diagnostic alignment passed; manuscript-table claim remains guarded

## Audit Result

ABL-07 now has two full synthetic train/eval runs, both full-split evaluations, seed-repeat stability evidence, real-stack no-reference alignment diagnostics, and machine-readable audit records. The ABL-07 evidence audit passed 74 checks with 0 errors and 0 warnings. The extended claim-safety audit passed 15 checks with 0 errors and 0 warnings. The real-stack alignment audit passed 34 checks with 0 errors and 0 warnings.

## What Is Now Supported

ABL-07 can be used as the main internal candidate for the manuscript story. The supported mechanism statement is:

> Confidence-gated DFF/GADFF prior consistency is more stable than direct glare weighting, and its clearest gain appears in low-confidence focus regions.

This statement is supported by two synthetic runs:

| Run | Mean MAE um | Gain vs DFF | Low-Confidence Gain |
|---|---:|---:|---:|
| ABL-07 full candidate | 55.9286 | 44.38% | 55.64% |
| ABL-07 seed repeat | 66.0954 | 34.27% | 49.74% |

Both runs remain below the previous ABL-04 threshold of 75.4572 um.

## What Is Still Blocked

ABL-07 should not yet be promoted to calibrated real-height accuracy evidence. The real-stack evidence is no-reference diagnostic alignment and still lacks calibrated height ground truth. The correct boundary is:

- claim_eligible: false
- main_table_eligible: false
- manuscript-table use: allowed only as guarded internal/diagnostic evidence until the paper draft receives final claim-safety review

## Next Research Step

The next useful step is manuscript-level consolidation rather than another synthetic seed repeat. The method should be written as one confidence-gated prior consistency model, and the evidence chain should be organized as:

1. synthetic GT evaluation for quantitative height-error evidence;
2. low-confidence strata analysis for the mechanism of improvement;
3. real-stack no-reference diagnostics for Simulation-to-Real alignment;
4. final claim-safety review before moving any result into a paper table.

## Artifact Pointers

- Evidence audit: `tmp/ablation_results/eligibility_audits/ABL07_confidence_gated_evidence_audit.md`
- Claim-safety audit: `tmp/ablation_results/eligibility_audits/ABL07_claim_safety_audit.md`
- Stability report: `submission_planning/optical_mechanism_analysis/abl07_seed_repeat_stability_report.md`
- Full candidate evidence report: `submission_planning/optical_mechanism_analysis/abl07_full_candidate_evidence_report.md`
- Real-stack alignment report: `submission_planning/optical_mechanism_analysis/abl07_real_stack_alignment/abl07_real_stack_alignment_report.md`
- Real-stack decision note: `submission_planning/optical_mechanism_analysis/abl07_real_stack_alignment_decision_note.md`
- Real-stack audit: `tmp/ablation_results/eligibility_audits/ABL07_real_stack_alignment_audit.md`
