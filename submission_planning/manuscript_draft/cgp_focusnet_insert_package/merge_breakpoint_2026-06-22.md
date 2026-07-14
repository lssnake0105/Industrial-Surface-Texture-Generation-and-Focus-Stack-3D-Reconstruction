# CGP-FocusNet Merge Breakpoint

- Date: 2026-06-22
- Status: insert package drafted, compiled, and claim-audited.
- Main artifact: `cgp_focusnet_manuscript_insert.tex`
- Verified PDF: `build_verify3/cgp_focusnet_manuscript_insert.pdf`

## Current Decision

Use `CGP-FocusNet` as the preferred manuscript method name for the next revision. It expands to Confidence-Gated Prior FocusNet and matches the current ABL-07 evidence chain better than the older `S2R-FocusNet` or `Focus-ResUNet` naming.

## What Is Ready to Merge

1. Abstract replacement draft.
2. Introduction contribution list.
3. Method insert for 38-channel input representation.
4. Method insert for confidence-gated DFF/GADFF prior objective.
5. Synthetic quantitative table for the two ABL-07 runs.
6. Stratified mechanism table showing low-confidence focus-region gains.
7. Real-stack diagnostic alignment table.
8. Discussion paragraph defining claim boundaries.

## Verification

- LaTeX compile: passed with TeX Live.
- Build directory: `submission_planning/manuscript_draft/cgp_focusnet_insert_package/build_verify3/`
- Claim-safety audit: passed, 13 checks, 0 errors, 0 warnings.
- Source files checked:
  - `cgp_focusnet_manuscript_insert.tex`
  - `README.md`
  - `claim_alignment_matrix.md`

## Merge Plan for Main Manuscript

1. Replace the method macro in `s2r_focus_stack_manuscript.tex`:
   - from `S2R-FocusNet`
   - to `CGP-FocusNet`

2. Replace the old abstract with the insert-package abstract.

3. Update method sections:
   - add the 38-channel input representation;
   - add the confidence-gated prior objective;
   - describe the glare/risk cue as a softening cue for prior consistency.

4. Update experiment sections:
   - replace older Focus-ResUNet synthetic numbers with ABL-07 guarded numbers;
   - add the low-confidence stratum table;
   - add real-stack diagnostic alignment as a separate no-reference result.

5. Keep old Focus-ResUNet / TinyDepthNet / residual variants as internal baselines or implementation history.

6. Run final main-draft claim audit after merging.

## Claim Boundary to Preserve

Safe:

> CGP-FocusNet improves synthetic height prediction and aligns with no-reference real-stack diagnostics by suppressing unstable DFF behavior in low-confidence and spike-prone regions.

Guarded:

- ABL-07 synthetic values are audit-passed internal evidence, but final table placement still needs paper-level review.
- Real-stack results support diagnostic alignment only.

Avoid:

- calibrated real-height accuracy on real samples;
- external SOTA superiority;
- equating no-reference stability with geometric correctness;
- presenting glare/risk as the primary mechanism without further ablation.

