# Manuscript Claim Safety Audit

Updated: 2026-06-18

Purpose: define a lightweight audit for the current LaTeX manuscript draft. The audit checks whether the draft stays inside the current evidence boundary and avoids unsupported claims about external SOTA, real absolute accuracy, component ablation, and broad industrial generalization.

## 1. Target Manuscript

```text
submission_planning/manuscript_draft/s2r_focus_stack_manuscript.tex
```

## 2. Current Evidence Boundary

The current manuscript may safely claim:

1. synthetic focus stacks provide known height maps for controlled quantitative evaluation;
2. Focus-ResUNet has the lowest mean MAE and edge MAE among current internal baselines;
3. real samples are evaluated with no-reference morphology metrics;
4. real results support spike suppression and output stability, not calibrated height accuracy;
5. DFV and DDFFNet are priority external baselines, with results still pending;
6. Depth Anything V2 is a training-strategy and single-frame auxiliary-prior reference.

The manuscript should not claim:

1. superiority over DFV, DDFFNet, HybridDepth, or external deep DFF SOTA;
2. real absolute height accuracy;
3. validated module contribution before ablation experiments;
4. domain randomization or pseudo-label training gains before those experiments exist;
5. generalization to all industrial surfaces or all microscopy systems.
6. Depth Anything V2 auxiliary outputs as focus-stack main-table evidence.

## 3. Audit Tool

Use:

```powershell
python -X utf8 submission_planning/tools/audit_manuscript_claim_safety.py
```

Default outputs:

```text
tmp/manuscript_audits/manuscript_claim_safety_audit.md
tmp/manuscript_audits/manuscript_claim_safety_audit.json
```

## 4. Interpretation

The audit is a text-safety check. Passing it means the draft does not contain the most obvious unsupported claim patterns. It does not prove that the manuscript is scientifically complete, because external SOTA results, ablations, and calibrated real validation may still be missing.
