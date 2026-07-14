# Manuscript Draft Notes

This folder contains an independent plain-LaTeX manuscript draft for the simulation-to-real focus-stack reconstruction submission plan.

## Files

- `s2r_focus_stack_manuscript.tex`: first article-style manuscript draft.
- `references.bib`: local bibliography entries used by the draft.

## Scope

The draft is intentionally separate from `Updated_English_Project_Paper.tex` and the original `zjuthesis-master` materials. It does not modify source data, model outputs, figures, or the existing project paper.

## Current Claim Boundary

- Synthetic results can report absolute height errors because synthetic height maps are available.
- Real results should report no-reference morphology stability only.
- The current draft does not claim superiority over external SOTA methods.
- DFV and DDFFNet results are still needed before a strong SOTA comparison.
- Ablation results are still needed before claiming the independent effect of each module.
- Depth Anything V2 is included as a simulation-to-real training-strategy reference and single-image auxiliary prior, not as a direct focus-stack baseline.

## Next Submission Tasks

1. Decide the final method name: `S2R-FocusNet` or `Prior-Guided Focus-ResUNet`.
2. Verify exact input channels and implemented loss terms from the training code.
3. Add DFV and DDFFNet results under the same synthetic split.
4. Add ablations for priors, focal-difference representation, and glare/high-risk cues.
5. Compile the manuscript after figure paths and bibliography style are fixed for the target venue.
