# CGP-FocusNet Claim Alignment Matrix

- Date: 2026-06-22
- Scope: claims introduced by `cgp_focusnet_manuscript_insert.tex`

| Claim | Evidence | Current Level | Safe Placement |
|---|---|---|---|
| CGP-FocusNet uses 17 stack layers, 16 adjacent focal-difference layers, and 5 prior channels | `src/train_focus_resunet_loss_experiment.py`, `run_confidence_weighted_loss_training.py` | implementation-verified | Method |
| The loss keeps the supervised data term uniform and gates only DFF/GADFF prior consistency | `ConfidenceGatedPriorLoss` implementation | implementation-verified | Method |
| ABL-07 full candidate achieves 55.93 um mean MAE on the fixed synthetic test split | method summary CSV | audit-passed internal evidence | Results, guarded table |
| ABL-07 seed repeat achieves 66.10 um mean MAE | method summary CSV | audit-passed internal evidence | Results, guarded table |
| The clearest mechanism appears in low-confidence focus regions | stratum summary CSV, both seeds | audit-passed internal evidence | Results/Discussion |
| Real-stack diagnostics show suppression of DFF local-deviation spikes | real-stack alignment CSV and audit | audit-passed diagnostic evidence | Results/Discussion |
| Real-stack results prove calibrated real-height accuracy | no calibrated real GT | unsupported | avoid |
| CGP-FocusNet outperforms updated external SOTA | external baselines not yet run | unsupported | avoid |

## Minimal Next Gate Before Updating Main Manuscript

1. Decide whether `CGP-FocusNet` should replace `S2R-FocusNet` globally.
2. Add the new tables and method objective to the main LaTeX draft.
3. Run LaTeX compile.
4. Run claim-safety audit over the updated main draft.
5. If external baseline experiments are added later, keep them in a separate table until protocol compatibility is verified.

