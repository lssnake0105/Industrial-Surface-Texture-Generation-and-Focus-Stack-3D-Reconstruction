# ABL-07 Decision Note: Confidence-Gated DFF/GADFF Prior

- Date: 2026-06-22
- Scope: mechanism-oriented preflight for the reflective-surface DFF failure story
- Artifact level: smoke and diagnostic only; not manuscript evidence

## Core Decision

The confidence-gated prior loss has advanced from smoke preflight to a full matched candidate run. It should now be treated as the leading mechanism candidate, with the next gate being seed repeat, full eligibility audit, and real-stack diagnostic alignment.

## Evidence Snapshot

| Evidence | Result | Interpretation |
|---|---:|---|
| ABL-07 standalone smoke last val MAE norm | 0.09089682 | The isolated loss runner can train through the current FocusResUNet interface. |
| Paired smoke, gated minus baseline val MAE | +0.00131364 | Under one random validation patch, baseline is slightly better by 0.62%. |
| Stratified smoke high-risk delta | -0.00697670 | In high-risk patches, confidence-gated checkpoint reduces MAE by 12.15%. |
| Full candidate validation MAE norm | 0.04790895 | Full train/validation split training completed successfully on CUDA. |
| Full candidate test MAE | 55.9286 um | ABL-07 outperforms prior longer-repeat ABL-04 test MAE of 75.4572 um. |
| Full candidate low-confidence gain vs DFF | 55.64% | The strongest stratified gain appears in unreliable focus regions. |

## Mechanism Reading

The paired smoke by itself was weak because its validation patch did not activate a high-risk condition. The full candidate result is stronger: ABL-07 improves overall synthetic test MAE and shows its most stable gain in low-confidence focus regions. This matches the desired story that the DFF prior should be treated as a confidence-aware observation rather than as a uniformly trusted target.

## Claim Boundary

These results justify elevating ABL-07 to the main candidate, but they should not enter a final paper table, abstract, or contribution statement yet. A defensible claim still requires seed repeat, real-stack no-reference diagnostics, and an eligibility audit.

## Recommended Next Experiment

Run a seed repeat or longer repeat using the same train/validation split and evaluation protocol:

```powershell
python -B submission_planning/tools/run_confidence_weighted_loss_training.py --tag 2026-06-22_confidence_gated_prior_seed_repeat --max-epochs 12 --train-patches 384 --val-patches 96 --batch-size 6 --max-train-samples 27 --max-val-samples 10 --device cuda
python -B submission_planning/tools/evaluate_confidence_gated_prior_full_split.py --tag 2026-06-22_confidence_gated_prior_seed_repeat_eval --checkpoint-tag 2026-06-22_confidence_gated_prior_seed_repeat --max-samples 0 --tile 256 --overlap 80 --device cuda
```

Then compare three tables:

1. overall held-out synthetic MAE and RMSE against ABL-04 and ABL-07 full candidate;
2. region-stratified metrics for high-risk, low-confidence, and normal areas;
3. real-stack no-reference diagnostics aligned to focus-curve morphology.

## Decision Rule

Keep ABL-07 as the manuscript candidate if the repeat remains near or below the previous best ABL-04 threshold of 75.46 um mean MAE and preserves the low-confidence gain. Redesign it if the repeat collapses, shifts error into normal regions, or loses the confidence-stratified advantage.
