# Confidence-Gated Prior Loss Runner Report

- Date: 2026-06-22
- Run id: `ABL-07`
- Variant: Confidence-gated DFF/GADFF prior loss
- Current artifact level: smoke only, claim-ineligible.
- Seed: `20260623`

## Rationale

Earlier mechanism probes show that DFF reliability varies with focus-curve confidence and that glare/risk alone is not a stable universal quality signal. This runner keeps the supervised ground-truth term uniform and uses confidence to gate only the DFF/GADFF consistency term.

## Channel and Loss Mapping

| Channel | Meaning | Loss role |
|---|---|---|
| 33 | glare/risk prior | softens prior consistency in high-risk regions |
| 34 | DFF depth prior | fused prior target |
| 35 | DFF confidence | dominant prior gate |
| 36 | GADFF depth prior | fused prior target |
| 37 | GADFF confidence | secondary prior gate |

## Smoke Result

- Epochs: `12`
- Prepared train samples: `27`
- Prepared validation samples: `10`
- Last validation MAE norm: `0.06470213`
- Last validation focus confidence mean: `0.31773595`
- Last validation prior weight mean: `0.18933724`
- Checkpoint: `tmp\ablation_results\ABL-07\checkpoints\2026-06-22_confidence_gated_prior_seed_repeat.pt`
- Metrics: `tmp\ablation_results\ABL-07\metrics\2026-06-22_confidence_gated_prior_seed_repeat_metrics.csv`

## Claim Boundary

This result only proves that the new loss passes data generation, feature augmentation, forward, backward, checkpoint, and logging. It should not be used as a paper table value until a full matched-split run, external evaluation, seed repeat, and eligibility audit pass.

## Next Full Candidate Command

```powershell
python -B submission_planning/tools/run_confidence_weighted_loss_training.py --tag 2026-06-22_confidence_gated_prior_full_candidate --max-epochs 12 --train-patches 384 --val-patches 96 --batch-size 6 --max-train-samples 27 --max-val-samples 10 --device cpu
```
