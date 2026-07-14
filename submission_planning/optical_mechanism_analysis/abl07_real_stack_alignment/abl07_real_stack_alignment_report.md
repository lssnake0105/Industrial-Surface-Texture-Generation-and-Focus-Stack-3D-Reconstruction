# ABL-07 Real Focus-Stack Alignment Diagnostics

- Date: 2026-06-22
- Checkpoints: `['2026-06-22_confidence_gated_prior_full_candidate', '2026-06-22_confidence_gated_prior_seed_repeat']`
- Evidence type: no-reference real-stack diagnostic
- Claim boundary: no real calibrated height ground truth is used here.

## Aggregate Summary

| Checkpoint | Stacks | Low-conf dev reduction | Spike-top10 dev reduction | Saturated dev reduction | Confident corr. | Confident std ratio |
|---|---:|---:|---:|---:|---:|---:|
| 2026-06-22_confidence_gated_prior_full_candidate | 7 | 94.85% | 96.25% | 83.16% | 0.5073 | 0.6299 |
| 2026-06-22_confidence_gated_prior_seed_repeat | 7 | 95.32% | 96.93% | 81.40% | 0.4213 | 0.4727 |

## Per-Stack Core Metrics

| Stack | Checkpoint | Low-conf reduction | Spike-top10 reduction | Quality-top10 reduction | Confident corr. |
|---|---|---:|---:|---:|---:|
| 1124 | 2026-06-22_confidence_gated_prior_full_candidate | 94.74% | 97.02% | 95.99% | 0.3961 |
| 1124 | 2026-06-22_confidence_gated_prior_seed_repeat | 94.54% | 97.47% | 96.08% | 0.0342 |
| 3D层纹 | 2026-06-22_confidence_gated_prior_full_candidate | 97.85% | 98.10% | 98.12% | 0.0899 |
| 3D层纹 | 2026-06-22_confidence_gated_prior_seed_repeat | 98.66% | 98.90% | 98.91% | 0.0933 |
| 3D表面 | 2026-06-22_confidence_gated_prior_full_candidate | 98.27% | 98.40% | 98.37% | 0.4749 |
| 3D表面 | 2026-06-22_confidence_gated_prior_seed_repeat | 98.45% | 98.51% | 98.50% | 0.4754 |
| 圆孔50um | 2026-06-22_confidence_gated_prior_full_candidate | 97.56% | 97.60% | 97.62% | 0.2678 |
| 圆孔50um | 2026-06-22_confidence_gated_prior_seed_repeat | 98.82% | 98.85% | 98.85% | 0.3089 |
| 磕碰孔5um | 2026-06-22_confidence_gated_prior_full_candidate | 90.96% | 93.77% | 92.34% | 0.4995 |
| 磕碰孔5um | 2026-06-22_confidence_gated_prior_seed_repeat | 91.99% | 95.29% | 93.54% | 0.3103 |
| 钥匙尖头50um | 2026-06-22_confidence_gated_prior_full_candidate | 87.61% | 90.93% | 89.20% | 0.8589 |
| 钥匙尖头50um | 2026-06-22_confidence_gated_prior_seed_repeat | 87.56% | 91.40% | 89.41% | 0.8013 |
| 钥匙纹路100um | 2026-06-22_confidence_gated_prior_full_candidate | 97.00% | 97.93% | 97.83% | 0.9639 |
| 钥匙纹路100um | 2026-06-22_confidence_gated_prior_seed_repeat | 97.22% | 98.11% | 98.03% | 0.9256 |

## Interpretation

Positive local-deviation reduction means ABL-07 is smoother than DFF in regions that DFF diagnostics mark as unreliable. This is useful only when the model still tracks DFF in confident regions. The confident-region correlation and standard-deviation ratio are therefore included as structure-retention checks.

This analysis does not prove absolute real-height accuracy. It only tests whether ABL-07 aligns with the no-reference real-stack failure diagnosis developed from focus-margin, spike, saturation, and morphology probes.
