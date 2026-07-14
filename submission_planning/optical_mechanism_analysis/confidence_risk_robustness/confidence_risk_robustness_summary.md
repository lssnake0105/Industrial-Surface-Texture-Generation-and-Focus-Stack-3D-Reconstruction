# Focus Confidence Robustness Summary

Cases: 135

| Score | AUC mean | AUC std | AUC>0.5 | AUC>0.55 | Error lift mean | Failure rate top10 mean |
|---|---:|---:|---:|---:|---:|---:|
| low_margin | 0.6412 | 0.0133 | 1.00 | 1.00 | 0.0078 | 0.1568 |
| hybrid_confidence | 0.5447 | 0.0364 | 0.95 | 0.40 | 0.0046 | 0.1109 |
| hybrid_risk_confidence | 0.5324 | 0.0421 | 0.68 | 0.27 | 0.0027 | 0.0875 |
| risk_max | 0.5087 | 0.0540 | 0.46 | 0.16 | 0.0061 | 0.0796 |
| risk_mean | 0.5075 | 0.0552 | 0.45 | 0.16 | 0.0063 | 0.0747 |
| sat_persistence | 0.4961 | 0.0064 | 0.23 | 0.00 | -0.0002 | 0.0974 |
| bright_persistence | 0.4759 | 0.0398 | 0.19 | 0.00 | -0.0049 | 0.0873 |
| focus_entropy | 0.4438 | 0.0419 | 0.12 | 0.00 | 0.0010 | 0.0867 |
| low_peak_strength | 0.3858 | 0.0315 | 0.00 | 0.00 | -0.0017 | 0.0308 |

## Key Finding

The best mean AUC is `low_margin` (0.6412 +/- 0.0133).
Focus-confidence diagnostics should be treated as probabilistic quality cues rather than deterministic failure masks.
