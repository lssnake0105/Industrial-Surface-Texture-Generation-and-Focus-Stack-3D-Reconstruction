# Focus Confidence and Glare-Risk Study Summary

This branch evaluates whether focus-confidence and glare-risk maps can identify DFF failure regions in a synthetic super-resolution-integrated focal stack.

| Score | AUC failure top10 | Effect size | Mean error top10 score | Mean error rest | Failure rate in top10 score |
|---|---:|---:|---:|---:|---:|
| focus_entropy | 0.6111 | 0.3826 | 0.0765 | 0.0626 | 0.1434 |
| low_margin | 0.5907 | 0.3177 | 0.0665 | 0.0637 | 0.1391 |
| hybrid_risk_entropy | 0.5372 | 0.0093 | 0.0646 | 0.0640 | 0.0914 |
| sat_persistence | 0.5061 | 0.0000 | 0.0648 | 0.0639 | 0.1039 |
| low_peak_strength | 0.4920 | 0.0132 | 0.0710 | 0.0632 | 0.1125 |
| bright_persistence | 0.4629 | -0.2971 | 0.0510 | 0.0655 | 0.0238 |
| risk_max | 0.4458 | -0.1373 | 0.0662 | 0.0638 | 0.0820 |
| risk_mean | 0.4443 | -0.1140 | 0.0669 | 0.0637 | 0.0801 |

## Key Finding

The best single diagnostic in this run is `focus_entropy`, with AUC=0.6111 for identifying the top-10% DFF error pixels.
Hybrid risk/confidence scores are candidates for training masks, loss weighting, and no-reference real-sample diagnostics.
