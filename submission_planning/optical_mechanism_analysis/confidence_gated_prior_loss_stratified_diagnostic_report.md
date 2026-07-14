# ABL-07 Stratified Diagnostic: Paired Smoke Checkpoints

- Date: 2026-06-22
- Run id: `ABL-07`
- Source tag: `2026-06-22_paired_loss_smoke`
- Artifact level: `smoke_checkpoint_diagnostic_claim_ineligible`

## Stratum Summary

| Variant | Stratum | N | Risk Mean | Focus Conf Mean | MAE Norm | P90 Abs Error |
|---|---|---:|---:|---:|---:|---:|
| Baseline HybridDFFLoss | high_risk | 3 | 0.981467 | 0.385110 | 0.05740696 | 0.10614591 |
| Baseline HybridDFFLoss | low_confidence | 3 | 0.030553 | 0.227418 | 0.17792278 | 0.31566142 |
| Baseline HybridDFFLoss | normal | 3 | 0.025044 | 0.322173 | 0.20699534 | 0.29861238 |
| Confidence-gated prior loss | high_risk | 3 | 0.981467 | 0.385110 | 0.05043026 | 0.08833828 |
| Confidence-gated prior loss | low_confidence | 3 | 0.030553 | 0.227418 | 0.17415300 | 0.31293452 |
| Confidence-gated prior loss | normal | 3 | 0.025044 | 0.322173 | 0.20718053 | 0.29294085 |

## Delta: Confidence-Gated Minus Baseline

| Stratum | Baseline MAE | Confidence-Gated MAE | Delta | Relative Delta | Preferred |
|---|---:|---:|---:|---:|---|
| high_risk | 0.05740696 | 0.05043026 | -0.00697670 | -12.15% | confidence_gated |
| low_confidence | 0.17792278 | 0.17415300 | -0.00376978 | -2.12% | confidence_gated |
| normal | 0.20699534 | 0.20718053 | 0.00018518 | 0.09% | baseline_hybrid |

## Interpretation Boundary

This diagnostic evaluates only two smoke checkpoints trained on a tiny budget. It is useful for deciding whether the proposed mechanism deserves a full candidate run, but it remains outside manuscript evidence.

## Artifacts

| Artifact | Path |
|---|---|
| detail_csv | `tmp\ablation_results\ABL-07\stratified_diagnostics\2026-06-22_stratified_diagnostic_detail.csv` |
| summary_csv | `tmp\ablation_results\ABL-07\stratified_diagnostics\2026-06-22_stratified_diagnostic_summary.csv` |
| summary_json | `tmp\ablation_results\ABL-07\stratified_diagnostics\2026-06-22_stratified_diagnostic_summary.json` |
| mechanism_report | `submission_planning\optical_mechanism_analysis\confidence_gated_prior_loss_stratified_diagnostic_report.md` |
| baseline_checkpoint | `tmp\ablation_results\ABL-07\checkpoints\2026-06-22_paired_loss_smoke_baseline_hybrid.pt` |
| confidence_gated_checkpoint | `tmp\ablation_results\ABL-07\checkpoints\2026-06-22_paired_loss_smoke_confidence_gated.pt` |
