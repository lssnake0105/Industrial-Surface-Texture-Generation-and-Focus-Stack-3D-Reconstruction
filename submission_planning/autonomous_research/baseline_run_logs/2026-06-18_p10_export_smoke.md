# P10 External Baseline Export Smoke Log

Date: 2026-06-18

## Run Identity

| Field | Value |
|---|---|
| Run ID | `p10_export_smoke_20260618` |
| Purpose | Prepare one synthetic sample package for external DFV/DDFFNet dataloader smoke tests |
| Script | `submission_planning/tools/export_one_external_baseline_sample.py` |
| Command | `python -X utf8 submission_planning/tools/export_one_external_baseline_sample.py --overwrite` |
| Output root | `tmp/external_baseline_data/` |
| Export mode | single-sample default |

## Exported Sample

| Field | Value |
|---|---|
| Sample ID | `test_V谷_P10_宽谷粗糙平底` |
| Split | `test` |
| Category | `P10 V谷-宽谷粗糙平底` |
| Resolution | `960x540` |
| Stack frames | 17 |
| z-step | 75.0 um |
| Depth range | 1200 um |

## Verified Files

| Artifact | Verification |
|---|---|
| `stack/000.png` to `stack/016.png` | 17 PNG frames exist |
| `height_gt.npy` | shape `(540, 960)`, dtype `float32`, range `[0.0, 1.0]` |
| `masks/high_risk_mask.npy` | shape `(540, 960)` |
| `masks/risk_layers.npy` | shape `(17, 540, 960)` |
| `priors/dff_depth.npy` | shape `(540, 960)` |
| `meta.json` | records focus positions, depth range, z-step, and source generator |
| `manifest.csv` | one-row manifest written under `tmp/external_baseline_data/` |

## Eligibility

This run is not eligible for the manuscript main comparison table. It is a dataloader smoke-test artifact only. A valid SOTA comparison still requires at least:

1. full train/validation/test export or a clearly justified external evaluation protocol;
2. DFV/DDFFNet run configuration and code version;
3. output scale-alignment rule;
4. MAE, edge MAE, high-risk MAE calculation on the fixed test split;
5. failure logs for any skipped samples.

## Next Step

Run a DFV dataloader-only smoke test against this exported P10 package before downloading weights or training models.

## Dataloader Smoke Test Result

| Field | Value |
|---|---|
| Script | `submission_planning/tools/smoke_test_external_baseline_package.py` |
| Command | `python -X utf8 submission_planning/tools/smoke_test_external_baseline_package.py` |
| Status | pass |
| Report | `tmp/external_baseline_data/smoke_reports/p10_dataloader_smoke.md` |
| Frame tensor | `[17, 540, 960]` |
| PyTorch-like grayscale tensor | `[1, 17, 1, 540, 960]` |
| PyTorch-like RGB tensor | `[1, 17, 3, 540, 960]` |

The exported P10 package can be read as a focal-stack sample. This still does not provide DFV/DDFFNet numerical results.

## Tool Update

The exporter now supports explicit larger temporary exports with `--split test` or `--all`. The default command remains single-sample P10 export to avoid accidental large-file generation.

## Prediction Evaluator Smoke Test

| Field | Value |
|---|---|
| Script | `submission_planning/tools/evaluate_external_prediction.py` |
| DFF prior command | `python -X utf8 submission_planning/tools/evaluate_external_prediction.py --prediction tmp/external_baseline_data/samples/test_V谷_P10_宽谷粗糙平底/priors/dff_depth.npy --method DFF_prior_export_smoke --scale-mode raw_norm --out-dir tmp/external_baseline_results/evaluation_smoke/dff_prior` |
| GADFF prior command | `python -X utf8 submission_planning/tools/evaluate_external_prediction.py --prediction tmp/external_baseline_data/samples/test_V谷_P10_宽谷粗糙平底/priors/gadff_depth.npy --method GADFF_prior_export_smoke --scale-mode raw_norm --out-dir tmp/external_baseline_results/evaluation_smoke/gadff_prior` |
| DFF prior MAE / edge / high-risk | 160.1707 um / 270.7708 um / 39.2270 um |
| GADFF prior MAE / edge / high-risk | 167.4631 um / 282.0009 um / 39.2270 um |

The MAE, P90, edge MAE, and high-risk MAE values align with the existing P10 comparison table. The evaluator now uses the same high-risk and Sobel-edge mask rules as the original project metrics function.

## Batch Evaluation Smoke Test

| Field | Value |
|---|---|
| Script | `submission_planning/tools/evaluate_external_prediction_batch.py` |
| Prediction manifest | `tmp/external_baseline_results/prediction_manifests/p10_prior_predictions.csv` |
| Output | `tmp/external_baseline_results/batch_evaluation/p10_prior_smoke/` |
| Status | pass |

The batch evaluator produced `per_sample_metrics.csv` and `method_summary_metrics.csv`. This confirms the downstream table-generation path for future DFV/DDFFNet predictions.
