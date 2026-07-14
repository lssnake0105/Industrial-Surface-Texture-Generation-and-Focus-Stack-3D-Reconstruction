# Batch External Evaluation Utility

Updated: 2026-06-18

Purpose: provide a table-generation entry point for future DFV/DDFFNet predictions across multiple exported synthetic samples.

## Tool

`submission_planning/tools/evaluate_external_prediction_batch.py`

The tool reads:

1. an exported-sample manifest, default `tmp/external_baseline_data/manifest.csv`;
2. a prediction manifest listing method, sample, prediction path, scale mode, and training setting.

It writes:

```text
tmp/external_baseline_results/batch_evaluation/
  per_sample_metrics.csv
  per_sample_metrics.json
  method_summary_metrics.csv
  method_summary_metrics.json
  README.md
```

## Prediction Manifest Format

```csv
method,training_setting,sample_id,prediction_path,scale_mode,clip
DFF_prior_export_smoke,internal_prior,test_V谷_P10_宽谷粗糙平底,tmp/external_baseline_data/samples/test_V谷_P10_宽谷粗糙平底/priors/dff_depth.npy,raw_norm,false
```

Template:

```text
submission_planning/autonomous_research/templates/external_prediction_manifest_template.csv
```

## Smoke-Test Command

```powershell
python -X utf8 submission_planning/tools/evaluate_external_prediction_batch.py `
  --prediction-manifest tmp/external_baseline_results/prediction_manifests/p10_prior_predictions.csv `
  --out-dir tmp/external_baseline_results/batch_evaluation/p10_prior_smoke
```

## Smoke-Test Result

| Method | Training | Scale | Samples | Mean MAE | Mean Edge MAE | Mean High-risk MAE |
|---|---|---|---:|---:|---:|---:|
| DFF_prior_project_metric_smoke | internal_prior | raw_norm | 1 | 160.1707 | 270.7708 | 39.2270 |
| GADFF_prior_project_metric_smoke | internal_prior | raw_norm | 1 | 167.4631 | 282.0009 | 39.2270 |

Output directory:

```text
tmp/external_baseline_results/batch_evaluation/p10_prior_smoke/
```

## Boundary

This tool only aggregates predictions that already exist. It does not run DFV/DDFFNet, download external repositories, or create SOTA results by itself. A method should enter the manuscript main table only after predictions cover the fixed evaluation split and the run log documents training setting, code source, input frame count, and scale alignment.
