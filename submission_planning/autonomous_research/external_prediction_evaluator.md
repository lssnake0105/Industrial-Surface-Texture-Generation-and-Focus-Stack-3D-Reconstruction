# External Prediction Evaluator

Updated: 2026-06-18

Purpose: prepare a reusable metric entry point for future DFV/DDFFNet predictions after an external model produces a `.npy` depth or height map.

## Tool

`submission_planning/tools/evaluate_external_prediction.py`

Example:

```powershell
python -X utf8 submission_planning/tools/evaluate_external_prediction.py `
  --prediction tmp/external_baseline_data/samples/test_V谷_P10_宽谷粗糙平底/priors/dff_depth.npy `
  --method DFF_prior_export_smoke `
  --scale-mode raw_norm `
  --out-dir tmp/external_baseline_results/evaluation_smoke/dff_prior
```

## Supported Prediction Format

| Field | Rule |
|---|---|
| File type | `.npy` |
| Shape | 2D `[H, W]`; singleton dimensions can be squeezed |
| Default unit | normalized height `[0, 1]` |
| Synthetic GT | `height_gt.npy` from the exported sample package |
| Output metrics | MAE, RMSE, P90, edge MAE, high-risk MAE |

## Scale Alignment Modes

| Mode | Use Case |
|---|---|
| `raw_norm` | prediction already uses normalized project height |
| `minmax` | relative prediction with arbitrary range |
| `affine` | relative prediction needing least-squares affine alignment to synthetic GT |
| `scale_to_um` | reserved for outputs already interpreted as micrometer height |

## Smoke-Test Results

The evaluator was checked with exported P10 DFF/GADFF priors and synchronized with `src/simulate_antiglare_highres_samples.py::metrics`.

| Prediction | MAE | P90 | Edge MAE | High-risk MAE |
|---|---:|---:|---:|---:|
| DFF prior | 160.1707 um | 515.3698 um | 270.7708 um | 39.2270 um |
| GADFF prior | 167.4631 um | 532.5328 um | 282.0009 um | 39.2270 um |

These values align with the existing P10 comparison table for Original DFF and GADFF. The synchronized mask rules are:

- high-risk mask: `risk > max(percentile(risk, 84), 0.08)`;
- edge mask: Sobel gradient magnitude on GT height, thresholded by percentile 88.

## Manuscript Boundary

This evaluator prepares the metric pipeline. It does not provide DFV/DDFFNet results by itself and should not be cited as an external SOTA comparison until real external predictions are evaluated under the fixed test split.
