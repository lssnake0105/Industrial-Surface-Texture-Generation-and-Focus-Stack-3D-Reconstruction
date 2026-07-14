# DFF Resolution Sensitivity Summary

| Factor | Mode | MAE | RMSE | P90 | Edge MAE | High-risk MAE | Peak layer mean | Risk mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | superres_integrated | 0.0696 | 0.0827 | 0.1224 | 0.0705 | 0.0804 | 12.49 | 0.0150 |
| 2 | direct_lowres | 0.0688 | 0.0878 | 0.1216 | 0.0628 | 0.0914 | 12.46 | 0.0216 |
| 4 | superres_integrated | 0.0654 | 0.0759 | 0.1162 | 0.0719 | 0.0686 | 12.58 | 0.0108 |
| 4 | direct_lowres | 0.0717 | 0.0913 | 0.1247 | 0.0659 | 0.0963 | 12.54 | 0.0254 |
| 8 | superres_integrated | 0.0624 | 0.0717 | 0.1118 | 0.0664 | 0.0624 | 12.43 | 0.0062 |
| 8 | direct_lowres | 0.0748 | 0.0989 | 0.1272 | 0.0651 | 0.1057 | 12.34 | 0.0288 |

## Interpretation

This study measures how the simulation sampling strategy propagates into DFF depth selection.
The target comparison is not absolute optical realism; it isolates whether computing normals and glare at sensor resolution changes depth errors relative to a super-resolution integrated pipeline.
