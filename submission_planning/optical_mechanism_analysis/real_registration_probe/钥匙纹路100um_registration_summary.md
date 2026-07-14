# Real Focus-Stack Registration Probe: 钥匙纹路100um

Reference layer: 20

## Shift Summary

- Max shift magnitude: 0.1118 px
- Median shift magnitude: 0.0000 px
- X shift range: -0.1000 to 0.0500 px
- Y shift range: -0.0500 to 0.0500 px

## Registration Sensitivity

| Metric | Value |
|---|---:|
| `peak_layer_changed_fraction` | 0.0468 |
| `peak_layer_mean_abs_change` | 0.3875 |
| `peak_layer_p90_abs_change` | 0.0000 |
| `spike_proxy_mean_before` | 0.0995 |
| `spike_proxy_mean_after` | 0.1041 |
| `spike_proxy_mean_delta` | 0.0046 |
| `spike_proxy_pearson_before_after` | 0.9478 |
| `quality_proxy_mean_before` | 0.6375 |
| `quality_proxy_mean_after` | 0.5976 |
| `quality_proxy_pearson_before_after` | 0.9825 |
| `low_margin_auc_spike_top10_before` | 0.7293 |
| `low_margin_auc_spike_top10_after` | 0.7287 |
| `quality_proxy_auc_spike_top10_before` | 0.9289 |
| `quality_proxy_auc_spike_top10_after` | 0.9299 |
| `sat_persistence_auc_spike_top10_before` | 0.5384 |
| `sat_persistence_auc_spike_top10_after` | 0.5398 |

## Interpretation

The estimated inter-layer shifts should be interpreted as a global registration diagnostic, not as a full optical calibration.
If the peak-layer and quality maps remain highly correlated after alignment, the observed DFF instability is unlikely to be explained only by small global translations.
