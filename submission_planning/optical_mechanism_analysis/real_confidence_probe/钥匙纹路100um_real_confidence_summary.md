# Real Focus-Stack Confidence Probe: 钥匙纹路100um

Layers: 40, resized shape: 512 x 640

## ROI Metrics

| ROI | peak layer mean | low margin mean | focus entropy mean | sat persistence mean | spike proxy mean | quality proxy mean |
|---|---:|---:|---:|---:|---:|---:|
| highlight_edge | 3.03 | 0.7910 | 0.8454 | 0.0364 | 0.0204 | 0.5877 |
| ordinary_texture | 18.88 | 0.8395 | 0.9757 | 0.0000 | 0.1174 | 0.6451 |
| dark_region | 18.49 | 0.8442 | 0.9794 | 0.0000 | 0.1223 | 0.6506 |

## Proxy Association

| Score | Spearman spike | AUC spike top10 | AUC saturation top10 | AUC early-peak top10 |
|---|---:|---:|---:|---:|
| low_margin | 0.4710 | 0.7293 | 0.5002 | 0.4420 |
| focus_entropy | 0.6136 | 0.7804 | 0.4091 | 0.2048 |
| low_peak_strength | 0.7085 | 0.8491 | 0.4649 | 0.2090 |
| sat_persistence | 0.1251 | 0.5384 | 1.0000 | 0.6728 |
| bright_persistence | 0.0738 | 0.5172 | 0.9780 | 0.6792 |
| quality_proxy | 0.6522 | 0.9289 | 0.5182 | 0.4367 |

## Interpretation

These values are no-reference proxy diagnostics, not absolute reconstruction errors.
A high association with spike, saturation, or early-peak proxies indicates that the score identifies internally unstable or glare-dominated DFF regions in the real stack.
