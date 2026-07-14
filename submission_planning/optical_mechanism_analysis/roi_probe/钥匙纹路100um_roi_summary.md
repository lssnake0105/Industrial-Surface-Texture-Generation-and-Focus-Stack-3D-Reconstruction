# ROI Probe Summary: 钥匙纹路100um

| ROI | Box | max p99 | max sat I>=0.98 | best Laplacian layer | best Tenengrad layer |
|---|---|---:|---:|---:|---:|
| highlight_edge | (np.int64(587), np.int64(275), 640, np.int64(403)) | 1.0000 | 0.2328 | 2 | 2 |
| ordinary_texture | (218, 164, 333, 287) | 0.2627 | 0.0000 | 18 | 18 |
| dark_region | (358, 174, 486, 317) | 0.2000 | 0.0000 | 19 | 19 |

## Interpretation

The `highlight_edge` ROI is selected around the brightest persistent area in the focal stack. It should be interpreted as a diagnostic candidate rather than a manually verified defect annotation.
A close match between high p99/saturation layers and focus-measure peak layers supports the hypothesis that glare edges can drive DFF focus selection.
