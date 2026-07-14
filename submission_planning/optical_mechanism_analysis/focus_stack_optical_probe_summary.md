# Focus-Stack Optical Probe Summary

This automatically generated summary reports lightweight diagnostics for selected real focal stacks.
Intensity values are normalized to [0, 1]. Saturation is approximated as pixels with intensity >= 0.98.

| Stack | Layers | Size | Max sat. ratio | Bright any | Bright half+ | Best focus layer | Median shift px | Max shift px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3D表面 | 50 | 640x512 | 0.0000 | 0.0000 | 0.0000 | 15 | 0.00 | 0.00 |
| 3D层纹 | 36 | 640x512 | 0.0000 | 0.0000 | 0.0000 | 17 | 0.00 | 0.00 |
| 磕碰孔5um | 21 | 640x512 | 0.0000 | 0.0003 | 0.0000 | 9 | 0.00 | 2.00 |
| 钥匙尖头50um | 28 | 640x512 | 0.0000 | 0.0024 | 0.0000 | 13 | 0.00 | 0.00 |
| 钥匙纹路100um | 40 | 640x512 | 0.0126 | 0.0520 | 0.0016 | 2 | 0.00 | 0.00 |
| 圆孔50um | 20 | 640x512 | 0.0000 | 0.0000 | 0.0000 | 7 | 0.00 | 0.00 |

## Interpretation Notes

- `Max sat. ratio` estimates whether a stack contains near-clipped highlights.
- `Bright any` and `Bright half+` summarize whether bright/glare-prone pixels persist across focal layers.
- `Median shift px` and `Max shift px` are phase-correlation diagnostics against the first layer; large values should be checked visually because specular changes can also bias this estimate.
