# Super-Resolution Downsample Simulation Summary

This branch compares two simulation paths:

- `direct_lowres`: compute normals, glare risk, and focal stack directly on the sensor grid.
- `superres_integrated`: compute micro-geometry and glare at a higher resolution, then block-average down to sensor pixels.

| Factor | Mode | Risk mean | Risk high >=0.75 | Max p99 | Max sat I>=0.98 | Best Lap. layer | Best Ten. layer |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | superres_integrated | 0.0035 | 0.0000 | 0.8317 | 0.0000 | 8 | 8 |
| 2 | direct_lowres | 0.0063 | 0.0054 | 0.8612 | 0.0031 | 12 | 8 |
| 4 | superres_integrated | 0.0020 | 0.0000 | 0.8162 | 0.0000 | 8 | 8 |
| 4 | direct_lowres | 0.0072 | 0.0065 | 0.8696 | 0.0039 | 12 | 13 |
| 8 | superres_integrated | 0.0011 | 0.0000 | 0.7968 | 0.0000 | 9 | 9 |
| 8 | direct_lowres | 0.0113 | 0.0100 | 0.8884 | 0.0063 | 12 | 9 |

## Interpretation

The super-resolution-integrated path preserves subpixel microfacet contributions before sensor integration. Direct low-resolution simulation can under- or over-estimate glare risk because normals are computed after geometric averaging.
This supports using high-resolution microgeometry followed by physically meaningful downsampling when generating synthetic focal stacks for reflective surfaces.
