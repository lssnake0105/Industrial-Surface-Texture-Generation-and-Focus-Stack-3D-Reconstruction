# Glare-Risk Microfacet Demo Summary

This simulation is a minimal optical explanation aid, not a calibrated renderer.
It converts synthetic micro-height maps into normals and estimates whether specular reflection enters the objective acceptance cone.

| Surface | NA | Mean risk | High-risk fraction >=0.75 | P95 risk |
|---|---:|---:|---:|---:|
| v_valley | 0.20 | 0.0151 | 0.0044 | 0.0247 |
| v_valley | 0.40 | 0.0628 | 0.0169 | 0.6500 |
| v_valley | 0.65 | 0.1852 | 0.0480 | 0.7409 |
| circular_pit | 0.20 | 0.0298 | 0.0080 | 0.1657 |
| circular_pit | 0.40 | 0.1159 | 0.0333 | 0.6503 |
| circular_pit | 0.65 | 0.2916 | 0.0955 | 0.8645 |
| rough_ridge | 0.20 | 0.0028 | 0.0007 | 0.0000 |
| rough_ridge | 0.40 | 0.0121 | 0.0032 | 0.0128 |
| rough_ridge | 0.65 | 0.0389 | 0.0091 | 0.2292 |
| key_like_edge | 0.20 | 0.0009 | 0.0002 | 0.0000 |
| key_like_edge | 0.40 | 0.0040 | 0.0010 | 0.0000 |
| key_like_edge | 0.65 | 0.0136 | 0.0030 | 0.0283 |

## Interpretation

Increasing NA broadens the acceptance cone and increases the fraction of microfacets whose specular reflection can enter the imaging path.
Edges, pits, and rough ridges create spatially localized high-risk patterns, supporting the use of a glare-risk prior rather than a global brightness correction.
