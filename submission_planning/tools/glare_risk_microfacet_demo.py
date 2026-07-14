from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def normalize01(a: np.ndarray) -> np.ndarray:
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def smooth_noise(rng: np.random.Generator, shape: tuple[int, int], passes: int = 5) -> np.ndarray:
    a = rng.normal(size=shape)
    for _ in range(passes):
        a = (
            a
            + np.roll(a, 1, axis=0)
            + np.roll(a, -1, axis=0)
            + np.roll(a, 1, axis=1)
            + np.roll(a, -1, axis=1)
        ) / 5.0
    return normalize01(a) * 2.0 - 1.0


def make_surfaces(size: int = 256, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    r = np.sqrt(x * x + y * y)

    v_valley = 1.0 - np.abs(x) * 0.85
    v_valley += 0.08 * smooth_noise(rng, (size, size), passes=8)

    circular_pit = 1.0 - 0.85 * np.exp(-(r / 0.34) ** 2)
    circular_pit += 0.06 * smooth_noise(rng, (size, size), passes=7)

    ridge = 0.45 * np.exp(-((x + 0.15) / 0.13) ** 2) + 0.35 * np.exp(-((y - 0.25) / 0.18) ** 2)
    ridge += 0.18 * smooth_noise(rng, (size, size), passes=4)

    key_like_edge = 0.35 + 0.65 / (1.0 + np.exp(-(x - 0.68) * 24.0))
    key_like_edge += 0.10 * np.sin(30 * y + 8 * smooth_noise(rng, (size, size), passes=6))
    key_like_edge += 0.12 * smooth_noise(rng, (size, size), passes=5)

    return {
        "v_valley": normalize01(v_valley),
        "circular_pit": normalize01(circular_pit),
        "rough_ridge": normalize01(ridge),
        "key_like_edge": normalize01(key_like_edge),
    }


def normals(height: np.ndarray, lateral_scale_um: float = 300.0, height_scale_um: float = 80.0) -> np.ndarray:
    dzdy, dzdx = np.gradient(height * height_scale_um, lateral_scale_um / height.shape[0], lateral_scale_um / height.shape[1])
    n = np.dstack([-dzdx, -dzdy, np.ones_like(height)])
    n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-12)
    return n


def glare_risk(n: np.ndarray, na: float, incident_tilt_deg: float = 0.0) -> np.ndarray:
    tilt = np.deg2rad(incident_tilt_deg)
    light = np.array([np.sin(tilt), 0.0, -np.cos(tilt)], dtype=np.float64)
    view = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    light = light / np.linalg.norm(light)
    dot = np.sum(n * light[None, None, :], axis=2, keepdims=True)
    reflected = light[None, None, :] - 2.0 * dot * n
    reflected /= np.maximum(np.linalg.norm(reflected, axis=2, keepdims=True), 1e-12)
    cosang = np.clip(np.sum(reflected * view[None, None, :], axis=2), -1.0, 1.0)
    theta = np.arccos(cosang)
    acceptance = np.arcsin(np.clip(na, 0.0, 0.99))
    soft = np.exp(-(theta / max(acceptance, 1e-6)) ** 2)
    hard = (theta <= acceptance).astype(float)
    return np.maximum(0.65 * hard, soft)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submission_planning/optical_mechanism_analysis/glare_sim")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    surfaces = make_surfaces()
    nas = [0.20, 0.40, 0.65]
    rows = []

    for name, height in surfaces.items():
        n = normals(height)
        slope = np.sqrt(n[:, :, 0] ** 2 + n[:, :, 1] ** 2)

        fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.4))
        axes[0, 0].imshow(height, cmap="viridis")
        axes[0, 0].set_title("height")
        axes[0, 1].imshow(slope, cmap="magma")
        axes[0, 1].set_title("normal tilt proxy")
        axes[0, 2].axis("off")
        axes[0, 2].text(
            0.05,
            0.85,
            "Specular risk model\n"
            "r = l - 2(n·l)n\n"
            "risk if angle(r,v) <= asin(NA)",
            fontsize=10,
            va="top",
        )
        for col, na in enumerate(nas):
            risk = glare_risk(n, na=na, incident_tilt_deg=0.0)
            axes[1, col].imshow(risk, cmap="inferno", vmin=0, vmax=1)
            axes[1, col].set_title(f"glare risk NA={na:.2f}")
            rows.append(
                {
                    "surface": name,
                    "na": na,
                    "mean_risk": float(np.mean(risk)),
                    "high_risk_fraction_075": float(np.mean(risk >= 0.75)),
                    "p95_risk": float(np.quantile(risk, 0.95)),
                }
            )
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Micro-surface glare-risk demo: {name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / f"{name}_glare_risk_demo.png", dpi=180)
        plt.close(fig)

    metrics_path = out / "glare_risk_demo_metrics.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("surface,na,mean_risk,high_risk_fraction_075,p95_risk\n")
        for row in rows:
            f.write(
                f"{row['surface']},{row['na']},{row['mean_risk']:.6f},{row['high_risk_fraction_075']:.6f},{row['p95_risk']:.6f}\n"
            )

    summary = [
        "# Glare-Risk Microfacet Demo Summary",
        "",
        "This simulation is a minimal optical explanation aid, not a calibrated renderer.",
        "It converts synthetic micro-height maps into normals and estimates whether specular reflection enters the objective acceptance cone.",
        "",
        "| Surface | NA | Mean risk | High-risk fraction >=0.75 | P95 risk |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        summary.append(
            f"| {row['surface']} | {row['na']:.2f} | {row['mean_risk']:.4f} | {row['high_risk_fraction_075']:.4f} | {row['p95_risk']:.4f} |"
        )
    summary.extend(
        [
            "",
            "## Interpretation",
            "",
            "Increasing NA broadens the acceptance cone and increases the fraction of microfacets whose specular reflection can enter the imaging path.",
            "Edges, pits, and rough ridges create spatially localized high-risk patterns, supporting the use of a glare-risk prior rather than a global brightness correction.",
        ]
    )
    (out / "glare_risk_demo_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(out)
    print(metrics_path)


if __name__ == "__main__":
    main()
