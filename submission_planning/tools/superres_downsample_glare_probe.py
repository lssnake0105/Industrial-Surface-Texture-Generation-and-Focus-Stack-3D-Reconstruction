from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def normalize01(a: np.ndarray) -> np.ndarray:
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def block_average(a: np.ndarray, factor: int) -> np.ndarray:
    h, w = a.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    a = a[:h2, :w2]
    return a.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def block_max(a: np.ndarray, factor: int) -> np.ndarray:
    h, w = a.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    a = a[:h2, :w2]
    return a.reshape(h2 // factor, factor, w2 // factor, factor).max(axis=(1, 3))


def box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return a
    out = a.copy()
    for _ in range(radius):
        out = (
            out
            + np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
        ) / 5.0
    return out


def make_micro_surface(size: int, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]

    edge = 0.50 + 0.35 / (1.0 + np.exp(-(x - 0.48) * 26.0))
    groove = -0.22 * np.exp(-((x + 0.18) / 0.10) ** 2)
    scratch = 0.08 * np.sin(80 * y + 14 * np.sin(10 * x))
    micro = rng.normal(0.0, 1.0, size=(size, size))
    for _ in range(4):
        micro = box_blur(micro, 1)
    micro = normalize01(micro) * 2.0 - 1.0
    height = edge + groove + scratch + 0.10 * micro
    return normalize01(height)


def normals(height: np.ndarray, lateral_scale_um: float, height_scale_um: float) -> np.ndarray:
    dzdy, dzdx = np.gradient(
        height * height_scale_um,
        lateral_scale_um / height.shape[0],
        lateral_scale_um / height.shape[1],
    )
    n = np.dstack([-dzdx, -dzdy, np.ones_like(height)])
    n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-12)
    return n


def specular_risk(n: np.ndarray, na: float = 0.4, tilt_deg: float = 0.0) -> np.ndarray:
    tilt = np.deg2rad(tilt_deg)
    light = np.array([np.sin(tilt), 0.0, -np.cos(tilt)], dtype=np.float64)
    light = light / np.linalg.norm(light)
    view = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = np.sum(n * light[None, None, :], axis=2, keepdims=True)
    reflected = light[None, None, :] - 2.0 * dot * n
    reflected /= np.maximum(np.linalg.norm(reflected, axis=2, keepdims=True), 1e-12)
    cosang = np.clip(np.sum(reflected * view[None, None, :], axis=2), -1.0, 1.0)
    theta = np.arccos(cosang)
    acceptance = np.arcsin(np.clip(na, 0.0, 0.99))
    hard = (theta <= acceptance).astype(float)
    soft = np.exp(-(theta / max(acceptance, 1e-6)) ** 2)
    return np.maximum(0.75 * hard, soft)


def synthesize_focus_stack(
    height: np.ndarray,
    risk: np.ndarray,
    layers: int = 25,
    blur_base: float = 0.4,
    blur_gain: float = 6.0,
    exposure: float = 1.25,
) -> list[np.ndarray]:
    focus_planes = np.linspace(0.0, 1.0, layers)
    texture = 0.35 + 0.25 * normalize01(np.sin(45 * height) + 0.4 * np.cos(70 * height))
    images = []
    for z in focus_planes:
        defocus = np.abs(height - z)
        radius = int(round(blur_base + blur_gain * float(np.mean(defocus))))
        local_blur = box_blur(texture + 1.10 * risk, max(0, radius))
        # Nearby focus keeps texture contrast; far defocus keeps broad highlight halos.
        sharp_weight = np.exp(-(defocus / 0.12) ** 2)
        image = 0.20 + 0.55 * sharp_weight * texture + 0.85 * local_blur * (0.35 + 0.65 * risk)
        image = np.clip(image * exposure, 0.0, 1.0)
        images.append(image.astype(np.float32))
    return images


def laplacian_energy(arr: np.ndarray) -> float:
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0
    c = arr[1:-1, 1:-1]
    lap = arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:] - 4 * c
    return float(np.mean(lap * lap))


def tenengrad(arr: np.ndarray) -> float:
    gy, gx = np.gradient(arr)
    return float(np.mean(gx * gx + gy * gy))


def compare_direct_and_superres(factor: int, low_size: int, layers: int, out: Path) -> list[dict[str, float | str]]:
    high_size = low_size * factor
    lateral_um = 300.0
    height_um = 80.0

    height_hr = make_micro_surface(high_size)
    normal_hr = normals(height_hr, lateral_scale_um=lateral_um, height_scale_um=height_um)
    risk_hr = specular_risk(normal_hr, na=0.40)
    stack_hr = synthesize_focus_stack(height_hr, risk_hr, layers=layers)
    stack_sr = [block_average(im, factor) for im in stack_hr]
    risk_sr = block_average(risk_hr, factor)
    height_sr = block_average(height_hr, factor)

    height_lr = block_average(height_hr, factor)
    normal_lr = normals(height_lr, lateral_scale_um=lateral_um, height_scale_um=height_um)
    risk_lr = specular_risk(normal_lr, na=0.40)
    stack_lr = synthesize_focus_stack(height_lr, risk_lr, layers=layers)

    rows: list[dict[str, float | str]] = []
    for mode, stack, risk in [
        ("superres_integrated", stack_sr, risk_sr),
        ("direct_lowres", stack_lr, risk_lr),
    ]:
        for i, im in enumerate(stack, start=1):
            rows.append(
                {
                    "mode": mode,
                    "factor": factor,
                    "layer": i,
                    "mean": float(np.mean(im)),
                    "p99": float(np.quantile(im, 0.99)),
                    "sat_ratio_098": float(np.mean(im >= 0.98)),
                    "bright_ratio_090": float(np.mean(im >= 0.90)),
                    "laplacian_energy": laplacian_energy(im),
                    "tenengrad": tenengrad(im),
                    "risk_mean": float(np.mean(risk)),
                    "risk_high_fraction_075": float(np.mean(risk >= 0.75)),
                }
            )

    # Figures.
    diff_risk = risk_sr - risk_lr
    selected = [0, layers // 4, layers // 2, (3 * layers) // 4, layers - 1]
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 9.0))
    axes[0, 0].imshow(height_sr, cmap="viridis")
    axes[0, 0].set_title("height, HR integrated")
    axes[0, 1].imshow(risk_sr, cmap="inferno", vmin=0, vmax=1)
    axes[0, 1].set_title("risk, HR integrated")
    axes[0, 2].imshow(risk_lr, cmap="inferno", vmin=0, vmax=1)
    axes[0, 2].set_title("risk, direct low-res")
    axes[0, 3].imshow(diff_risk, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0, 3].set_title("risk difference")

    for col, idx in enumerate(selected[:4]):
        axes[1, col].imshow(stack_sr[idx], cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"SR layer {idx+1}")
        axes[2, col].imshow(stack_lr[idx], cmap="gray", vmin=0, vmax=1)
        axes[2, col].set_title(f"LR layer {idx+1}")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Super-resolution integration vs direct low-resolution simulation (factor={factor})")
    fig.tight_layout()
    fig.savefig(out / f"sr_factor_{factor}_comparison_panel.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(8.0, 9.0), sharex=True)
    for mode, color in [("superres_integrated", "#E41A1C"), ("direct_lowres", "#377EB8")]:
        sub = [r for r in rows if r["mode"] == mode]
        x = np.array([int(r["layer"]) for r in sub])
        p99 = np.array([float(r["p99"]) for r in sub])
        sat = np.array([float(r["sat_ratio_098"]) for r in sub])
        lap = np.array([float(r["laplacian_energy"]) for r in sub])
        ten = np.array([float(r["tenengrad"]) for r in sub])
        axes[0].plot(x, [float(r["mean"]) for r in sub], color=color, label=mode)
        axes[1].plot(x, p99, color=color, label=mode)
        axes[2].plot(x, sat * 100.0, color=color, label=mode)
        axes[3].plot(x, lap / max(float(lap.max()), 1e-12), color=color, linestyle="-", label=f"{mode} Lap.")
        axes[3].plot(x, ten / max(float(ten.max()), 1e-12), color=color, linestyle="--", label=f"{mode} Ten.")
    axes[0].set_ylabel("mean")
    axes[1].set_ylabel("p99")
    axes[2].set_ylabel("I>=0.98 (%)")
    axes[3].set_ylabel("normalized focus")
    axes[3].set_xlabel("focal layer")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(f"Resolution-sensitive focus response (factor={factor})")
    fig.tight_layout()
    fig.savefig(out / f"sr_factor_{factor}_curves.png", dpi=180)
    plt.close(fig)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submission_planning/optical_mechanism_analysis/superres_sampling_branch")
    parser.add_argument("--low-size", type=int, default=160)
    parser.add_argument("--layers", type=int, default=25)
    parser.add_argument("--factors", nargs="*", type=int, default=[2, 4, 8])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, float | str]] = []
    for factor in args.factors:
        all_rows.extend(compare_direct_and_superres(factor, args.low_size, args.layers, out))

    metrics_path = out / "superres_downsample_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_lines = [
        "# Super-Resolution Downsample Simulation Summary",
        "",
        "This branch compares two simulation paths:",
        "",
        "- `direct_lowres`: compute normals, glare risk, and focal stack directly on the sensor grid.",
        "- `superres_integrated`: compute micro-geometry and glare at a higher resolution, then block-average down to sensor pixels.",
        "",
        "| Factor | Mode | Risk mean | Risk high >=0.75 | Max p99 | Max sat I>=0.98 | Best Lap. layer | Best Ten. layer |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for factor in args.factors:
        for mode in ["superres_integrated", "direct_lowres"]:
            sub = [r for r in all_rows if int(r["factor"]) == factor and r["mode"] == mode]
            p99 = np.array([float(r["p99"]) for r in sub])
            sat = np.array([float(r["sat_ratio_098"]) for r in sub])
            lap = np.array([float(r["laplacian_energy"]) for r in sub])
            ten = np.array([float(r["tenengrad"]) for r in sub])
            summary_lines.append(
                "| {factor} | {mode} | {risk_mean:.4f} | {risk_high:.4f} | {p99:.4f} | {sat:.4f} | {lap_layer} | {ten_layer} |".format(
                    factor=factor,
                    mode=mode,
                    risk_mean=float(sub[0]["risk_mean"]),
                    risk_high=float(sub[0]["risk_high_fraction_075"]),
                    p99=float(p99.max()),
                    sat=float(sat.max()),
                    lap_layer=int(lap.argmax() + 1),
                    ten_layer=int(ten.argmax() + 1),
                )
            )
    summary_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The super-resolution-integrated path preserves subpixel microfacet contributions before sensor integration. Direct low-resolution simulation can under- or over-estimate glare risk because normals are computed after geometric averaging.",
            "This supports using high-resolution microgeometry followed by physically meaningful downsampling when generating synthetic focal stacks for reflective surfaces.",
        ]
    )
    (out / "superres_downsample_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(out)
    print(metrics_path)


if __name__ == "__main__":
    main()
