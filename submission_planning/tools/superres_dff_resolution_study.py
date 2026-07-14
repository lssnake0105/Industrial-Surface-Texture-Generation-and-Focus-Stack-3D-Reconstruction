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
    return a[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def block_max(a: np.ndarray, factor: int) -> np.ndarray:
    h, w = a.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    return a[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).max(axis=(1, 3))


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


def make_micro_surface(size: int, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    v_valley = 0.58 + 0.22 * np.abs(x + 0.18)
    key_edge = 0.30 / (1.0 + np.exp(-(x - 0.52) * 26.0))
    circular_pit = -0.22 * np.exp(-(((x + 0.34) / 0.25) ** 2 + ((y - 0.10) / 0.34) ** 2))
    periodic_texture = 0.055 * np.sin(70 * y + 12 * np.sin(8 * x))
    noise = rng.normal(size=(size, size))
    for _ in range(5):
        noise = box_blur(noise, 1)
    height = v_valley + key_edge + circular_pit + periodic_texture + 0.07 * normalize01(noise)
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


def specular_risk(n: np.ndarray, na: float = 0.4) -> np.ndarray:
    light = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    view = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = np.sum(n * light[None, None, :], axis=2, keepdims=True)
    reflected = light[None, None, :] - 2.0 * dot * n
    reflected /= np.maximum(np.linalg.norm(reflected, axis=2, keepdims=True), 1e-12)
    theta = np.arccos(np.clip(np.sum(reflected * view[None, None, :], axis=2), -1.0, 1.0))
    acceptance = np.arcsin(np.clip(na, 0.0, 0.99))
    hard = (theta <= acceptance).astype(float)
    soft = np.exp(-(theta / max(acceptance, 1e-6)) ** 2)
    return np.maximum(0.75 * hard, soft)


def synthesize_stack(
    height: np.ndarray,
    risk: np.ndarray,
    layers: int,
    exposure: float = 1.42,
) -> tuple[np.ndarray, np.ndarray]:
    focus_planes = np.linspace(0.0, 1.0, layers)
    texture = 0.45 + 0.23 * normalize01(np.sin(54 * height) + 0.4 * np.cos(85 * height))
    stack = []
    for z in focus_planes:
        defocus = np.abs(height - z)
        sharp_weight = np.exp(-(defocus / 0.105) ** 2)
        halo_radius = max(1, int(round(1.0 + 6.0 * float(np.mean(defocus)))))
        halo = box_blur(0.55 * texture + 1.25 * risk, halo_radius)
        image = 0.18 + 0.58 * sharp_weight * texture + (0.18 + 0.82 * risk) * halo
        image = np.clip(image * exposure, 0.0, 1.0)
        stack.append(image.astype(np.float32))
    return np.stack(stack, axis=0), focus_planes


def focus_volume(stack: np.ndarray) -> np.ndarray:
    vols = []
    for im in stack:
        c = im[1:-1, 1:-1]
        lap = np.abs(im[:-2, 1:-1] + im[2:, 1:-1] + im[1:-1, :-2] + im[1:-1, 2:] - 4 * c)
        pad = np.pad(lap, 1, mode="edge")
        vols.append(box_blur(pad, 2))
    return np.stack(vols, axis=0)


def dff_depth(stack: np.ndarray, focus_planes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fv = focus_volume(stack)
    idx = np.argmax(fv, axis=0)
    pred = focus_planes[idx]
    sorted_fv = np.sort(fv, axis=0)
    confidence = (sorted_fv[-1] - sorted_fv[-2]) / np.maximum(sorted_fv[-1], 1e-8)
    return pred.astype(np.float32), idx.astype(np.int16), confidence.astype(np.float32)


def edge_mask(height: np.ndarray, q: float = 0.90) -> np.ndarray:
    gy, gx = np.gradient(height)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag >= np.quantile(mag, q)


def evaluate(pred: np.ndarray, gt: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    err = np.abs(pred - gt)
    edge = edge_mask(gt)
    high = risk >= np.quantile(risk, 0.90)
    return {
        "mae_norm": float(np.mean(err)),
        "rmse_norm": float(np.sqrt(np.mean(err * err))),
        "p90_norm": float(np.quantile(err, 0.90)),
        "edge_mae_norm": float(np.mean(err[edge])) if np.any(edge) else float("nan"),
        "high_risk_mae_norm": float(np.mean(err[high])) if np.any(high) else float("nan"),
        "max_error_norm": float(np.max(err)),
        "mean_confidence": float(np.mean(np.isfinite(pred))) if pred.size else 0.0,
    }


def run_case(factor: int, low_size: int, layers: int, out: Path) -> list[dict[str, float | str]]:
    high_size = low_size * factor
    lateral_um = 300.0
    height_um = 80.0
    height_hr = make_micro_surface(high_size)
    normal_hr = normals(height_hr, lateral_um, height_um)
    risk_hr = specular_risk(normal_hr, na=0.40)
    stack_hr, planes = synthesize_stack(height_hr, risk_hr, layers)

    height_sr = block_average(height_hr, factor)
    risk_mean = block_average(risk_hr, factor)
    risk_max = block_max(risk_hr, factor)
    stack_sr = np.stack([block_average(im, factor) for im in stack_hr], axis=0)

    height_lr = height_sr.copy()
    normal_lr = normals(height_lr, lateral_um, height_um)
    risk_lr = specular_risk(normal_lr, na=0.40)
    stack_lr, _ = synthesize_stack(height_lr, risk_lr, layers)

    rows: list[dict[str, float | str]] = []
    outputs = {}
    for mode, stack, risk_for_eval in [
        ("superres_integrated", stack_sr, risk_mean),
        ("direct_lowres", stack_lr, risk_lr),
    ]:
        pred, peak_idx, conf = dff_depth(stack, planes)
        metrics = evaluate(pred, height_sr, risk_for_eval)
        metrics.update(
            {
                "factor": factor,
                "mode": mode,
                "risk_mean": float(np.mean(risk_for_eval)),
                "risk_high_fraction_090": float(np.mean(risk_for_eval >= np.quantile(risk_for_eval, 0.90))),
                "peak_layer_mean": float(np.mean(peak_idx + 1)),
                "peak_layer_std": float(np.std(peak_idx + 1)),
            }
        )
        rows.append(metrics)
        outputs[mode] = (pred, peak_idx, conf, stack, risk_for_eval)

    err_sr = np.abs(outputs["superres_integrated"][0] - height_sr)
    err_lr = np.abs(outputs["direct_lowres"][0] - height_sr)
    vmax = max(float(np.quantile(err_sr, 0.99)), float(np.quantile(err_lr, 0.99)), 1e-6)

    fig, axes = plt.subplots(3, 4, figsize=(12.5, 9.0))
    axes[0, 0].imshow(height_sr, cmap="viridis", vmin=0, vmax=1)
    axes[0, 0].set_title("GT height")
    axes[0, 1].imshow(risk_mean, cmap="inferno", vmin=0, vmax=1)
    axes[0, 1].set_title("SR risk mean")
    axes[0, 2].imshow(risk_max, cmap="inferno", vmin=0, vmax=1)
    axes[0, 2].set_title("SR risk max")
    axes[0, 3].imshow(risk_lr, cmap="inferno", vmin=0, vmax=1)
    axes[0, 3].set_title("LR direct risk")

    for row, mode in enumerate(["superres_integrated", "direct_lowres"], start=1):
        pred, peak_idx, _conf, _stack, _risk = outputs[mode]
        err = np.abs(pred - height_sr)
        axes[row, 0].imshow(pred, cmap="viridis", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{mode} DFF depth")
        axes[row, 1].imshow(err, cmap="magma", vmin=0, vmax=vmax)
        axes[row, 1].set_title(f"{mode} abs error")
        axes[row, 2].imshow(peak_idx + 1, cmap="turbo", vmin=1, vmax=layers)
        axes[row, 2].set_title(f"{mode} peak layer")
        axes[row, 3].imshow(outputs[mode][3][layers // 2], cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title(f"{mode} mid-stack image")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"DFF resolution sensitivity: factor={factor}")
    fig.tight_layout()
    fig.savefig(out / f"dff_factor_{factor}_depth_error_panel.png", dpi=180)
    plt.close(fig)

    # Error profiles by x to reveal edge-driven bias.
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True)
    xs = np.arange(height_sr.shape[1])
    axes[0].plot(xs, err_sr.mean(axis=0), label="superres_integrated", color="#E41A1C")
    axes[0].plot(xs, err_lr.mean(axis=0), label="direct_lowres", color="#377EB8")
    axes[0].set_ylabel("column mean abs error")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(xs, risk_mean.mean(axis=0), label="SR risk mean", color="#E41A1C")
    axes[1].plot(xs, risk_lr.mean(axis=0), label="LR risk", color="#377EB8")
    axes[1].set_ylabel("column mean risk")
    axes[1].set_xlabel("x pixel")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"Error/risk profiles: factor={factor}")
    fig.tight_layout()
    fig.savefig(out / f"dff_factor_{factor}_error_risk_profile.png", dpi=180)
    plt.close(fig)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submission_planning/optical_mechanism_analysis/dff_resolution_study")
    parser.add_argument("--low-size", type=int, default=160)
    parser.add_argument("--layers", type=int, default=25)
    parser.add_argument("--factors", nargs="*", type=int, default=[2, 4, 8])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    for factor in args.factors:
        rows.extend(run_case(factor, args.low_size, args.layers, out))

    metrics_path = out / "dff_resolution_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# DFF Resolution Sensitivity Summary",
        "",
        "| Factor | Mode | MAE | RMSE | P90 | Edge MAE | High-risk MAE | Peak layer mean | Risk mean |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {factor} | {mode} | {mae_norm:.4f} | {rmse_norm:.4f} | {p90_norm:.4f} | {edge_mae_norm:.4f} | {high_risk_mae_norm:.4f} | {peak_layer_mean:.2f} | {risk_mean:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This study measures how the simulation sampling strategy propagates into DFF depth selection.",
            "The target comparison is not absolute optical realism; it isolates whether computing normals and glare at sensor resolution changes depth errors relative to a super-resolution integrated pipeline.",
        ]
    )
    (out / "dff_resolution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out)
    print(metrics_path)


if __name__ == "__main__":
    main()
