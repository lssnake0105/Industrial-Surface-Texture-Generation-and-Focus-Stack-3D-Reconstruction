from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift

from focus_confidence_risk_study import auc_score, confidence_maps, focus_volume, normalize01, top_fraction_mask


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def numeric_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def image_files(stack_dir: Path) -> list[Path]:
    return sorted(
        [p for p in stack_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=numeric_key,
    )


def safe_name(name: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff-]+", "_", name)


def display_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return cleaned or "real_stack"


def read_gray(path: Path, max_side: int = 640) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        scale = min(1.0, max_side / max(im.size))
        if scale < 1.0:
            new_size = (max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale)))
            im = im.resize(new_size, Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def center_crop_for_registration(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    # Exclude saturated side edges and use the central sample region.
    y0 = int(round(0.16 * h))
    y1 = int(round(0.86 * h))
    x0 = int(round(0.12 * w))
    x1 = int(round(0.82 * w))
    crop = arr[y0:y1, x0:x1]
    lo, hi = np.quantile(crop, [0.02, 0.98])
    crop = np.clip((crop - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return crop.astype(np.float32)


def estimate_shifts(stack: np.ndarray, reference_idx: int) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    ref = center_crop_for_registration(stack[reference_idx])
    shifts = []
    rows: list[dict[str, float | int]] = []
    prev_shift = np.array([0.0, 0.0], dtype=np.float64)
    for i, im in enumerate(stack):
        moving = center_crop_for_registration(im)
        shift_yx, error, phasediff = phase_cross_correlation(ref, moving, upsample_factor=20)
        # Guard against occasional phase-correlation outliers in defocused layers.
        shift_yx = np.asarray(shift_yx, dtype=np.float64)
        if i > 0 and np.linalg.norm(shift_yx - prev_shift) > 25:
            shift_yx = prev_shift.copy()
        prev_shift = shift_yx
        shifts.append(shift_yx)
        rows.append(
            {
                "layer": i + 1,
                "shift_y_px": float(shift_yx[0]),
                "shift_x_px": float(shift_yx[1]),
                "shift_mag_px": float(np.sqrt(np.sum(shift_yx * shift_yx))),
                "registration_error": float(error),
                "phasediff": float(phasediff),
            }
        )
    return np.vstack(shifts), rows


def apply_shifts(stack: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    aligned = []
    for im, shift_yx in zip(stack, shifts):
        aligned.append(ndi_shift(im, shift=shift_yx, order=1, mode="nearest", prefilter=False))
    return np.stack(aligned, axis=0)


def local_mean(arr: np.ndarray, radius: int = 2) -> np.ndarray:
    pad = np.pad(arr, radius, mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out += pad[radius + dy : radius + dy + arr.shape[0], radius + dx : radius + dx + arr.shape[1]]
            count += 1
    return out / float(count)


def compute_diagnostics(stack: np.ndarray) -> dict[str, np.ndarray | float]:
    fv = focus_volume(stack)
    conf = confidence_maps(fv)
    peak_idx = np.argmax(fv, axis=0).astype(np.float32)
    low_margin = 1.0 - conf["confidence_margin"]
    sat_persistence = np.mean(stack >= 0.98, axis=0)
    bright_persistence = np.mean(stack >= 0.90, axis=0)
    spike_proxy = normalize01(np.abs(peak_idx - local_mean(peak_idx, radius=2)))
    quality_proxy = normalize01(0.68 * normalize01(low_margin) + 0.22 * normalize01(spike_proxy) + 0.10 * normalize01(sat_persistence))
    return {
        "peak_layer": peak_idx + 1,
        "low_margin": low_margin,
        "focus_entropy": conf["focus_entropy"],
        "low_peak_strength": 1.0 - conf["confidence_peak_strength"],
        "sat_persistence": sat_persistence,
        "bright_persistence": bright_persistence,
        "spike_proxy": spike_proxy,
        "quality_proxy": quality_proxy,
        "spike_mean": float(np.mean(spike_proxy)),
        "spike_p90": float(np.quantile(spike_proxy, 0.90)),
        "quality_mean": float(np.mean(quality_proxy)),
        "quality_p90": float(np.quantile(quality_proxy, 0.90)),
        "sat_mean": float(np.mean(sat_persistence)),
        "bright_mean": float(np.mean(bright_persistence)),
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = a.ravel().astype(np.float64)
    y = b.ravel().astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def compare_diagnostics(before: dict[str, np.ndarray | float], after: dict[str, np.ndarray | float]) -> dict[str, float]:
    before_peak = before["peak_layer"]
    after_peak = after["peak_layer"]
    assert isinstance(before_peak, np.ndarray)
    assert isinstance(after_peak, np.ndarray)
    peak_diff = np.abs(after_peak - before_peak)
    row = {
        "peak_layer_changed_fraction": float(np.mean(peak_diff > 0.5)),
        "peak_layer_mean_abs_change": float(np.mean(peak_diff)),
        "peak_layer_p90_abs_change": float(np.quantile(peak_diff, 0.90)),
    }
    for key in ["spike_proxy", "quality_proxy", "low_margin", "focus_entropy", "sat_persistence"]:
        a = before[key]
        b = after[key]
        assert isinstance(a, np.ndarray)
        assert isinstance(b, np.ndarray)
        row[f"{key}_mean_before"] = float(np.mean(a))
        row[f"{key}_mean_after"] = float(np.mean(b))
        row[f"{key}_mean_delta"] = float(np.mean(b) - np.mean(a))
        row[f"{key}_pearson_before_after"] = pearson(a, b)
    labels_before = top_fraction_mask(before["spike_proxy"], 0.10)  # type: ignore[arg-type]
    labels_after = top_fraction_mask(after["spike_proxy"], 0.10)  # type: ignore[arg-type]
    for key in ["low_margin", "focus_entropy", "low_peak_strength", "quality_proxy", "sat_persistence"]:
        score_before = before[key]
        score_after = after[key]
        assert isinstance(score_before, np.ndarray)
        assert isinstance(score_after, np.ndarray)
        row[f"{key}_auc_spike_top10_before"] = auc_score(score_before, labels_before)
        row[f"{key}_auc_spike_top10_after"] = auc_score(score_after, labels_after)
    return row


def draw_shift_plot(stack_name: str, rows: list[dict[str, float | int]], out: Path) -> None:
    layer = np.array([float(r["layer"]) for r in rows])
    sx = np.array([float(r["shift_x_px"]) for r in rows])
    sy = np.array([float(r["shift_y_px"]) for r in rows])
    mag = np.array([float(r["shift_mag_px"]) for r in rows])
    err = np.array([float(r["registration_error"]) for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.0), sharex=True)
    axes[0].plot(layer, sx, label="x shift", color="#377EB8")
    axes[0].plot(layer, sy, label="y shift", color="#E41A1C")
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_ylabel("shift to ref (px)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(layer, mag, color="#4DAF4A")
    axes[1].set_ylabel("shift magnitude (px)")
    axes[1].grid(alpha=0.25)
    axes[2].plot(layer, err, color="#984EA3")
    axes[2].set_ylabel("phase corr. error")
    axes[2].set_xlabel("layer")
    axes[2].grid(alpha=0.25)
    fig.suptitle(f"{display_name(stack_name)}: inter-layer shift estimate")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def draw_comparison_maps(
    stack_name: str,
    before: dict[str, np.ndarray | float],
    after: dict[str, np.ndarray | float],
    out: Path,
) -> None:
    panels = [
        ("peak before", before["peak_layer"], "turbo"),
        ("peak after", after["peak_layer"], "turbo"),
        ("abs peak change", np.abs(after["peak_layer"] - before["peak_layer"]), "magma"),
        ("spike before", before["spike_proxy"], "magma"),
        ("spike after", after["spike_proxy"], "magma"),
        ("quality before", before["quality_proxy"], "magma"),
        ("quality after", after["quality_proxy"], "magma"),
        ("low margin before", before["low_margin"], "magma"),
        ("low margin after", after["low_margin"], "magma"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.0))
    for ax, (title, data, cmap) in zip(axes.ravel(), panels):
        assert isinstance(data, np.ndarray)
        if "peak" in title and "change" not in title:
            vmin, vmax = 1, 40
        else:
            vmin, vmax = 0, float(np.quantile(data, 0.995))
            vmax = max(vmax, 1e-6)
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{display_name(stack_name)}: registration sensitivity")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_summary(
    out: Path,
    stack_name: str,
    reference_layer: int,
    shift_rows: list[dict[str, float | int]],
    compare: dict[str, float],
) -> None:
    mags = np.array([float(r["shift_mag_px"]) for r in shift_rows])
    sx = np.array([float(r["shift_x_px"]) for r in shift_rows])
    sy = np.array([float(r["shift_y_px"]) for r in shift_rows])
    lines = [
        f"# Real Focus-Stack Registration Probe: {stack_name}",
        "",
        f"Reference layer: {reference_layer}",
        "",
        "## Shift Summary",
        "",
        f"- Max shift magnitude: {mags.max():.4f} px",
        f"- Median shift magnitude: {np.median(mags):.4f} px",
        f"- X shift range: {sx.min():.4f} to {sx.max():.4f} px",
        f"- Y shift range: {sy.min():.4f} to {sy.max():.4f} px",
        "",
        "## Registration Sensitivity",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in [
        "peak_layer_changed_fraction",
        "peak_layer_mean_abs_change",
        "peak_layer_p90_abs_change",
        "spike_proxy_mean_before",
        "spike_proxy_mean_after",
        "spike_proxy_mean_delta",
        "spike_proxy_pearson_before_after",
        "quality_proxy_mean_before",
        "quality_proxy_mean_after",
        "quality_proxy_pearson_before_after",
        "low_margin_auc_spike_top10_before",
        "low_margin_auc_spike_top10_after",
        "quality_proxy_auc_spike_top10_before",
        "quality_proxy_auc_spike_top10_after",
        "sat_persistence_auc_spike_top10_before",
        "sat_persistence_auc_spike_top10_after",
    ]:
        lines.append(f"| `{key}` | {compare[key]:.4f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The estimated inter-layer shifts should be interpreted as a global registration diagnostic, not as a full optical calibration.",
            "If the peak-layer and quality maps remain highly correlated after alignment, the observed DFF instability is unlikely to be explained only by small global translations.",
        ]
    )
    (out / f"{safe_name(stack_name)}_registration_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(stack_dir: Path, out_dir: Path, max_side: int = 640, reference_layer: int | None = None) -> None:
    files = image_files(stack_dir)
    if not files:
        raise FileNotFoundError(f"No image files in {stack_dir}")
    arrays = [read_gray(p, max_side=max_side) for p in files]
    h = min(a.shape[0] for a in arrays)
    w = min(a.shape[1] for a in arrays)
    stack = np.stack([a[:h, :w] for a in arrays], axis=0)
    if reference_layer is None:
        reference_idx = stack.shape[0] // 2
    else:
        reference_idx = max(0, min(stack.shape[0] - 1, reference_layer - 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_name(stack_dir.name)
    shifts, shift_rows = estimate_shifts(stack, reference_idx)
    aligned = apply_shifts(stack, shifts)
    before = compute_diagnostics(stack)
    after = compute_diagnostics(aligned)
    compare = compare_diagnostics(before, after)

    with (out_dir / f"{safe}_registration_shifts.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(shift_rows[0].keys()))
        writer.writeheader()
        writer.writerows(shift_rows)

    compare_rows = [{"metric": k, "value": v} for k, v in compare.items()]
    with (out_dir / f"{safe}_registration_sensitivity.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(compare_rows)

    draw_shift_plot(stack_dir.name, shift_rows, out_dir / f"{safe}_registration_shifts.png")
    draw_comparison_maps(stack_dir.name, before, after, out_dir / f"{safe}_registration_sensitivity_maps.png")
    write_summary(out_dir, stack_dir.name, reference_idx + 1, shift_rows, compare)
    print(out_dir / f"{safe}_registration_summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", default="论文与PPT制作项目包/06_Samples/real_focus_stacks/钥匙纹路100um")
    parser.add_argument("--out", default="submission_planning/optical_mechanism_analysis/real_registration_probe")
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--reference-layer", type=int, default=20)
    args = parser.parse_args()
    analyze(Path(args.stack), Path(args.out), max_side=args.max_side, reference_layer=args.reference_layer)


if __name__ == "__main__":
    main()
