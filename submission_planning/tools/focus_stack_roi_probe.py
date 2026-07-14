from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def numeric_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def image_files(stack_dir: Path) -> list[Path]:
    files = [p for p in stack_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=numeric_key)


def read_gray(path: Path, max_side: int = 640) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        scale = min(1.0, max_side / max(im.size))
        if scale < 1.0:
            new_size = (max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale)))
            im = im.resize(new_size, Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def laplacian_energy(arr: np.ndarray) -> float:
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0
    center = arr[1:-1, 1:-1]
    lap = arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:] - 4 * center
    return float(np.mean(lap * lap))


def tenengrad(arr: np.ndarray) -> float:
    gy, gx = np.gradient(arr)
    return float(np.mean(gx * gx + gy * gy))


def roi_from_fraction(shape: tuple[int, int], frac: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    h, w = shape
    x0 = int(round(frac[0] * w))
    y0 = int(round(frac[1] * h))
    x1 = int(round(frac[2] * w))
    y1 = int(round(frac[3] * h))
    x0 = max(0, min(w - 2, x0))
    y0 = max(0, min(h - 2, y0))
    x1 = max(x0 + 2, min(w, x1))
    y1 = max(y0 + 2, min(h, y1))
    return x0, y0, x1, y1


def crop(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1]


def draw_roi_overview(stack_name: str, first: np.ndarray, rois: dict[str, tuple[int, int, int, int]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.imshow(first, cmap="gray", vmin=0, vmax=1)
    colors = {
        "highlight_edge": "#E41A1C",
        "ordinary_texture": "#377EB8",
        "dark_region": "#4DAF4A",
    }
    for name, (x0, y0, x1, y1) in rois.items():
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2.0, edgecolor=colors[name])
        ax.add_patch(rect)
        ax.text(x0, max(0, y0 - 5), name, color=colors[name], fontsize=8, weight="bold")
    ax.set_title(f"{stack_name}: ROI locations")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def draw_roi_curves(stack_name: str, rows: list[dict[str, float | str]], out: Path) -> None:
    names = ["highlight_edge", "ordinary_texture", "dark_region"]
    colors = {
        "highlight_edge": "#E41A1C",
        "ordinary_texture": "#377EB8",
        "dark_region": "#4DAF4A",
    }
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 9.2), sharex=True)
    for name in names:
        sub = [r for r in rows if r["roi"] == name]
        x = np.array([float(r["layer"]) for r in sub])
        mean = np.array([float(r["mean"]) for r in sub])
        p99 = np.array([float(r["p99"]) for r in sub])
        sat = np.array([float(r["sat_ratio_098"]) for r in sub])
        lap = np.array([float(r["laplacian_energy"]) for r in sub])
        ten = np.array([float(r["tenengrad"]) for r in sub])
        axes[0].plot(x, mean, color=colors[name], label=f"{name} mean")
        axes[1].plot(x, p99, color=colors[name], label=f"{name} p99")
        axes[2].plot(x, sat * 100.0, color=colors[name], label=f"{name} sat")
        axes[3].plot(x, lap / max(lap.max(), 1e-12), color=colors[name], linestyle="-", label=f"{name} Lap.")
        axes[3].plot(x, ten / max(ten.max(), 1e-12), color=colors[name], linestyle="--", label=f"{name} Ten.")

    axes[0].set_ylabel("mean intensity")
    axes[1].set_ylabel("p99 intensity")
    axes[2].set_ylabel("I>=0.98 (%)")
    axes[3].set_ylabel("normalized focus")
    axes[3].set_xlabel("focal layer index")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=7)
    fig.suptitle(f"{stack_name}: ROI-level brightness and focus response", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def draw_roi_montage(stack_name: str, arrays: list[np.ndarray], rois: dict[str, tuple[int, int, int, int]], out: Path) -> None:
    idxs = np.linspace(0, len(arrays) - 1, min(8, len(arrays))).round().astype(int).tolist()
    names = ["highlight_edge", "ordinary_texture", "dark_region"]
    fig, axes = plt.subplots(len(names), len(idxs), figsize=(1.65 * len(idxs), 1.65 * len(names)))
    for row, name in enumerate(names):
        box = rois[name]
        for col, idx in enumerate(idxs):
            ax = axes[row, col]
            patch = crop(arrays[idx], box)
            ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
            if row == 0:
                ax.set_title(f"L{idx + 1}", fontsize=8)
            if col == 0:
                ax.set_ylabel(name, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{stack_name}: ROI crops over focal layers", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def analyze(stack_dir: Path, out_dir: Path) -> None:
    files = image_files(stack_dir)
    if not files:
        raise FileNotFoundError(f"No image files in {stack_dir}")
    arrays = [read_gray(p) for p in files]
    h = min(a.shape[0] for a in arrays)
    w = min(a.shape[1] for a in arrays)
    arrays = [a[:h, :w] for a in arrays]

    stack = np.stack(arrays, axis=0)
    max_map = stack.max(axis=0)
    mean_map = stack.mean(axis=0)
    std_map = stack.std(axis=0)

    rois = {
        "highlight_edge": roi_from_fraction((h, w), (0.84, 0.18, 0.98, 0.86)),
        "ordinary_texture": roi_from_fraction((h, w), (0.34, 0.32, 0.52, 0.56)),
        "dark_region": roi_from_fraction((h, w), (0.56, 0.34, 0.76, 0.62)),
    }

    # Adjust the highlight ROI center to the brightest persistent area, while keeping
    # a broad edge crop around it.
    bright_score = max_map + std_map
    y, x = np.unravel_index(int(np.argmax(bright_score)), bright_score.shape)
    half_w = max(32, w // 14)
    half_h = max(48, h // 8)
    rois["highlight_edge"] = (
        max(0, x - half_w),
        max(0, y - half_h),
        min(w, x + half_w),
        min(h, y + half_h),
    )

    # Keep ordinary and dark ROIs away from the brightest edge if needed.
    if float(np.mean(crop(max_map, rois["ordinary_texture"]))) > 0.75:
        rois["ordinary_texture"] = roi_from_fraction((h, w), (0.25, 0.28, 0.45, 0.52))
    if float(np.mean(crop(mean_map, rois["dark_region"]))) > 0.30:
        rois["dark_region"] = roi_from_fraction((h, w), (0.52, 0.50, 0.75, 0.76))

    rows: list[dict[str, float | str]] = []
    for name, box in rois.items():
        for i, (path, arr) in enumerate(zip(files, arrays), start=1):
            roi = crop(arr, box)
            rows.append(
                {
                    "stack": stack_dir.name,
                    "roi": name,
                    "layer": i,
                    "file": str(path),
                    "x0": box[0],
                    "y0": box[1],
                    "x1": box[2],
                    "y1": box[3],
                    "mean": float(np.mean(roi)),
                    "p95": float(np.quantile(roi, 0.95)),
                    "p99": float(np.quantile(roi, 0.99)),
                    "sat_ratio_098": float(np.mean(roi >= 0.98)),
                    "bright_ratio_090": float(np.mean(roi >= 0.90)),
                    "laplacian_energy": laplacian_energy(roi),
                    "tenengrad": tenengrad(roi),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", stack_dir.name)
    metrics_path = out_dir / f"{safe}_roi_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    draw_roi_overview(stack_dir.name, arrays[0], rois, out_dir / f"{safe}_roi_locations.png")
    draw_roi_curves(stack_dir.name, rows, out_dir / f"{safe}_roi_curves.png")
    draw_roi_montage(stack_dir.name, arrays, rois, out_dir / f"{safe}_roi_montage.png")

    summary_lines = [
        f"# ROI Probe Summary: {stack_dir.name}",
        "",
        "| ROI | Box | max p99 | max sat I>=0.98 | best Laplacian layer | best Tenengrad layer |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, box in rois.items():
        sub = [r for r in rows if r["roi"] == name]
        p99 = np.array([float(r["p99"]) for r in sub])
        sat = np.array([float(r["sat_ratio_098"]) for r in sub])
        lap = np.array([float(r["laplacian_energy"]) for r in sub])
        ten = np.array([float(r["tenengrad"]) for r in sub])
        summary_lines.append(
            f"| {name} | {box} | {p99.max():.4f} | {sat.max():.4f} | {int(lap.argmax()+1)} | {int(ten.argmax()+1)} |"
        )
    summary_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The `highlight_edge` ROI is selected around the brightest persistent area in the focal stack. It should be interpreted as a diagnostic candidate rather than a manually verified defect annotation.",
            "A close match between high p99/saturation layers and focus-measure peak layers supports the hypothesis that glare edges can drive DFF focus selection.",
        ]
    )
    (out_dir / f"{safe}_roi_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(metrics_path)
    print(out_dir / f"{safe}_roi_summary.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", default="论文与PPT制作项目包/06_Samples/real_focus_stacks/钥匙纹路100um")
    parser.add_argument("--out", default="submission_planning/optical_mechanism_analysis/roi_probe")
    args = parser.parse_args()
    analyze(Path(args.stack), Path(args.out))


if __name__ == "__main__":
    main()
