from __future__ import annotations

import argparse
import csv
import math
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
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def laplacian_energy(arr: np.ndarray) -> float:
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0
    center = arr[1:-1, 1:-1]
    lap = arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:] - 4.0 * center
    return float(np.mean(lap * lap))


def tenengrad(arr: np.ndarray) -> float:
    gy, gx = np.gradient(arr)
    return float(np.mean(gx * gx + gy * gy))


def phase_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
    h = min(reference.shape[0], moving.shape[0])
    w = min(reference.shape[1], moving.shape[1])
    ref = reference[:h, :w] - float(np.mean(reference[:h, :w]))
    mov = moving[:h, :w] - float(np.mean(moving[:h, :w]))
    if h < 8 or w < 8 or np.std(ref) < 1e-6 or np.std(mov) < 1e-6:
        return 0.0, 0.0
    f_ref = np.fft.fft2(ref)
    f_mov = np.fft.fft2(mov)
    cross = f_ref * np.conj(f_mov)
    cross /= np.maximum(np.abs(cross), 1e-9)
    corr = np.fft.ifft2(cross)
    peak = np.unravel_index(int(np.argmax(np.abs(corr))), corr.shape)
    dy, dx = float(peak[0]), float(peak[1])
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return dy, dx


def make_contact_sheet(stack_name: str, files: list[Path], arrays: list[np.ndarray], out_path: Path) -> None:
    if not arrays:
        return
    sample_count = min(8, len(arrays))
    idxs = np.linspace(0, len(arrays) - 1, sample_count).round().astype(int).tolist()
    fig, axes = plt.subplots(1, sample_count, figsize=(2.2 * sample_count, 2.4), squeeze=False)
    for ax, idx in zip(axes[0], idxs):
        ax.imshow(arrays[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{idx + 1}\n{files[idx].name}", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"{stack_name}: sampled focal layers", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_curves(stack_name: str, rows: list[dict[str, float | str]], out_path: Path) -> None:
    x = np.arange(1, len(rows) + 1)
    mean = np.array([float(r["mean"]) for r in rows])
    p99 = np.array([float(r["p99"]) for r in rows])
    sat = np.array([float(r["sat_ratio_098"]) for r in rows])
    lap = np.array([float(r["laplacian_energy"]) for r in rows])
    ten = np.array([float(r["tenengrad"]) for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True)
    axes[0].plot(x, mean, label="mean intensity", color="#355C7D")
    axes[0].plot(x, p99, label="p99 intensity", color="#C06C84")
    axes[0].set_ylabel("normalized intensity")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, sat * 100.0, color="#D95F02")
    axes[1].set_ylabel("pixels >= 0.98 (%)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, lap / max(lap.max(), 1e-9), label="Laplacian energy", color="#1B9E77")
    axes[2].plot(x, ten / max(ten.max(), 1e-9), label="Tenengrad", color="#7570B3")
    axes[2].set_ylabel("normalized focus response")
    axes[2].set_xlabel("focal layer index")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.suptitle(f"{stack_name}: brightness, saturation, and focus response", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_persistence(stack_name: str, arrays: list[np.ndarray], out_path: Path) -> dict[str, float]:
    stack = np.stack(arrays, axis=0)
    high = stack >= 0.98
    bright = stack >= 0.90
    sat_count = high.sum(axis=0)
    bright_count = bright.sum(axis=0)
    std_map = stack.std(axis=0)
    max_map = stack.max(axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.0))
    axes[0].imshow(max_map, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("max intensity", fontsize=9)
    axes[1].imshow(std_map, cmap="magma")
    axes[1].set_title("focal std", fontsize=9)
    axes[2].imshow(bright_count, cmap="inferno", vmin=0, vmax=max(1, stack.shape[0]))
    axes[2].set_title("layers >= 0.90", fontsize=9)
    axes[3].imshow(sat_count, cmap="inferno", vmin=0, vmax=max(1, stack.shape[0]))
    axes[3].set_title("layers >= 0.98", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{stack_name}: glare persistence diagnostics", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "pixels_saturated_any": float(np.mean(sat_count > 0)),
        "pixels_saturated_all": float(np.mean(sat_count == stack.shape[0])),
        "pixels_bright_any": float(np.mean(bright_count > 0)),
        "pixels_bright_half_or_more": float(np.mean(bright_count >= math.ceil(stack.shape[0] / 2))),
        "mean_focal_std": float(std_map.mean()),
        "p99_focal_std": float(np.quantile(std_map, 0.99)),
    }


def analyze_stack(stack_dir: Path, figures_dir: Path) -> tuple[list[dict[str, float | str]], dict[str, float | str]]:
    files = image_files(stack_dir)
    if not files:
        return [], {"stack": stack_dir.name, "num_layers": 0}

    arrays = [read_gray(p) for p in files]
    min_h = min(arr.shape[0] for arr in arrays)
    min_w = min(arr.shape[1] for arr in arrays)
    arrays = [arr[:min_h, :min_w] for arr in arrays]

    rows: list[dict[str, float | str]] = []
    ref = arrays[0]
    for i, (path, arr) in enumerate(zip(files, arrays), start=1):
        dy, dx = phase_shift(ref, arr)
        rows.append(
            {
                "stack": stack_dir.name,
                "layer": i,
                "file": str(path),
                "mean": float(np.mean(arr)),
                "p95": float(np.quantile(arr, 0.95)),
                "p99": float(np.quantile(arr, 0.99)),
                "max": float(np.max(arr)),
                "sat_ratio_098": float(np.mean(arr >= 0.98)),
                "bright_ratio_090": float(np.mean(arr >= 0.90)),
                "laplacian_energy": laplacian_energy(arr),
                "tenengrad": tenengrad(arr),
                "shift_dy_vs_first_px": dy,
                "shift_dx_vs_first_px": dx,
            }
        )

    safe_name = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", stack_dir.name)
    make_contact_sheet(stack_dir.name, files, arrays, figures_dir / f"{safe_name}_contact_sheet.png")
    make_curves(stack_dir.name, rows, figures_dir / f"{safe_name}_curves.png")
    persistence = make_persistence(stack_dir.name, arrays, figures_dir / f"{safe_name}_persistence.png")

    sat = np.array([float(r["sat_ratio_098"]) for r in rows])
    bright = np.array([float(r["bright_ratio_090"]) for r in rows])
    lap = np.array([float(r["laplacian_energy"]) for r in rows])
    shifts = np.array(
        [
            math.hypot(float(r["shift_dy_vs_first_px"]), float(r["shift_dx_vs_first_px"]))
            for r in rows
        ]
    )
    summary: dict[str, float | str] = {
        "stack": stack_dir.name,
        "num_layers": len(files),
        "height": min_h,
        "width": min_w,
        "mean_intensity_min": float(min(float(r["mean"]) for r in rows)),
        "mean_intensity_max": float(max(float(r["mean"]) for r in rows)),
        "p99_intensity_max": float(max(float(r["p99"]) for r in rows)),
        "max_sat_ratio_098": float(sat.max()),
        "max_bright_ratio_090": float(bright.max()),
        "best_focus_layer_laplacian": int(lap.argmax() + 1),
        "median_shift_vs_first_px": float(np.median(shifts)),
        "max_shift_vs_first_px": float(np.max(shifts)),
        **persistence,
    }
    return rows, summary


def write_summary_md(summaries: list[dict[str, float | str]], out_path: Path) -> None:
    lines = [
        "# Focus-Stack Optical Probe Summary",
        "",
        "This automatically generated summary reports lightweight diagnostics for selected real focal stacks.",
        "Intensity values are normalized to [0, 1]. Saturation is approximated as pixels with intensity >= 0.98.",
        "",
        "| Stack | Layers | Size | Max sat. ratio | Bright any | Bright half+ | Best focus layer | Median shift px | Max shift px |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        if int(s.get("num_layers", 0)) == 0:
            continue
        lines.append(
            "| {stack} | {num_layers} | {width}x{height} | {sat:.4f} | {bright_any:.4f} | {bright_half:.4f} | {best} | {med_shift:.2f} | {max_shift:.2f} |".format(
                stack=s["stack"],
                num_layers=s["num_layers"],
                width=s["width"],
                height=s["height"],
                sat=float(s["max_sat_ratio_098"]),
                bright_any=float(s["pixels_bright_any"]),
                bright_half=float(s["pixels_bright_half_or_more"]),
                best=s["best_focus_layer_laplacian"],
                med_shift=float(s["median_shift_vs_first_px"]),
                max_shift=float(s["max_shift_vs_first_px"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `Max sat. ratio` estimates whether a stack contains near-clipped highlights.",
            "- `Bright any` and `Bright half+` summarize whether bright/glare-prone pixels persist across focal layers.",
            "- `Median shift px` and `Max shift px` are phase-correlation diagnostics against the first layer; large values should be checked visually because specular changes can also bias this estimate.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="论文与PPT制作项目包/06_Samples/real_focus_stacks")
    parser.add_argument("--out", default="submission_planning/optical_mechanism_analysis")
    parser.add_argument(
        "--stacks",
        nargs="*",
        default=["3D表面", "3D层纹", "磕碰孔5um", "钥匙尖头50um", "钥匙纹路100um", "圆孔50um"],
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, float | str]] = []
    summaries: list[dict[str, float | str]] = []
    for name in args.stacks:
        stack_dir = root / name
        rows, summary = analyze_stack(stack_dir, figures_dir)
        all_rows.extend(rows)
        summaries.append(summary)

    metrics_path = out_dir / "focus_stack_optical_probe_metrics.csv"
    if all_rows:
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    summary_path = out_dir / "focus_stack_optical_probe_summary.md"
    write_summary_md(summaries, summary_path)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
