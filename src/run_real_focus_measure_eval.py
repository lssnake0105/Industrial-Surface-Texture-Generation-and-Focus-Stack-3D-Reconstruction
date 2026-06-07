from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from dff_depth_direction import focus_index_to_relative_height


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "DFFcode" / "ALL_IMAGES"
OUT_ROOT = ROOT / "结题交付包" / "05_图表与结果" / "实物样本_混合聚焦评价DFF"

SAMPLES = [
    "3D层纹",
    "3D表面",
    "磕碰孔5um",
    "钥匙纹路100um",
    "圆孔50um",
    "钥匙尖头50um",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\Deng.ttf"),
]
for font_path in FONT_CANDIDATES:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(font_path)).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False


def natural_key(path: Path) -> list[object]:
    parts: list[object] = []
    buf = ""
    is_digit = False
    for ch in path.stem:
        if ch.isdigit():
            if buf and not is_digit:
                parts.append(buf.lower())
                buf = ""
            buf += ch
            is_digit = True
        else:
            if buf and is_digit:
                parts.append(int(buf))
                buf = ""
            buf += ch
            is_digit = False
    if buf:
        parts.append(int(buf) if is_digit else buf.lower())
    return parts


def imread_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    buf.tofile(str(path))


def normalize01(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def colorize(arr: np.ndarray, cmap: int) -> np.ndarray:
    u8 = np.clip(normalize01(arr) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cmap)


def focus_maps_from_stack(stack: np.ndarray) -> np.ndarray:
    maps = []
    for layer in stack:
        u8 = np.clip(layer * 255, 0, 255).astype(np.uint8)
        blur = cv2.GaussianBlur(u8, (3, 3), 0)
        lap = np.abs(cv2.Laplacian(blur, cv2.CV_32F, ksize=3))
        sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        tenengrad = sx * sx + sy * sy
        focus = cv2.boxFilter(lap, -1, (7, 7), normalize=True)
        focus += 0.0018 * cv2.boxFilter(tenengrad, -1, (7, 7), normalize=True)
        maps.append(focus.astype(np.float32))
    return np.stack(maps, axis=0)


def dff_from_focus(focus: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.argmax(focus, axis=0).astype(np.float32)
    depth = focus_index_to_relative_height(idx, focus.shape[0])
    sorted_focus = np.sort(focus, axis=0)
    peak = sorted_focus[-1]
    second = sorted_focus[-2] if focus.shape[0] > 1 else np.zeros_like(peak)
    conf_raw = (peak - second) / (peak + 1e-6)
    scale = float(np.percentile(conf_raw, 98.5)) + 1e-6
    conf = np.clip(conf_raw / scale, 0, 1).astype(np.float32)
    return idx, depth.astype(np.float32), peak.astype(np.float32), conf


def save_panel(
    out_dir: Path,
    sample_name: str,
    stack: np.ndarray,
    idx: np.ndarray,
    depth: np.ndarray,
    peak: np.ndarray,
    conf: np.ndarray,
) -> None:
    mid = stack.shape[0] // 2
    best_med = int(round(float(np.median(idx))))
    best_med = max(0, min(stack.shape[0] - 1, best_med))

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), dpi=160)
    items = [
        (stack[mid], f"代表帧：第 {mid + 1} 层", "gray"),
        (stack[best_med], f"中位最佳层原图：第 {best_med + 1} 层", "gray"),
        (idx + 1, "最佳聚焦层号", "turbo"),
        (depth, "DFF 相对高度/深度", "viridis"),
        (normalize01(peak), "峰值聚焦响应", "magma"),
        (conf, "峰值置信度", "magma"),
    ]
    for ax, (arr, title, cmap) in zip(axes.flat, items):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        if cmap != "gray":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"{sample_name}：Box7(|Laplacian(G)|) + 0.0018·Box7(Tenengrad)", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / "06_focus_formula_comparison_panel.png", bbox_inches="tight")
    plt.close(fig)


def save_3d_preview(out_dir: Path, sample_name: str, depth: np.ndarray) -> None:
    h, w = depth.shape
    stride = max(1, max(h, w) // 180)
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    zz = depth[::stride, ::stride]
    fig = plt.figure(figsize=(9.2, 6.8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title(f"{sample_name}：DFF 相对三维预览")
    ax.set_xlabel("x / pixel")
    ax.set_ylabel("y / pixel")
    ax.set_zlabel("relative z")
    ax.view_init(elev=34, azim=-132)
    fig.tight_layout()
    fig.savefig(out_dir / "07_relative_3d_preview.png", bbox_inches="tight")
    plt.close(fig)


def load_sample_stack(sample_dir: Path) -> tuple[np.ndarray, list[Path]]:
    files = sorted([p for p in sample_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=natural_key)
    if not files:
        raise RuntimeError(f"No images found in: {sample_dir}")
    frames = [imread_gray(path) for path in files]
    h, w = frames[0].shape
    aligned = []
    for frame in frames:
        if frame.shape != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        aligned.append(frame.astype(np.float32))
    return np.stack(aligned, axis=0), files


def process_sample(sample_name: str) -> dict[str, object]:
    sample_dir = DATA_ROOT / sample_name
    out_dir = OUT_ROOT / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stack, files = load_sample_stack(sample_dir)
    focus = focus_maps_from_stack(stack)
    idx, depth, peak, conf = dff_from_focus(focus)

    mid = stack.shape[0] // 2
    write_image(out_dir / "01_representative_frame.png", np.clip(stack[mid] * 255, 0, 255).astype(np.uint8))
    write_image(out_dir / "02_best_focus_layer_index.png", colorize(idx + 1, cv2.COLORMAP_TURBO))
    write_image(out_dir / "03_focus_peak_response.png", colorize(peak, cv2.COLORMAP_MAGMA))
    write_image(out_dir / "04_focus_confidence.png", colorize(conf, cv2.COLORMAP_MAGMA))
    write_image(out_dir / "05_relative_dff_depth.png", colorize(depth, cv2.COLORMAP_VIRIDIS))
    np.save(out_dir / "focus_layer_index.npy", idx.astype(np.float32))
    np.save(out_dir / "relative_dff_depth.npy", depth.astype(np.float32))
    np.save(out_dir / "focus_confidence.npy", conf.astype(np.float32))

    save_panel(out_dir, sample_name, stack, idx, depth, peak, conf)
    save_3d_preview(out_dir, sample_name, depth)

    high_conf = conf > 0.55
    mid_conf = (conf > 0.25) & (conf <= 0.55)
    row = {
        "sample": sample_name,
        "image_count": int(stack.shape[0]),
        "height": int(stack.shape[1]),
        "width": int(stack.shape[2]),
        "mean_confidence": float(np.mean(conf)),
        "median_confidence": float(np.median(conf)),
        "high_conf_area_percent": float(np.mean(high_conf) * 100),
        "mid_conf_area_percent": float(np.mean(mid_conf) * 100),
        "median_best_layer_1based": float(np.median(idx + 1)),
        "p10_best_layer_1based": float(np.percentile(idx + 1, 10)),
        "p90_best_layer_1based": float(np.percentile(idx + 1, 90)),
        "panel": str(out_dir / "06_focus_formula_comparison_panel.png"),
        "preview_3d": str(out_dir / "07_relative_3d_preview.png"),
        "first_file": files[0].name,
        "last_file": files[-1].name,
    }
    (out_dir / "sample_metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def write_report(rows: list[dict[str, object]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "real_focus_formula_metrics.csv"
    fields = [
        "sample",
        "image_count",
        "height",
        "width",
        "mean_confidence",
        "median_confidence",
        "high_conf_area_percent",
        "mid_conf_area_percent",
        "median_best_layer_1based",
        "p10_best_layer_1based",
        "p90_best_layer_1based",
        "panel",
        "preview_3d",
        "first_file",
        "last_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 实物样本混合聚焦评价函数 DFF 运行结果",
        "",
        "聚焦评价函数：",
        "",
        "```text",
        "F_k(x, y) = Box7(|Laplacian(G(I_k))|)",
        "          + 0.0018 * Box7(Sobel_x(G(I_k))^2 + Sobel_y(G(I_k))^2)",
        "```",
        "",
        "说明：所有输出均为未标定的相对高度/相对焦层结果，适合观察清晰度峰值稳定性和缺陷形貌可视化效果，不作为绝对深度精度。",
        "",
        "| 样本 | 图像数 | 分辨率 | 平均置信度 | 高置信面积/% | 最佳层中位数 | 预览图 | 三维预览 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        sample = str(row["sample"])
        rel_panel = Path(str(row["panel"])).relative_to(OUT_ROOT).as_posix()
        rel_3d = Path(str(row["preview_3d"])).relative_to(OUT_ROOT).as_posix()
        lines.append(
            f"| {sample} | {row['image_count']} | {row['width']}×{row['height']} | "
            f"{float(row['mean_confidence']):.3f} | {float(row['high_conf_area_percent']):.2f} | "
            f"{float(row['median_best_layer_1based']):.1f} | "
            f"[panel]({rel_panel}) | [3D]({rel_3d}) |"
        )

    lines.extend(
        [
            "",
            "## 快速解读",
            "",
            "- `最佳聚焦层号` 越连续，说明该评价函数在该样品上的 DFF 层选择越稳定。",
            "- `峰值置信度` 越亮，说明第一峰与第二峰差距越大，DFF 判断越明确。",
            "- 如果高亮金属区域出现碎片化焦层或低置信区域，通常说明眩光/杂散光仍在干扰聚焦评价。",
            "- `DFF 相对高度/深度` 只表示焦层相对位置，未经过 z 轴标定，不应写成绝对高度。",
        ]
    )
    (OUT_ROOT / "实物样本_混合聚焦评价DFF_运行报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = []
    for sample in SAMPLES:
        rows.append(process_sample(sample))
    write_report(rows)
    print(f"Wrote {len(rows)} samples to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
