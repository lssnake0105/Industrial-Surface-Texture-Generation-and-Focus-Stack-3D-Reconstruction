from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from dff_depth_direction import focus_index_to_height_um


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "结题交付包" / "05_图表与结果" / "眩光感知DFF原型"

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


SAMPLES = [
    ("3D打印层纹", ROOT / "DFFcode" / "ALL_IMAGES" / "3D层纹", "PLA", "层纹/周期性凹槽"),
    ("3D打印表面", ROOT / "DFFcode" / "ALL_IMAGES" / "3D表面", "PLA", "表面粗糙/填充缝隙"),
    ("金属表面磕碰孔", ROOT / "DFFcode" / "ALL_IMAGES" / "磕碰孔5um", "金属", "磕碰孔/局部凹陷"),
    ("钥匙纹路", ROOT / "DFFcode" / "ALL_IMAGES" / "钥匙纹路100um", "黄铜/镀层金属", "沟槽/刻痕纹理"),
]


def natural_key(path: Path) -> list[object]:
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts], key=natural_key)


def safe_name(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]+", "_", name)


def imread_gray(path: Path, max_dim: int = 760) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    h, w = img.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Cannot write image: {path}")
    buf.tofile(str(path))


def normalize_u8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if float(np.max(image) - np.min(image)) < 1e-6:
        return np.zeros_like(image, dtype=np.uint8)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def load_stack(folder: Path) -> tuple[list[Path], np.ndarray]:
    files = list_images(folder)
    images = [imread_gray(path) for path in files]
    shape = images[0].shape
    images = [cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA) if img.shape != shape else img for img in images]
    return files, np.stack(images, axis=2).astype(np.uint8)


def local_texture(image: np.ndarray, kernel: int = 11) -> np.ndarray:
    image_f = image.astype(np.float32)
    mean = cv2.boxFilter(image_f, -1, (kernel, kernel), normalize=True)
    mean2 = cv2.boxFilter(image_f * image_f, -1, (kernel, kernel), normalize=True)
    var = np.maximum(mean2 - mean * mean, 0)
    return cv2.GaussianBlur(normalize_u8(np.sqrt(var)).astype(np.float32) / 255.0, (0, 0), 1.0)


def glare_risk_for_layer(image: np.ndarray, global_high_threshold: float) -> tuple[np.ndarray, dict[str, float]]:
    image_f = image.astype(np.float32)
    hard_sat = image >= 250
    near_sat = image >= 245

    local_med = cv2.medianBlur(image, 31).astype(np.float32)
    local_excess = image_f - local_med
    soft_high = (image_f >= global_high_threshold) | ((image_f >= 180) & (local_excess >= 35))

    grad_x = cv2.Sobel(image_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    grad_threshold = max(float(np.percentile(grad, 99.2)), 18.0)
    high_edge = (grad >= grad_threshold) & (image_f >= np.percentile(image_f, 92.0))

    risk = np.zeros_like(image_f, dtype=np.float32)
    risk += hard_sat.astype(np.float32) * 1.0
    risk += near_sat.astype(np.float32) * 0.55
    risk += soft_high.astype(np.float32) * 0.42
    risk += high_edge.astype(np.float32) * 0.22
    risk = np.clip(risk, 0, 1)

    return risk, {
        "hard_sat": float(np.mean(hard_sat)),
        "near_sat": float(np.mean(near_sat)),
        "soft_high": float(np.mean(soft_high)),
        "high_edge": float(np.mean(high_edge)),
    }


def focus_measure(image: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    lap = np.abs(cv2.Laplacian(blur, cv2.CV_32F, ksize=3))
    sml = cv2.boxFilter(lap, -1, (9, 9), normalize=True)
    sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = cv2.boxFilter(sx * sx + sy * sy, -1, (7, 7), normalize=True)
    return 0.72 * sml + 0.28 * normalize_u8(tenengrad).astype(np.float32)


def reconstruct(stack: np.ndarray, glare_aware: bool) -> dict[str, np.ndarray | float]:
    images = [stack[:, :, i] for i in range(stack.shape[2])]
    global_high_threshold = max(220.0, float(np.percentile(stack, 99.3)))
    texture = local_texture(images[len(images) // 2])

    focus_maps = []
    risk_layers = []
    risk_stats = []
    for image in images:
        focus = focus_measure(image)
        risk_layer, stats = glare_risk_for_layer(image, global_high_threshold)
        risk_layers.append(risk_layer)
        risk_stats.append(stats)
        if glare_aware:
            # Risk-aware weighting happens before argmax, so it can change the selected focus layer.
            weight = np.clip(1.0 - 0.86 * risk_layer, 0.08, 1.0)
            weight *= 0.42 + 0.58 * texture
            focus = focus * weight
        focus_maps.append(focus.astype(np.float32))

    focus_volume = np.stack(focus_maps, axis=2)
    risk_volume = np.stack(risk_layers, axis=2)
    best = np.argmax(focus_volume, axis=2)
    height = focus_index_to_height_um(best, stack.shape[2], 100.0)
    height_u8 = normalize_u8(height)
    height_u8 = cv2.medianBlur(height_u8, 5)
    height_u8 = cv2.GaussianBlur(height_u8, (9, 9), 0)

    sorted_focus = np.sort(focus_volume, axis=2)
    peak = sorted_focus[:, :, -1]
    second = sorted_focus[:, :, -2] if stack.shape[2] > 1 else np.zeros_like(peak)
    ratio_conf = np.clip((peak - second) / (peak + 1e-6) * 3.0, 0, 1)

    peak_idx = best[:, :, None]
    prev_idx = np.maximum(best - 1, 0)[:, :, None]
    next_idx = np.minimum(best + 1, stack.shape[2] - 1)[:, :, None]
    prev_f = np.take_along_axis(focus_volume, prev_idx, axis=2)[:, :, 0]
    next_f = np.take_along_axis(focus_volume, next_idx, axis=2)[:, :, 0]
    peak_f = np.take_along_axis(focus_volume, peak_idx, axis=2)[:, :, 0]
    sharp_conf = np.clip((2 * peak_f - prev_f - next_f) / (peak_f + 1e-6), 0, 1)

    glare_risk = np.max(risk_volume, axis=2)
    glare_risk = cv2.GaussianBlur(glare_risk, (0, 0), 2.0)
    confidence = (0.46 * ratio_conf + 0.34 * sharp_conf + 0.20 * texture) * (1.0 - np.clip(glare_risk * 1.25, 0, 0.94))
    confidence = np.clip(confidence, 0, 1)

    risk_threshold = max(0.20, float(np.percentile(glare_risk, 97.0)))
    low_threshold = float(np.percentile(confidence, 18.0))
    repair_mask = ((glare_risk >= risk_threshold) | (confidence <= low_threshold)).astype(np.uint8) * 255
    repair_mask = cv2.morphologyEx(repair_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    repair_mask = cv2.dilate(repair_mask, np.ones((3, 3), np.uint8), iterations=1)
    repaired = cv2.inpaint(height_u8, repair_mask, 3, cv2.INPAINT_TELEA)
    blend_weight = np.clip(glare_risk * 1.4 + (confidence <= low_threshold).astype(np.float32) * 0.25, 0, 1)
    repaired = (height_u8.astype(np.float32) * (1 - blend_weight) + repaired.astype(np.float32) * blend_weight).astype(np.uint8)
    repaired = cv2.bilateralFilter(repaired, 7, 35, 35)

    return {
        "height_u8": height_u8,
        "pseudo": cv2.applyColorMap(height_u8, cv2.COLORMAP_TURBO),
        "repaired_u8": repaired,
        "repaired_pseudo": cv2.applyColorMap(repaired, cv2.COLORMAP_TURBO),
        "best_layer": normalize_u8(best.astype(np.float32)),
        "glare_risk": glare_risk,
        "glare_risk_u8": normalize_u8(glare_risk),
        "confidence": confidence,
        "confidence_u8": normalize_u8(confidence),
        "repair_mask": repair_mask,
        "global_high_threshold": global_high_threshold,
        "hard_sat_ratio": float(np.mean([s["hard_sat"] for s in risk_stats])),
        "near_sat_ratio": float(np.mean([s["near_sat"] for s in risk_stats])),
        "soft_high_ratio": float(np.mean([s["soft_high"] for s in risk_stats])),
        "high_edge_ratio": float(np.mean([s["high_edge"] for s in risk_stats])),
        "glare_risk_area": float(np.mean(glare_risk >= risk_threshold)),
        "low_conf_area": float(np.mean(confidence <= low_threshold)),
        "mean_confidence": float(np.mean(confidence)),
    }


def save_surface(height_u8: np.ndarray, title: str, out: Path) -> None:
    small = cv2.resize(height_u8, (220, 180), interpolation=cv2.INTER_AREA)
    y = np.arange(small.shape[0])
    x = np.arange(small.shape[1])
    x_grid, y_grid = np.meshgrid(x, y)
    fig = plt.figure(figsize=(6.3, 4.8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_grid, y_grid, small.astype(np.float32), cmap="viridis", linewidth=0, antialiased=True)
    ax.view_init(elev=42, azim=-58)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X / pixel")
    ax.set_ylabel("Y / pixel")
    ax.set_zlabel("relative height")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def make_comparison_panel(name: str, material: str, defect: str, stack: np.ndarray, original: dict, aware: dict, out: Path) -> None:
    mid = stack[:, :, stack.shape[2] // 2]
    glare_overlay = cv2.cvtColor(mid, cv2.COLOR_GRAY2RGB)
    risk = aware["glare_risk"]
    mask = risk >= max(0.20, float(np.percentile(risk, 97.0)))
    overlay = glare_overlay.copy()
    overlay[mask] = (244, 63, 94)
    glare_overlay = cv2.addWeighted(glare_overlay, 0.70, overlay, 0.30, 0)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), dpi=170)
    panels = [
        (mid, "代表帧", "gray"),
        (glare_overlay, "眩光风险叠加", None),
        (cv2.cvtColor(original["pseudo"], cv2.COLOR_BGR2RGB), "原始 DFF", None),
        (cv2.cvtColor(aware["pseudo"], cv2.COLOR_BGR2RGB), "眩光降权 DFF", None),
        (aware["confidence_u8"], "焦点置信度", "magma"),
        (aware["repair_mask"], "低置信补全掩膜", "gray"),
        (cv2.cvtColor(aware["repaired_pseudo"], cv2.COLOR_BGR2RGB), "置信度补全结果", None),
        (aware["best_layer"], "加权最佳聚焦层", "viridis"),
    ]
    for ax, (img, title, cmap) in zip(axes.flat, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"{name}：GA-CG-DFF 原型对比 | {material} | {defect}", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def process_sample(name: str, folder: Path, material: str, defect: str) -> dict:
    files, stack = load_stack(folder)
    out_dir = OUT / safe_name(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    original = reconstruct(stack, glare_aware=False)
    aware = reconstruct(stack, glare_aware=True)
    abs_diff = cv2.absdiff(original["height_u8"], aware["height_u8"])
    repair_diff = cv2.absdiff(aware["height_u8"], aware["repaired_u8"])

    imwrite(out_dir / "01_representative_frame.png", stack[:, :, stack.shape[2] // 2])
    imwrite(out_dir / "02_original_dff_height.png", original["height_u8"])
    imwrite(out_dir / "03_original_dff_pseudo.png", original["pseudo"])
    imwrite(out_dir / "04_glare_risk_map.png", aware["glare_risk_u8"])
    imwrite(out_dir / "05_glare_aware_height.png", aware["height_u8"])
    imwrite(out_dir / "06_glare_aware_pseudo.png", aware["pseudo"])
    imwrite(out_dir / "07_confidence_map.png", aware["confidence_u8"])
    imwrite(out_dir / "08_repair_mask.png", aware["repair_mask"])
    imwrite(out_dir / "09_confidence_repaired_height.png", aware["repaired_u8"])
    imwrite(out_dir / "10_confidence_repaired_pseudo.png", aware["repaired_pseudo"])
    imwrite(out_dir / "11_original_vs_glare_aware_diff.png", normalize_u8(abs_diff))
    imwrite(out_dir / "12_repair_diff.png", normalize_u8(repair_diff))

    save_surface(original["height_u8"], f"{name} 原始 DFF", out_dir / "13_surface_original.png")
    save_surface(aware["repaired_u8"], f"{name} 眩光感知补全", out_dir / "14_surface_glare_aware.png")
    make_comparison_panel(name, material, defect, stack, original, aware, out_dir / "15_glare_aware_comparison_panel.png")

    return {
        "sample": name,
        "material": material,
        "defect": defect,
        "frames": len(files),
        "high_threshold": round(float(aware["global_high_threshold"]), 2),
        "hard_saturation_percent": round(float(aware["hard_sat_ratio"]) * 100, 4),
        "near_saturation_percent": round(float(aware["near_sat_ratio"]) * 100, 4),
        "soft_highlight_percent": round(float(aware["soft_high_ratio"]) * 100, 4),
        "highlight_edge_percent": round(float(aware["high_edge_ratio"]) * 100, 4),
        "glare_risk_area_percent": round(float(aware["glare_risk_area"]) * 100, 4),
        "low_confidence_area_percent": round(float(aware["low_conf_area"]) * 100, 4),
        "mean_confidence": round(float(aware["mean_confidence"]), 4),
        "mean_abs_height_change": round(float(np.mean(abs_diff)), 3),
        "mean_repair_change": round(float(np.mean(repair_diff)), 3),
        "result_panel": str(out_dir / "15_glare_aware_comparison_panel.png"),
    }


def write_summary(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "glare_aware_dff_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "glare_aware_dff_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    ranked = sorted(rows, key=lambda row: (row["glare_risk_area_percent"], row["soft_highlight_percent"]), reverse=True)
    lines = [
        "# GA-CG-DFF 原型执行结果",
        "",
        "该结果是后续探索原型，不替代结题主结果。它用于证明“眩光/强反射应在焦点层选择前处理”，而不是只做高度图后处理。",
        "",
        "| 样品 | 软高亮/% | 眩光风险面积/% | 低置信面积/% | 平均置信度 | 原始-降权平均差 | 判断 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        if row["glare_risk_area_percent"] > 1.5 or row["soft_highlight_percent"] > 1.0:
            judgment = "眩光主例"
        elif row["low_confidence_area_percent"] > 18:
            judgment = "低置信/低纹理主例"
        else:
            judgment = "对照"
        lines.append(
            f"| {row['sample']} | {row['soft_highlight_percent']} | {row['glare_risk_area_percent']} | "
            f"{row['low_confidence_area_percent']} | {row['mean_confidence']} | {row['mean_abs_height_change']} | {judgment} |"
        )
    lines.extend(
        [
            "",
            "## 可写入材料的结论",
            "",
            "眩光感知原型将高亮/饱和/高光边缘区域转化为风险权重，并在聚焦评价阶段对这些区域降权。与二维 SG 空间滤波相比，该方法的关键区别是：它在最佳聚焦层选择前处理眩光伪峰，能够输出风险图、置信度图和低置信区域补全结果，因此更适合作为后续算法主线。",
            "",
            "当前原型仍属于启发式算法，后续需要通过标准样块和受控光照实验验证其是否真实降低深度误差。",
        ]
    )
    (OUT / "GA-CG-DFF原型执行结果.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [process_sample(*sample) for sample in SAMPLES]
    write_summary(rows)
    print(f"Wrote GA-CG-DFF prototype outputs to: {OUT}")


if __name__ == "__main__":
    main()
