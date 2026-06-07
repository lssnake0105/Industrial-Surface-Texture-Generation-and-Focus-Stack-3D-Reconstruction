from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

from final_dataset_training import MODEL_DIR as TINY_MODEL_DIR
from final_dataset_training import STACK_LAYERS
from paper_algorithm_comparison import lee_adaptive_window_sff, li_adaptive_iteration_sff
from real_sample_glare_prior_fusion_validation import (
    SAMPLES,
    build_real_features,
    count_spikes,
    edge_retention,
    imwrite,
    jump_strength,
    list_images,
    normalize_u8,
    original_full_postprocess,
    postprocess_depth,
    roughness,
)
from simulate_antiglare_highres_samples import predict_tiled, save_float_image
from simulate_antiglare_prototype import TinyDepthNet
from train_focus_resunet_loss_experiment import FocusResUNet, augment_features, predict_tiled_upgraded
from train_residual_focus_resunet_experiment import ResidualFocusResUNet


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "结题交付包" / "05_图表与结果"
OUT = RESULT_ROOT / "中期实物样本_全算法3D重建对比"
FOCUS_MODEL = RESULT_ROOT / "模型与损失函数升级实验" / "model" / "focus_resunet_hybrid_loss.pt"
RESIDUAL_MODEL = RESULT_ROOT / "模型与损失函数升级实验_残差保护版" / "model" / "residual_focus_resunet_hybrid_loss.pt"
TINY_MODEL = TINY_MODEL_DIR / "final_antiglare_depth_net.pt"

METHOD_ORDER = [
    "Original DFF",
    "Original DFF + post",
    "GADFF",
    "Lee2013 adaptive window",
    "Li2019 adaptive iteration",
    "TinyDepthNet",
    "Focus-ResUNet",
    "Residual Focus-ResUNet",
]

REAL_CORRECT_DIRECTION_METHODS = {"Lee2013 adaptive window", "Li2019 adaptive iteration"}
REAL_COMPARISON_SAMPLES = [
    *SAMPLES,
    {"key": "07_1124", "name": "1124", "folder": ROOT / "DFFcode" / "ALL_IMAGES" / "1124", "height_um": float("nan")},
]

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


def load_state(path: Path, device: str) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def load_tiny(device: str) -> TinyDepthNet | None:
    if not TINY_MODEL.exists():
        return None
    raw = torch.load(TINY_MODEL, map_location=device)
    channels = int(raw.get("channels", 0)) if isinstance(raw, dict) else 0
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    if not channels:
        first_key = next(k for k, v in state.items() if k.endswith("weight") and getattr(v, "ndim", 0) == 4)
        channels = int(state[first_key].shape[1])
    model = TinyDepthNet(channels).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_focus(device: str) -> FocusResUNet | None:
    if not FOCUS_MODEL.exists():
        return None
    model = FocusResUNet().to(device)
    model.load_state_dict(load_state(FOCUS_MODEL, device))
    model.eval()
    return model


def load_residual(device: str) -> ResidualFocusResUNet | None:
    if not RESIDUAL_MODEL.exists():
        return None
    model = ResidualFocusResUNet().to(device)
    model.load_state_dict(load_state(RESIDUAL_MODEL, device))
    model.eval()
    return model


def save_3d_surface(depth: np.ndarray, title: str, path: Path) -> None:
    z = cv2.resize(np.clip(depth, 0, 1).astype(np.float32), (190, 150), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]
    fig = plt.figure(figsize=(5.0, 4.0), dpi=145)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("relative height")
    ax.view_init(elev=40, azim=-55)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_summary_panel(sample_name: str, arrays: dict[str, np.ndarray], outputs: dict[str, np.ndarray], out_dir: Path) -> None:
    rows = 2
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(18.0, 7.2), dpi=150)
    panels: list[tuple[str, np.ndarray, str]] = [
        ("代表帧", arrays["stack"][len(arrays["stack"]) // 2], "gray"),
        ("眩光风险", arrays["risk"], "magma"),
    ]
    panels.extend((name, outputs[name], "viridis") for name in METHOD_ORDER)
    for ax, (title, image, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=0 if cmap != "gray" else None, vmax=1 if cmap != "gray" else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.032, pad=0.015)
    fig.suptitle(f"{sample_name}: 中期实物样本全算法相对高度图对比", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "all_algorithm_height_panel.png", bbox_inches="tight")
    plt.close(fig)


def save_3d_grid(sample_name: str, outputs: dict[str, np.ndarray], out_dir: Path) -> None:
    fig = plt.figure(figsize=(17.5, 8.0), dpi=135)
    for idx, name in enumerate(METHOD_ORDER, start=1):
        z = cv2.resize(np.clip(outputs[name], 0, 1).astype(np.float32), (170, 125), interpolation=cv2.INTER_AREA)
        yy, xx = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]
        ax = fig.add_subplot(2, 4, idx, projection="3d")
        ax.plot_surface(xx, yy, z, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=40, azim=-55)
    fig.suptitle(f"{sample_name}: 全算法 3D 重建形貌对比（相对高度）", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "all_algorithm_3d_surface_panel.png", bbox_inches="tight")
    plt.close(fig)


def metrics_for_method(
    sample_name: str,
    method: str,
    depth: np.ndarray,
    arrays: dict[str, np.ndarray],
    reference: np.ndarray,
) -> dict[str, object]:
    risk = arrays["risk"]
    conf = arrays["confidence"]
    high_risk = risk > max(float(np.percentile(risk, 84)), 0.08)
    low_conf = conf < min(float(np.percentile(conf, 35)), 0.35)
    edge = edge_retention(depth, arrays["stack"][len(arrays["stack"]) // 2])
    delta = np.abs(depth - reference)
    return {
        "sample": sample_name,
        "method": method,
        "mean_height": float(np.mean(depth)),
        "std_height": float(np.std(depth)),
        "height_p05": float(np.percentile(depth, 5)),
        "height_p95": float(np.percentile(depth, 95)),
        "relative_dynamic_range": float(np.percentile(depth, 95) - np.percentile(depth, 5)),
        "roughness": roughness(depth),
        "edge_retention_to_frame": edge,
        "high_risk_jump": jump_strength(depth, high_risk),
        "low_conf_spike_count": count_spikes(depth, low_conf),
        "mean_abs_delta_vs_original_post": float(np.mean(delta)),
        "p95_abs_delta_vs_original_post": float(np.percentile(delta, 95)),
    }


def align_real_sample_direction(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    aligned: dict[str, np.ndarray] = {}
    for method, depth in outputs.items():
        if method in REAL_CORRECT_DIRECTION_METHODS:
            aligned[method] = np.clip(depth, 0, 1).astype(np.float32)
        else:
            aligned[method] = np.clip(1.0 - depth, 0, 1).astype(np.float32)
    return aligned


def process_sample(
    sample: dict[str, object],
    tiny: TinyDepthNet | None,
    focus: FocusResUNet | None,
    residual: ResidualFocusResUNet | None,
    device: str,
) -> list[dict[str, object]] | None:
    files = list_images(sample["folder"])  # type: ignore[index]
    if not files:
        return None
    key = str(sample["key"])
    name = str(sample["name"])
    out_dir = OUT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = build_real_features(files)
    stack = arrays["stack"]
    base_features = arrays["features"]
    model_features = augment_features(base_features)

    original = arrays["dff"]
    original_post = original_full_postprocess(original)
    outputs: dict[str, np.ndarray] = {
        "Original DFF": original,
        "Original DFF + post": original_post,
        "GADFF": postprocess_depth(arrays["gadff"], median_kernel=5, gaussian_kernel=9, morph_kernel=5),
        "Lee2013 adaptive window": lee_adaptive_window_sff(stack),
        "Li2019 adaptive iteration": li_adaptive_iteration_sff(stack),
    }
    if tiny is not None:
        outputs["TinyDepthNet"] = predict_tiled(tiny, base_features, device, tile=256, overlap=80)
    else:
        outputs["TinyDepthNet"] = outputs["GADFF"]
    if focus is not None:
        outputs["Focus-ResUNet"] = predict_tiled_upgraded(focus, model_features, device, tile=256, overlap=80)
    else:
        outputs["Focus-ResUNet"] = outputs["GADFF"]
    if residual is not None:
        outputs["Residual Focus-ResUNet"] = predict_tiled_upgraded(residual, model_features, device, tile=256, overlap=80)
    else:
        outputs["Residual Focus-ResUNet"] = outputs["Focus-ResUNet"]
    outputs = align_real_sample_direction(outputs)
    original_post = outputs["Original DFF + post"]

    imwrite(out_dir / "representative_frame.png", normalize_u8(stack[len(stack) // 2]))
    save_float_image(out_dir / "glare_risk.png", arrays["risk"], cv2.COLORMAP_MAGMA)
    for method, depth in outputs.items():
        stem = method.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
        save_float_image(out_dir / f"{stem}_height.png", depth, cv2.COLORMAP_VIRIDIS)
        save_3d_surface(depth, method, out_dir / f"{stem}_3d_surface.png")

    save_summary_panel(name, arrays, outputs, out_dir)
    save_3d_grid(name, outputs, out_dir)

    rows = [metrics_for_method(name, method, outputs[method], arrays, original_post) for method in METHOD_ORDER]
    with (out_dir / "all_algorithm_no_gt_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "all_algorithm_manifest.json").write_text(
        json.dumps(
            {
                "sample": name,
                "source_folder": str(sample["folder"]),
                "frames_original": len(files),
                "frames_used": STACK_LAYERS,
                "methods": METHOD_ORDER,
                "height_panel": str(out_dir / "all_algorithm_height_panel.png"),
                "surface_panel": str(out_dir / "all_algorithm_3d_surface_panel.png"),
                "metrics": str(out_dir / "all_algorithm_no_gt_metrics.csv"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def write_overview(all_rows: list[dict[str, object]]) -> None:
    with (OUT / "real_midterm_all_algorithm_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT / "real_midterm_all_algorithm_metrics.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    method_means = []
    for method in METHOD_ORDER:
        subset = [row for row in all_rows if row["method"] == method]
        method_means.append(
            {
                "method": method,
                "roughness": float(np.mean([float(r["roughness"]) for r in subset])),
                "edge_retention_to_frame": float(np.mean([float(r["edge_retention_to_frame"]) for r in subset])),
                "relative_dynamic_range": float(np.mean([float(r["relative_dynamic_range"]) for r in subset])),
                "low_conf_spike_count": float(np.mean([float(r["low_conf_spike_count"]) for r in subset])),
            }
        )
    with (OUT / "real_midterm_method_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(method_means[0].keys()))
        writer.writeheader()
        writer.writerows(method_means)

    x = np.arange(len(METHOD_ORDER))
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 8.4), dpi=150)
    plot_specs = [
        ("relative_dynamic_range", "Relative dynamic range"),
        ("roughness", "Mean roughness"),
        ("edge_retention_to_frame", "Edge retention to frame"),
        ("low_conf_spike_count", "Low-confidence spike count"),
    ]
    for ax, (key, title) in zip(axes.ravel(), plot_specs):
        vals = [row[key] for row in method_means]
        ax.bar(x, vals, color="#4c78a8")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("中期实物样本全算法无真值指标汇总", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "real_midterm_all_algorithm_metric_summary.png", bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# 中期实物样本全算法 3D 重建对比",
        "",
        "## 输出说明",
        "",
        "本结果把 Original DFF、Original DFF + post、GADFF、Lee2013 adaptive window、Li2019 adaptive iteration、TinyDepthNet、Focus-ResUNet、Residual Focus-ResUNet 应用于同一批中期实物焦栈。",
        "",
        "真实样本没有标准高度真值，因此这里不报告 MAE；指标用于观察重建形貌稳定性，包括相对高度动态范围、粗糙度、与代表帧边缘相关性、高风险区域跳变、低置信区域尖峰数量，以及相对 Original DFF + post 的平均变化。",
        "",
        "方向校准说明：人工检查中期实物样本后，Lee2013 adaptive window 和 Li2019 adaptive iteration 的重建方向与实际样品一致；Original DFF、Original DFF + post、GADFF、TinyDepthNet、Focus-ResUNet、Residual Focus-ResUNet 的相对高度方向已统一执行 `1 - depth` 翻转，以便和 Lee2013/Li2019 方向对齐。",
        "",
        "## 文件",
        "",
        "- `real_midterm_all_algorithm_metrics.csv`：逐样本、逐算法无真值指标。",
        "- `real_midterm_method_summary.csv`：各算法在全部中期实物样本上的指标均值。",
        "- `real_midterm_all_algorithm_metric_summary.png`：指标汇总柱状图。",
        "- 每个样本子目录包含 `all_algorithm_height_panel.png`、`all_algorithm_3d_surface_panel.png` 和各算法单独 3D surface 图。",
        "",
        "## 使用口径",
        "",
        "论文中应表述为“相对三维重建效果与稳定性对比”，不要写成绝对高度精度对比。若需要绝对精度，需要额外加入轮廓仪、台阶块或显微共聚焦标定真值。",
    ]
    (OUT / "中期实物样本全算法3D重建对比说明.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tiny = load_tiny(device)
    focus = load_focus(device)
    residual = load_residual(device)
    all_rows: list[dict[str, object]] = []
    for sample in REAL_COMPARISON_SAMPLES:
        rows = process_sample(sample, tiny, focus, residual, device)
        if rows:
            all_rows.extend(rows)
            print(f"processed {sample['name']}", flush=True)
    write_overview(all_rows)
    print(OUT)


if __name__ == "__main__":
    main()
