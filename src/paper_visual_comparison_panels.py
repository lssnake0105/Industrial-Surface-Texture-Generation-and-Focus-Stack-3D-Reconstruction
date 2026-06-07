from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from final_dataset_training import build_dataset
from paper_algorithm_comparison import (
    DEFAULT_STACK_LAYERS,
    dff_original,
    lee_adaptive_window_sff,
    li_adaptive_iteration_sff,
)
from real_sample_glare_prior_fusion_validation import original_full_postprocess
from simulate_antiglare_highres_samples import generate_sample_arrays


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "结题交付包" / "05_图表与结果"
OUT = RESULT_ROOT / "论文算法对比与结论收束"
REAL_ROOT = RESULT_ROOT / "实物样本_混合聚焦评价DFF"
REAL_ALL_ALG_ROOT = RESULT_ROOT / "中期实物样本_全算法3D重建对比"
FOCUS_VIS_ROOT = RESULT_ROOT / "模型与损失函数升级实验" / "test"
METRICS_CSV = OUT / "paper_algorithm_comparison_metrics.csv"

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


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def resize_for_panel(arr: np.ndarray, width: int = 320) -> np.ndarray:
    h, w = arr.shape[:2]
    if w <= width:
        return arr
    scale = width / float(w)
    return cv2.resize(arr, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def method_value(df: pd.DataFrame, sample: str, method: str, key: str = "mae_um") -> float:
    row = df[(df["sample"] == sample) & (df["method"] == method)].iloc[0]
    return float(row[key])


def focus_resunet_visual(sample: str) -> np.ndarray:
    return read_image(FOCUS_VIS_ROOT / sample / "08_patch_finetuned_cnn.png")


def selected_simulation_samples() -> list[str]:
    return [
        "test_A型突起刃脊_柏林粗糙",
        "test_山脊_柏林粗糙",
        "test_阶跃_柏林粗糙",
        "test_山峰_分形粗糙",
        "test_周期_条纹粗糙",
    ]


def save_simulation_panel() -> None:
    df = pd.read_csv(METRICS_CSV)
    dataset = build_dataset()
    selected = selected_simulation_samples()
    scenario_map = {scenario.name: (category, scenario) for category, scenario in dataset["test"]}
    rows = len(selected)
    cols = 8
    fig, axes = plt.subplots(rows, cols, figsize=(21.0, 3.45 * rows), dpi=170)
    col_titles = [
        "代表帧",
        "真值高度",
        "原始DFF",
        "原算法+后处理",
        "Lee2013窗口",
        "Li2019迭代",
        "Focus-ResUNet",
        "误差摘要",
    ]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, pad=8)

    for r, name in enumerate(selected):
        category, scenario = scenario_map[name]
        arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
        stack = arrays["stack"]
        truth = arrays["truth"]
        assert isinstance(stack, np.ndarray)
        assert isinstance(truth, np.ndarray)
        preds = {
            "Original DFF": dff_original(stack),
            "Original DFF + post": original_full_postprocess(dff_original(stack)),
            "Lee2013 adaptive window": lee_adaptive_window_sff(stack),
            "Li2019 adaptive iteration": li_adaptive_iteration_sff(stack),
        }
        display = [
            (stack[stack.shape[0] // 2], "gray"),
            (truth, "viridis"),
            (preds["Original DFF"], "viridis"),
            (preds["Original DFF + post"], "viridis"),
            (preds["Lee2013 adaptive window"], "viridis"),
            (preds["Li2019 adaptive iteration"], "viridis"),
            (focus_resunet_visual(name), None),
        ]
        for c, (img, cmap) in enumerate(display):
            ax = axes[r, c]
            if cmap is None:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap, vmin=0 if cmap == "viridis" else None, vmax=1 if cmap == "viridis" else None)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(category, fontsize=10)
        ax = axes[r, -1]
        ax.axis("off")
        text_lines = [
            f"Original: {method_value(df, name, 'Original DFF'):.1f} um",
            f"Post: {method_value(df, name, 'Original DFF + post'):.1f} um",
            f"Lee2013: {method_value(df, name, 'Lee2013 adaptive window'):.1f} um",
            f"Li2019: {method_value(df, name, 'Li2019 adaptive iteration'):.1f} um",
            f"Tiny: {method_value(df, name, 'TinyDepthNet'):.1f} um",
            f"Focus-ResUNet: {method_value(df, name, 'Focus-ResUNet'):.1f} um",
            f"Residual: {method_value(df, name, 'Residual Focus-ResUNet'):.1f} um",
        ]
        ax.text(0.02, 0.95, "\n".join(text_lines), va="top", ha="left", fontsize=9, family="monospace")
    fig.suptitle("仿真样本多算法可视化对比：柏林粗糙、分形粗糙与周期纹理程序生成表面", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "simulation_multisample_algorithm_panel.png", bbox_inches="tight")
    plt.close(fig)


def rgb_height_proxy(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return normalize01(gray)


def plot_side_surface(ax: plt.Axes, depth: np.ndarray, title: str) -> None:
    z = cv2.resize(np.clip(depth, 0, 1).astype(np.float32), (150, 95), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]
    ax.plot_surface(xx, yy, z, cmap="viridis", linewidth=0, antialiased=True)
    ax.view_init(elev=18, azim=-82)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def save_simulation_3d_side_panel() -> None:
    dataset = build_dataset()
    selected = selected_simulation_samples()
    scenario_map = {scenario.name: (category, scenario) for category, scenario in dataset["test"]}
    methods = ["真值", "原始DFF", "原算法+后处理", "Lee2013", "Li2019", "Focus-ResUNet"]
    fig = plt.figure(figsize=(18.5, 2.75 * len(selected)), dpi=150)
    for r, name in enumerate(selected):
        category, scenario = scenario_map[name]
        arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
        stack = arrays["stack"]
        truth = arrays["truth"]
        assert isinstance(stack, np.ndarray)
        assert isinstance(truth, np.ndarray)
        original = dff_original(stack)
        surfaces = [
            truth,
            original,
            original_full_postprocess(original),
            lee_adaptive_window_sff(stack),
            li_adaptive_iteration_sff(stack),
            rgb_height_proxy(focus_resunet_visual(name)),
        ]
        for c, (title, depth) in enumerate(zip(methods, surfaces), start=1):
            ax = fig.add_subplot(len(selected), len(methods), r * len(methods) + c, projection="3d")
            plot_side_surface(ax, depth, title if r == 0 else "")
            if c == 1:
                ax.set_ylabel(category, fontsize=8, labelpad=0)
    fig.suptitle("仿真样本多算法 3D 侧视/斜视对比", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "simulation_multisample_3d_side_panel.png", bbox_inches="tight")
    plt.close(fig)


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_real_panel() -> None:
    selected = ["磕碰孔5um", "钥匙纹路100um", "钥匙尖头50um", "圆孔50um"]
    metrics = pd.read_csv(REAL_ROOT / "real_focus_formula_metrics.csv")
    fig, axes = plt.subplots(len(selected), 4, figsize=(14.5, 2.75 * len(selected)), dpi=170)
    col_titles = ["代表帧", "相对高度图", "聚焦置信度", "说明"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, pad=8)
    for r, sample in enumerate(selected):
        sample_dir = REAL_ROOT / sample
        frame = read_image(sample_dir / "01_representative_frame.png")
        depth = np.load(sample_dir / "relative_dff_depth.npy")
        conf = np.load(sample_dir / "focus_confidence.npy")
        row = metrics[metrics["sample"] == sample].iloc[0]
        items = [
            (resize_for_panel(frame), None),
            (resize_for_panel(depth), "viridis"),
            (resize_for_panel(conf), "magma"),
        ]
        for c, (img, cmap) in enumerate(items):
            ax = axes[r, c]
            if cmap is None:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(sample, fontsize=10)
        axes[r, 3].axis("off")
        text = (
            f"frames: {int(row['image_count'])}\n"
            f"size: {int(row['width'])}x{int(row['height'])}\n"
            f"mean conf: {float(row['mean_confidence']):.3f}\n"
            f"high-conf area: {float(row['high_conf_area_percent']):.1f}%\n"
            f"median best layer: {float(row['median_best_layer_1based']):.1f}"
        )
        axes[r, 3].text(0.02, 0.92, text, va="top", ha="left", fontsize=9, family="monospace")
    fig.suptitle("中期实物样本多样本可视化：磕碰孔与钥匙纹路为主的相对高度/置信度展示", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "real_midterm_multisample_panel.png", bbox_inches="tight")
    plt.close(fig)


def save_real_3d_side_overview() -> None:
    selected = [
        ("03_磕碰孔5um", "磕碰孔5um"),
        ("04_钥匙纹路100um", "钥匙纹路100um"),
        ("06_钥匙尖头50um", "钥匙尖头50um"),
        ("07_1124", "1124"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 9.2), dpi=150)
    for ax, (folder, title) in zip(axes.ravel(), selected):
        panel_path = REAL_ALL_ALG_ROOT / folder / "all_algorithm_3d_surface_panel.png"
        img = read_image(panel_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle("中期实物样本全算法 3D 重建侧视/斜视总览", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "real_midterm_all_algorithm_3d_side_overview.png", bbox_inches="tight")
    plt.close(fig)


def write_notes() -> None:
    notes = [
        "# 多样本可视化对比说明",
        "",
        "## 图件",
        "",
        "- `simulation_multisample_algorithm_panel.png`：选取 A型刃脊-柏林粗糙、山脊-柏林粗糙、阶跃-柏林粗糙、山峰-分形粗糙、周期-条纹粗糙五个仿真测试样本，突出柏林噪声、分形粗糙和周期纹理等程序生成表面的代表性。图中展示代表帧、真值高度、原始 DFF、原算法后处理、Lee2013 自适应窗口、Li2019 自适应迭代、Focus-ResUNet，并在右侧列出训练模型与传统方法的 MAE。",
        "- `simulation_multisample_3d_side_panel.png`：同一组仿真样本的 3D 侧视/斜视对比，用于观察不同算法在表面起伏、边缘和整体形貌上的差异。",
        "- `real_midterm_multisample_panel.png`：清洗后保留磕碰孔5um、钥匙纹路100um、钥匙尖头50um、圆孔50um 四个中期实物样本，展示代表帧、相对高度图、聚焦置信度和基本统计量；其中论文叙述重点放在磕碰孔和钥匙纹路两类真实形貌上。",
        "- `real_midterm_all_algorithm_3d_side_overview.png`：从中期实物全算法对比结果中汇总磕碰孔、钥匙纹路、钥匙尖头和 1124 的 3D 重建侧视/斜视图，便于直接放入论文图件包。",
        "",
        "## 论文使用口径",
        "",
        "仿真样本有高度真值，因此可以用于定量比较不同算法的 MAE、边缘误差和高风险区误差；实物样本没有标准高度真值，因此只用于展示算法流程在真实焦栈上的可运行性、相对三维形貌和聚焦置信度分布。两类图应配合使用：仿真图支撑定量结论，实物图支撑真实数据可视化与工程可复现性。",
        "",
        "建议在论文结论中明确：仿真侧重点展示柏林噪声、分形粗糙和周期纹理生成样本，用于说明算法在复杂程序表面上的误差修正能力；真实侧重点展示磕碰孔和钥匙纹路，用于说明方法对实际缺陷/纹理样本的可视化能力。由于真实样本缺少轮廓仪或标准台阶块标定，不宣称绝对高度精度。",
    ]
    (OUT / "多样本可视化对比说明.md").write_text("\n".join(notes), encoding="utf-8")
    manifest = {
        "simulation_panel": str(OUT / "simulation_multisample_algorithm_panel.png"),
        "simulation_3d_side_panel": str(OUT / "simulation_multisample_3d_side_panel.png"),
        "real_panel": str(OUT / "real_midterm_multisample_panel.png"),
        "real_3d_side_overview": str(OUT / "real_midterm_all_algorithm_3d_side_overview.png"),
        "notes": str(OUT / "多样本可视化对比说明.md"),
    }
    (OUT / "visual_comparison_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    save_simulation_panel()
    save_simulation_3d_side_panel()
    save_real_panel()
    save_real_3d_side_overview()
    write_notes()
    print(OUT / "simulation_multisample_algorithm_panel.png")
    print(OUT / "simulation_multisample_3d_side_panel.png")
    print(OUT / "real_midterm_multisample_panel.png")
    print(OUT / "real_midterm_all_algorithm_3d_side_overview.png")
    print(OUT / "多样本可视化对比说明.md")


if __name__ == "__main__":
    main()
