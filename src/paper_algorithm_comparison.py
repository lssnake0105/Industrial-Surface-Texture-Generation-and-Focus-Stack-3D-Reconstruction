from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from final_dataset_training import build_dataset
from real_sample_glare_prior_fusion_validation import original_full_postprocess, postprocess_depth
from simulate_antiglare_highres_samples import (
    DEFAULT_STACK_LAYERS,
    generate_sample_arrays,
    metrics,
)


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "结题交付包" / "05_图表与结果"
OUT = RESULT_ROOT / "论文算法对比与结论收束"

TINY_METRICS = RESULT_ROOT / "最终仿真数据集训练验证" / "final_metrics.csv"
FOCUS_METRICS = RESULT_ROOT / "模型与损失函数升级实验" / "focus_resunet_metrics.csv"
RESIDUAL_METRICS = RESULT_ROOT / "模型与损失函数升级实验_残差保护版" / "residual_focus_resunet_metrics.csv"


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def focus_volume(stack: np.ndarray, method: str = "laplacian", window: int = 7) -> np.ndarray:
    maps: list[np.ndarray] = []
    for layer in stack:
        u8 = np.clip(layer * 255.0, 0, 255).astype(np.uint8)
        blur = cv2.GaussianBlur(u8, (3, 3), 0)
        if method == "glv":
            mean = cv2.boxFilter(blur.astype(np.float32), -1, (window, window), normalize=True)
            mean2 = cv2.boxFilter((blur.astype(np.float32) ** 2), -1, (window, window), normalize=True)
            fm = np.maximum(mean2 - mean * mean, 0.0)
        elif method == "sml":
            f = blur.astype(np.float32)
            dx = np.abs(2.0 * f - np.roll(f, 1, axis=1) - np.roll(f, -1, axis=1))
            dy = np.abs(2.0 * f - np.roll(f, 1, axis=0) - np.roll(f, -1, axis=0))
            fm = cv2.boxFilter(dx + dy, -1, (window, window), normalize=True)
        else:
            lap = np.abs(cv2.Laplacian(blur, cv2.CV_32F, ksize=3))
            sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
            tenengrad = sx * sx + sy * sy
            fm = cv2.boxFilter(lap, -1, (window, window), normalize=True) + 0.0018 * cv2.boxFilter(
                tenengrad, -1, (window, window), normalize=True
            )
        maps.append(fm.astype(np.float32))
    return np.stack(maps, axis=0)


def depth_from_focus(focus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argmax(focus, axis=0).astype(np.float32)
    denom = max(focus.shape[0] - 1, 1)
    depth = 1.0 - idx / float(denom)
    sorted_focus = np.sort(focus, axis=0)
    peak = sorted_focus[-1]
    second = sorted_focus[-2] if focus.shape[0] > 1 else np.zeros_like(peak)
    conf = (peak - second) / (peak + 1e-6)
    conf = np.clip(conf / (np.percentile(conf, 98.5) + 1e-6), 0, 1)
    return depth.astype(np.float32), conf.astype(np.float32)


def dff_original(stack: np.ndarray) -> np.ndarray:
    depth, _ = depth_from_focus(focus_volume(stack, method="laplacian", window=7))
    return depth


def lee_adaptive_window_sff(stack: np.ndarray) -> np.ndarray:
    # Lee et al. 2013: dynamic local window selection. This is a practical,
    # lightweight reproduction of the core idea for the current 17-layer stacks.
    gray_mid = np.clip(stack[stack.shape[0] // 2] * 255.0, 0, 255).astype(np.float32)
    local_std = cv2.GaussianBlur(gray_mid * gray_mid, (15, 15), 0) - cv2.GaussianBlur(gray_mid, (15, 15), 0) ** 2
    local_std = normalize01(np.sqrt(np.maximum(local_std, 0.0)))
    small = focus_volume(stack, method="glv", window=3)
    mid = focus_volume(stack, method="glv", window=7)
    large = focus_volume(stack, method="glv", window=13)
    focus = np.where(local_std[None] < 0.28, large, np.where(local_std[None] > 0.62, small, mid))
    focus = cv2.GaussianBlur(np.moveaxis(focus, 0, -1), (3, 3), 0)
    focus = np.moveaxis(focus, -1, 0)
    depth, _ = depth_from_focus(focus.astype(np.float32))
    return postprocess_depth(depth, median_kernel=3, gaussian_kernel=5, morph_kernel=3, order="median_gaussian_morph")


def li_adaptive_iteration_sff(stack: np.ndarray) -> np.ndarray:
    # Li et al. 2019: adaptive window plus iterative focus-value enhancement.
    f3 = focus_volume(stack, method="sml", window=3)
    f7 = focus_volume(stack, method="sml", window=7)
    f13 = focus_volume(stack, method="sml", window=13)
    contrast = normalize01(np.mean(np.abs(np.diff(stack, axis=0)), axis=0))
    focus = np.where(contrast[None] < 0.22, f13, np.where(contrast[None] > 0.55, f3, f7))
    for _ in range(2):
        enhanced = []
        for layer in focus:
            enhanced.append(cv2.boxFilter(layer, -1, (5, 5), normalize=True))
        focus = 0.55 * focus + 0.45 * np.stack(enhanced, axis=0)
    depth, _ = depth_from_focus(focus.astype(np.float32))
    return postprocess_depth(depth, median_kernel=5, gaussian_kernel=7, morph_kernel=5, order="median_gaussian_morph")


def method_metrics(name: str, pred: np.ndarray, truth: np.ndarray, risk: np.ndarray, depth_range_um: float) -> dict[str, float | str]:
    m = metrics(pred, truth, risk, depth_range_um)
    return {
        "method": name,
        "mae_um": m["mae_um"],
        "rmse_norm": m["rmse_norm"],
        "p90_um": m["p90_norm"] * depth_range_um,
        "high_risk_mae_um": m["high_risk_mae_um"],
        "edge_mae_um": m["edge_mae_um"],
    }


def read_csv_by_sample(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["sample"]: row for row in csv.DictReader(f) if row.get("split") == "test"}


def append_trained_model_rows(rows: list[dict[str, object]], test_meta: dict[str, dict[str, object]]) -> None:
    tiny = read_csv_by_sample(TINY_METRICS)
    focus = read_csv_by_sample(FOCUS_METRICS)
    residual = read_csv_by_sample(RESIDUAL_METRICS)
    specs = [
        ("TinyDepthNet", tiny, "model_mae_um", "model_high_risk_mae_um", "model_edge_mae_um"),
        ("Focus-ResUNet", focus, "focus_resunet_mae_um", "focus_resunet_high_risk_mae_um", "focus_resunet_edge_mae_um"),
        (
            "Residual Focus-ResUNet",
            residual,
            "residual_focus_resunet_mae_um",
            "residual_focus_resunet_high_risk_mae_um",
            "residual_focus_resunet_edge_mae_um",
        ),
    ]
    for method, table, mae_key, high_key, edge_key in specs:
        for sample, meta in test_meta.items():
            source = table[sample]
            rows.append(
                {
                    "split": "test",
                    "category": meta["category"],
                    "sample": sample,
                    "resolution": meta["resolution"],
                    "depth_range_um": meta["depth_range_um"],
                    "method": method,
                    "mae_um": float(source[mae_key]),
                    "rmse_norm": float("nan"),
                    "p90_um": float("nan"),
                    "high_risk_mae_um": float(source[high_key]),
                    "edge_mae_um": float(source[edge_key]),
                }
            )


def save_overview_plot(rows: list[dict[str, object]]) -> None:
    test_rows = rows
    methods = [
        "Original DFF",
        "Original DFF + post",
        "Lee2013 adaptive window",
        "Li2019 adaptive iteration",
        "GADFF",
        "TinyDepthNet",
        "Focus-ResUNet",
        "Residual Focus-ResUNet",
    ]
    means = []
    for method in methods:
        vals = [float(r["mae_um"]) for r in test_rows if r["method"] == method]
        means.append(float(np.mean(vals)))
    plt.figure(figsize=(10.8, 4.8), dpi=160)
    colors = ["#7b8794", "#5c677d", "#d58936", "#c45b32", "#3a8f7b", "#4c78a8", "#2f6fba", "#6f5aa8"]
    plt.bar(methods, means, color=colors)
    plt.ylabel("Mean MAE / um")
    plt.title("Paper comparison on the same synthetic DFF test split")
    plt.xticks(rotation=28, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "paper_comparison_mean_mae.png")
    plt.close()


def write_report(rows: list[dict[str, object]]) -> None:
    methods = sorted({str(r["method"]) for r in rows})
    means = {
        method: {
            "mae": float(np.mean([float(r["mae_um"]) for r in rows if r["method"] == method])),
            "high": float(np.mean([float(r["high_risk_mae_um"]) for r in rows if r["method"] == method])),
            "edge": float(np.mean([float(r["edge_mae_um"]) for r in rows if r["method"] == method])),
        }
        for method in methods
    }
    best = min(means.items(), key=lambda item: item[1]["mae"])
    p10_rows = [r for r in rows if "P10" in str(r["sample"])]

    lines = [
        "# 论文算法对比与结论收束",
        "",
        "## 对比口径",
        "",
        "本实验只把能够在当前仿真焦栈上公平运行的算法纳入数值对比。Lee 2013 与 Li 2019 按其核心思想实现为自适应窗口/迭代增强传统 SFF 基线；Yang 2022 的 Differential Focus Volume 不直接复现原论文完整 3D CNN，而作为本项目 Focus-ResUNet 输入设计依据：即使用 17 层焦栈、16 层焦向差分和 DFF/眩光先验通道。",
        "",
        "## 平均指标",
        "",
        "| 方法 | 平均 MAE/um | 高风险区 MAE/um | 边缘区 MAE/um | 论文定位 |",
        "|---|---:|---:|---:|---|",
    ]
    descriptions = {
        "Original DFF": "原始 Laplacian/Tenengrad 聚焦评价基线",
        "Original DFF + post": "中期工程算法：原始 DFF 加后处理",
        "Lee2013 adaptive window": "文献传统方法：自适应窗口 SFF",
        "Li2019 adaptive iteration": "文献传统方法：自适应窗口迭代增强",
        "GADFF": "本项目眩光降权传统改进",
        "TinyDepthNet": "初始学习型校正基线",
        "Focus-ResUNet": "当前推荐学习型模型",
        "Residual Focus-ResUNet": "保守残差保护模型",
    }
    preferred_order = [
        "Original DFF",
        "Original DFF + post",
        "Lee2013 adaptive window",
        "Li2019 adaptive iteration",
        "GADFF",
        "TinyDepthNet",
        "Focus-ResUNet",
        "Residual Focus-ResUNet",
    ]
    for method in preferred_order:
        item = means[method]
        lines.append(
            f"| {method} | {item['mae']:.2f} | {item['high']:.2f} | {item['edge']:.2f} | {descriptions[method]} |"
        )
    lines += [
        "",
        "## P10 主案例",
        "",
        "| 方法 | P10 MAE/um | P10 高风险区 MAE/um | P10 边缘区 MAE/um |",
        "|---|---:|---:|---:|",
    ]
    for method in preferred_order:
        row = next(r for r in p10_rows if r["method"] == method)
        lines.append(
            f"| {method} | {float(row['mae_um']):.2f} | {float(row['high_risk_mae_um']):.2f} | {float(row['edge_mae_um']):.2f} |"
        )
    lines += [
        "",
        "## 可写入论文的结论",
        "",
        f"在当前 7 个仿真测试样本上，平均 MAE 最低的方法是 `{best[0]}`，平均 MAE 为 `{best[1]['mae']:.2f} um`。传统 DFF 与文献自适应窗口类方法能够形成稳定基线，但在 P10 V 谷、周期纹理等复杂表面上仍容易受到低纹理、多峰聚焦响应和眩光/杂散光影响。Focus-ResUNet 借鉴 DFV 的焦向差分思想，并加入 DFF/GADFF 先验，在复杂纹理和强眩光样本上更有优势；残差保护版在部分 DFF 已可靠样本上更保守，但会限制 P10 等困难样本的修正幅度。",
        "",
        "因此，最终论文建议收束为：传统 DFF 是可解释、可稳定复现的工程基线；文献中的自适应窗口方法改善了固定窗口带来的平滑/噪声矛盾；本项目的主要增量是把眩光风险、焦向差分和学习型校正结合起来。结论不应写成深度模型在所有样本上绝对优于传统算法，而应写成“在复杂纹理和强反射条件下，Focus-ResUNet 提供了更有效的误差校正方向；在凹坑/阶跃等局部结构上仍需要结构类型门控或更多真实标定数据”。",
        "",
        "## 参考文献对应关系",
        "",
        "- Lee et al., 2013: Adaptive window selection for 3D shape recovery from image focus。用于传统自适应窗口 SFF 对比。",
        "- Li et al., 2019: Adaptive window iteration algorithm for enhancing 3D shape recovery from image focus。用于传统自适应窗口迭代增强对比。",
        "- Yang et al., 2022: Deep Depth from Focus with Differential Focus Volume。用于解释焦向差分特征和 Focus-ResUNet 设计来源。",
        "- 国内综述与变焦显微测量文献用于引言、系统背景和传统 DFF 评价，不作为本实验的数值复现对象。",
    ]
    (OUT / "论文算法对比与结论收束.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    rows: list[dict[str, object]] = []
    test_meta: dict[str, dict[str, object]] = {}
    for category, scenario in dataset["test"]:
        print(f"evaluate test: {scenario.name}", flush=True)
        arrays = generate_sample_arrays(scenario, stack_layers=DEFAULT_STACK_LAYERS)
        stack = arrays["stack"]
        truth = arrays["truth"]
        risk = arrays["risk"]
        dff = arrays["dff"]
        gadff = arrays["gadff"]
        features = arrays["features"]
        assert isinstance(stack, np.ndarray)
        assert isinstance(truth, np.ndarray)
        assert isinstance(risk, np.ndarray)
        assert isinstance(dff, np.ndarray)
        assert isinstance(gadff, np.ndarray)
        test_meta[scenario.name] = {
            "category": category,
            "resolution": f"{scenario.width}x{scenario.height}",
            "depth_range_um": scenario.depth_range_um,
        }

        predictions = {
            "Original DFF": dff_original(stack),
            "Original DFF + post": original_full_postprocess(dff_original(stack)),
            "Lee2013 adaptive window": lee_adaptive_window_sff(stack),
            "Li2019 adaptive iteration": li_adaptive_iteration_sff(stack),
            "GADFF": gadff,
        }
        for name, pred in predictions.items():
            row = {
                "split": "test",
                "category": category,
                "sample": scenario.name,
                "resolution": f"{scenario.width}x{scenario.height}",
                "depth_range_um": scenario.depth_range_um,
                **method_metrics(name, pred, truth, risk, scenario.depth_range_um),
            }
            rows.append(row)
    append_trained_model_rows(rows, test_meta)

    csv_path = OUT / "paper_algorithm_comparison_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "paper_algorithm_comparison_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    save_overview_plot(rows)
    write_report(rows)
    print(csv_path, flush=True)
    print(OUT / "论文算法对比与结论收束.md", flush=True)


if __name__ == "__main__":
    main()
