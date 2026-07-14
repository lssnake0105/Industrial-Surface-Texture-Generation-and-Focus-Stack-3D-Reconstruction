from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
VENDOR_PYOTF = OUT_DIR / "vendor_pyotf"
if str(VENDOR_PYOTF) not in sys.path:
    sys.path.insert(0, str(VENDOR_PYOTF))

WAVELENGTH_NM = 525.0
WAVELENGTH_UM = WAVELENGTH_NM / 1000.0
NA = 0.40
NI = 1.0
PIXEL_SIZE_NM = 250.0
PSF_SIZE = 128
Z_STEP_NM = 200.0
Z_SIZE = 161

CURRENT_STACK_LAYERS = 17
CURRENT_FOCUS_RANGE_UM = 2.0 * (2.0 * WAVELENGTH_UM / (NA * NA))
CURRENT_FOCUS_POSITIONS_UM = np.linspace(
    -CURRENT_FOCUS_RANGE_UM / 2,
    CURRENT_FOCUS_RANGE_UM / 2,
    CURRENT_STACK_LAYERS,
).astype(np.float32)


def folder_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024 / 1024


def estimate_storage() -> dict[str, float]:
    psf_stack_mb = Z_SIZE * PSF_SIZE * PSF_SIZE * 8 / 1024 / 1024
    return {
        "pyotf_vendor_mb": folder_size_mb(VENDOR_PYOTF) if VENDOR_PYOTF.exists() else 0.0,
        "single_float64_psf_stack_mb": psf_stack_mb,
        "expected_outputs_mb": 8.0,
        "psf_generator_dry_run_download_mb_at_least": 214.0,
    }


def run_pyotf_psf() -> tuple[pd.DataFrame, dict[str, object], Path]:
    from pyotf.otf import HanserPSF, SheppardPSF

    rows = []
    model_outputs = {}
    for model_name, cls in [("hanser", HanserPSF), ("sheppard", SheppardPSF)]:
        psf_model = cls(
            wl=WAVELENGTH_NM,
            na=NA,
            ni=NI,
            res=PIXEL_SIZE_NM,
            size=PSF_SIZE,
            zres=Z_STEP_NM,
            zsize=Z_SIZE,
            vec_corr="none",
            condition="sine",
        )
        zrange = np.asarray(
            getattr(psf_model, "zrange", (np.arange(Z_SIZE) - Z_SIZE // 2) * Z_STEP_NM),
            dtype=np.float64,
        )
        psf = psf_model.PSFi.astype(np.float64)
        psf_sum = float(psf.sum())
        psf /= psf_sum
        center = PSF_SIZE // 2
        axial = psf[:, center, center].copy()
        axial /= axial.max()
        lateral_profiles = psf[Z_SIZE // 2, center, :].copy()
        lateral_profiles /= lateral_profiles.max()

        half_mask = axial >= 0.5
        if half_mask.any():
            fwhm_z_um = (zrange[half_mask][-1] - zrange[half_mask][0]) / 1000.0
        else:
            fwhm_z_um = float("nan")

        rows.append(
            {
                "route": f"pyotf_{model_name}",
                "status": "ok",
                "wavelength_nm": WAVELENGTH_NM,
                "NA": NA,
                "ni": NI,
                "pixel_size_nm": PIXEL_SIZE_NM,
                "z_step_nm": Z_STEP_NM,
                "z_range_um": (zrange[-1] - zrange[0]) / 1000.0,
                "axial_fwhm_um": fwhm_z_um,
                "energy_normalized_sum": float(psf.sum()),
                "center_peak": float(psf[Z_SIZE // 2, center, center]),
                "raw_psf_sum_before_normalization": psf_sum,
            }
        )
        model_outputs[model_name] = {
            "psf": psf,
            "axial": axial,
            "lateral": lateral_profiles,
            "zrange": zrange,
        }

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.4), dpi=170)
    for col, model_name in enumerate(["hanser", "sheppard"]):
        psf = model_outputs[model_name]["psf"]
        axial = model_outputs[model_name]["axial"]
        lateral = model_outputs[model_name]["lateral"]
        zrange = model_outputs[model_name]["zrange"]
        center = PSF_SIZE // 2
        im0 = axes[0, col].imshow(
            psf[Z_SIZE // 2],
            cmap="magma",
            extent=[
                -PSF_SIZE * PIXEL_SIZE_NM / 2000,
                PSF_SIZE * PIXEL_SIZE_NM / 2000,
                PSF_SIZE * PIXEL_SIZE_NM / 2000,
                -PSF_SIZE * PIXEL_SIZE_NM / 2000,
            ],
        )
        axes[0, col].set_title(f"{model_name}: focal-plane PSF")
        axes[0, col].set_xlabel("x (um)")
        axes[0, col].set_ylabel("y (um)")
        fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.03)

        axes[1, col].plot(zrange / 1000.0, axial, label="axial center intensity")
        for zf in CURRENT_FOCUS_POSITIONS_UM:
            axes[1, col].axvline(float(zf), color="tab:gray", alpha=0.25, linewidth=0.8)
        axes[1, col].set_title(f"{model_name}: axial response")
        axes[1, col].set_xlabel("defocus z (um)")
        axes[1, col].set_ylabel("normalized intensity")
        axes[1, col].grid(alpha=0.25)
        axes[1, col].legend(fontsize=8)

    axes[0, 2].plot(
        np.linspace(-PSF_SIZE / 2, PSF_SIZE / 2, PSF_SIZE) * PIXEL_SIZE_NM / 1000.0,
        model_outputs["hanser"]["lateral"],
        label="Hanser",
    )
    axes[0, 2].plot(
        np.linspace(-PSF_SIZE / 2, PSF_SIZE / 2, PSF_SIZE) * PIXEL_SIZE_NM / 1000.0,
        model_outputs["sheppard"]["lateral"],
        label="Sheppard",
    )
    axes[0, 2].set_title("Focal lateral PSF profile")
    axes[0, 2].set_xlabel("x (um)")
    axes[0, 2].set_ylabel("normalized intensity")
    axes[0, 2].grid(alpha=0.25)
    axes[0, 2].legend()

    axes[1, 2].axis("off")
    text = (
        f"lambda/NA^2 = {WAVELENGTH_UM / (NA * NA):.2f} um\n"
        f"2lambda/NA^2 = {2 * WAVELENGTH_UM / (NA * NA):.2f} um\n"
        f"current focus range = +/-{CURRENT_FOCUS_RANGE_UM / 2:.2f} um\n"
        f"current focus step = {CURRENT_FOCUS_RANGE_UM / (CURRENT_STACK_LAYERS - 1):.2f} um\n"
        "vertical gray lines: current simulation focus planes"
    )
    axes[1, 2].text(0.03, 0.95, text, va="top", ha="left", fontsize=10, family="monospace")

    fig.subplots_adjust(wspace=0.34, hspace=0.34)
    panel_path = OUT_DIR / "pyotf_psf_focus_depth_panel.png"
    fig.savefig(panel_path)
    plt.close(fig)

    meta = {
        "pyotf_available": True,
        "models": ["HanserPSF", "SheppardPSF"],
        "zrange_nm": np.asarray(next(iter(model_outputs.values()))["zrange"], dtype=float).tolist(),
        "current_focus_positions_um": CURRENT_FOCUS_POSITIONS_UM.astype(float).tolist(),
    }
    return pd.DataFrame(rows), meta, panel_path


def write_report(metrics: pd.DataFrame, meta: dict[str, object], storage: dict[str, float], panel_path: Path) -> Path:
    table = metrics.to_markdown(index=False, floatfmt=".4f")
    report = f"""# 显微 PSF 与金属反射开源路线专项探测

## 目的

本轮延续上一轮路线对比，重点探测两个专项：显微 PSF/焦深校准路线，以及金属反射/粗糙散射路线。固定成像条件仍围绕 `lambda=525 nm`、`NA=0.40` 和当前焦栈范围设置。

## 依赖与存储

- `pyOTF`：已用 `--no-deps` 安装到本任务 `vendor_pyotf/`，再补充小依赖 `dphtools`。
- `pySCATMECH`：Windows 当前环境缺少 Microsoft Visual C++ 14.0+，源码扩展编译失败。
- `psf-generator`：dry-run 显示会拉取新版 torch、numpy、scipy、scikit-image 等依赖，下载体积至少约 214 MB，本轮未安装。
- 当前 `vendor_pyotf` 大小：{storage['pyotf_vendor_mb']:.2f} MB。
- 单个 float64 3D PSF 栈估计：{storage['single_float64_psf_stack_mb']:.2f} MB。

## pyOTF PSF 校准结果

本轮使用 `HanserPSF` 与 `SheppardPSF` 生成宽场显微 PSF。参数为：`wl=525 nm`，`NA=0.40`，`ni=1.0`，横向采样 `250 nm`，轴向采样 `200 nm`，轴向范围约 `32 um`。pyOTF 要求当前 NA/波长下的轴向采样小于约 `262.5 nm`，所以本轮没有使用更粗的 `500 nm` 采样。

指标表：

{table}

图片说明：

- 左上/中上：Hanser 与 Sheppard 模型在焦平面的横向 PSF。
- 左下/中下：轴向中心强度曲线；灰色竖线是当前轻量波动光学仿真使用的 17 个焦平面。
- 右上：两个模型的焦平面横向强度剖面。
- 右下：当前焦栈范围和 `lambda/NA^2`、`2lambda/NA^2` 的尺度关系。

横向 PSF 在上排看起来接近单点，是因为 `NA=0.40`、`lambda=525 nm` 下焦平面主瓣宽度接近微米级，而图中横向视野显示约 `32 um`；主要信息应从右上剖面和下排轴向曲线读取。

![pyOTF PSF panel]({panel_path.relative_to(OUT_DIR).as_posix()})

## 判断

`pyOTF` 是本轮最可用的显微 PSF/焦深校准工具，依赖体积小，能直接给出轴向响应曲线。它适合作为当前自写传播核的焦深尺度校验，而不适合直接替代反射表面复场传播。

`pySCATMECH` 物理方向正确，但当前 Windows 环境需要 C++ Build Tools，短期不适合作为快速实验主线。若后续要提高金属反射可信度，可以单独配置编译环境，先用它生成角度相关 Fresnel/BRDF 参数，再接入主传播脚本。

`psf-generator` 可能适合深度学习/PyTorch PSF 生成，但隔离安装成本高。本项目当前已经有自写 Torch FFT 路线，短期优先级低于 `pyOTF` 和 `pySCATMECH`。

## 推荐路线更新

1. 主传播与数据生成：继续使用 NumPy/Torch 自控传播核。
2. 显微焦深校准：加入 `pyOTF` 输出的轴向 PSF 曲线，用于解释焦栈范围和层间距。
3. 金属反射物理：暂时保留 Fresnel + roughness coherence 自写模型；后续在可编译环境中专项尝试 `pySCATMECH`。
4. 全视场子波源进入 NA 的完整性：下一步应在主传播脚本中显式记录 pupil 接收、离轴传播和每点贡献假设，避免只用局部镜面 acceptance 代替全场子波传播。
"""
    path = OUT_DIR / "psf_metal_route_probe_report.md"
    path.write_text(report, encoding="utf-8", newline="\n")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    storage = estimate_storage()
    metrics, meta, panel_path = run_pyotf_psf()
    metrics_path = OUT_DIR / "pyotf_psf_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    (OUT_DIR / "pyotf_psf_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "storage_and_dependency_probe.json").write_text(json.dumps(storage, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(metrics, meta, storage, panel_path)
    print(report)
    print(metrics_path)
    print(f"folder_size_mb={folder_size_mb(OUT_DIR):.2f}")


if __name__ == "__main__":
    main()
