from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.special import j1


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = (
    ROOT
    / "submission_planning"
    / "optical_mechanism_analysis"
    / "diffraction_interference_validation_2026-06-26"
)

WAVELENGTH_UM = 0.525
MAGNIFICATION = 20.0
NA_VALUES = [0.30, 0.40, 0.45, 0.50, 0.60]
PRIMARY_NA_VALUES = [0.40, 0.45]


def airy_psf(x_um: np.ndarray, y_um: np.ndarray, na: float, wavelength_um: float) -> np.ndarray:
    r = np.hypot(x_um, y_um)
    u = 2.0 * np.pi * na * r / wavelength_um
    psf = np.ones_like(u)
    mask = np.abs(u) > 1e-12
    psf[mask] = (2.0 * j1(u[mask]) / u[mask]) ** 2
    return np.maximum(psf, 0.0)


def diffraction_metrics() -> pd.DataFrame:
    rows = []
    for na in NA_VALUES:
        rayleigh_um = 0.61 * WAVELENGTH_UM / na
        abbe_um = WAVELENGTH_UM / (2.0 * na)
        rows.append(
            {
                "NA": na,
                "rayleigh_lateral_um": rayleigh_um,
                "abbe_lateral_um": abbe_um,
                "airy_disk_diameter_um": 1.22 * WAVELENGTH_UM / na,
                "airy_fwhm_um": 0.514 * WAVELENGTH_UM / na,
                "incoherent_cutoff_cycles_per_um": 2.0 * na / WAVELENGTH_UM,
                "wave_dof_term_um": WAVELENGTH_UM / (na**2),
                "axial_scale_2lambda_over_na2_um": 2.0 * WAVELENGTH_UM / (na**2),
                "max_sensor_pitch_for_rayleigh_nyquist_um": MAGNIFICATION * rayleigh_um / 2.0,
                "max_sensor_pitch_for_abbe_nyquist_um": MAGNIFICATION * abbe_um / 2.0,
            }
        )
    return pd.DataFrame(rows)


def save_resolution_plot(metrics: pd.DataFrame) -> Path:
    na_grid = np.linspace(0.20, 0.80, 300)
    rayleigh = 0.61 * WAVELENGTH_UM / na_grid
    abbe = WAVELENGTH_UM / (2.0 * na_grid)
    airy_diameter = 1.22 * WAVELENGTH_UM / na_grid

    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=180)
    ax.plot(na_grid, rayleigh, label="Rayleigh lateral limit: 0.61 lambda / NA", lw=2.2)
    ax.plot(na_grid, abbe, label="Abbe criterion: lambda / (2 NA)", lw=2.2)
    ax.plot(na_grid, airy_diameter, label="Airy disk diameter: 1.22 lambda / NA", lw=2.2)
    for na in PRIMARY_NA_VALUES:
        ax.axvline(na, color="0.25", ls="--", lw=1.1)
        row = metrics.loc[np.isclose(metrics["NA"], na)].iloc[0]
        ax.scatter([na], [row["rayleigh_lateral_um"]], s=42, zorder=5)
        ax.text(
            na + 0.008,
            row["rayleigh_lateral_um"] + 0.035,
            f"NA={na:.2f}\nRayleigh={row['rayleigh_lateral_um']:.2f} um",
            fontsize=8.5,
            va="bottom",
        )
    ax.set_title("Diffraction-limited resolution estimate at lambda = 525 nm")
    ax.set_xlabel("Objective numerical aperture (NA)")
    ax.set_ylabel("Object-side length scale (um)")
    ax.set_xlim(0.20, 0.80)
    ax.set_ylim(0.0, 3.4)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    path = OUT_DIR / "resolution_vs_na_525nm.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_psf_interference_plot(na: float = 0.40) -> tuple[Path, dict[str, float]]:
    n = 768
    extent_um = 12.0
    axis = np.linspace(-extent_um / 2.0, extent_um / 2.0, n)
    x, y = np.meshgrid(axis, axis)
    r = np.hypot(x, y)

    ideal = airy_psf(x, y, na, WAVELENGTH_UM)
    ideal_peak = ideal / ideal.max()
    ideal_kernel = ideal / ideal.sum()

    # Weak coherent ghost field: a small parasitic reflection with a broad envelope.
    ghost_amp = 0.12 * np.exp(-(r / 4.2) ** 2)
    fringe_period_um = 4.0
    ghost_phase = 2.0 * np.pi * x / fringe_period_um + 0.7
    coherent = np.abs(np.sqrt(ideal_peak) + ghost_amp * np.exp(1j * ghost_phase)) ** 2
    coherent /= coherent.max()

    object_img = np.full_like(x, 0.18)
    bands = [
        (-5.6, -3.2, 0.55),
        (-3.0, -0.6, 0.70),
        (-0.4, 2.0, 0.95),
        (2.2, 5.4, 1.35),
    ]
    for y0, y1, period in bands:
        mask = (y >= y0) & (y < y1)
        bars = (np.sin(2.0 * np.pi * (x + 0.1) / period) > 0).astype(float)
        object_img[mask] = 0.12 + 0.78 * bars[mask]
    pit = np.abs(np.hypot(x + 3.4, y - 2.4) - 0.95) < 0.065
    ridge = np.abs(y + 0.28 * x + 3.1) < 0.06
    object_img[pit | ridge] = 1.0

    diffraction_only = fftconvolve(object_img, ideal_kernel, mode="same")
    diffraction_only = np.clip(diffraction_only, 0.0, 1.0)

    field_radius = np.hypot(x / (extent_um / 2.0), y / (extent_um / 2.0))
    vignette = np.clip(1.0 - 0.18 * field_radius**2, 0.72, 1.0)
    partial_coherence_visibility = 0.10
    interference = 1.0 + partial_coherence_visibility * np.cos(
        2.0 * np.pi * x / fringe_period_um + 0.35
    ) * np.exp(-(r / 5.4) ** 2)
    halo_source = (object_img > 0.88).astype(float)
    halo = 0.18 * gaussian_filter(halo_source, sigma=16)
    halo /= max(float(halo.max()), 1e-12)
    halo *= 0.10
    diffraction_interference = np.clip(diffraction_only * vignette * interference + halo, 0.0, 1.0)

    diff = diffraction_interference - diffraction_only
    ring_radius_um = 0.61 * WAVELENGTH_UM / na
    ideal_halo_fraction = float(ideal_kernel[r > ring_radius_um].sum())
    coherent_halo_fraction = float((coherent / coherent.sum())[r > ring_radius_um].sum())
    diff_modulation = float(np.percentile(diff, 95) - np.percentile(diff, 5))

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), dpi=180)
    panels = [
        (object_img, "Synthetic reflective target", "gray", 0.0, 1.0),
        (np.log10(ideal_peak + 1e-6), "Ideal Airy PSF (log10)", "magma", -6.0, 0.0),
        (np.log10(coherent + 1e-6), "Airy PSF + weak coherent ghost (log10)", "magma", -6.0, 0.0),
        (diffraction_only, "Diffraction-limited image", "gray", 0.0, 1.0),
        (
            diffraction_interference,
            "Diffraction + weak interference + vignette",
            "gray",
            0.0,
            1.0,
        ),
        (diff, "Difference map", "coolwarm", -0.18, 0.18),
    ]
    for ax, (img, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(
            img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[-extent_um / 2, extent_um / 2, -extent_um / 2, extent_um / 2],
            origin="lower",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("um")
        ax.set_ylabel("um")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Scalar diffraction and weak partial-coherence halo simulation", y=0.995)
    fig.tight_layout()
    path = OUT_DIR / "psf_interference_halo_simulation_525nm_na040.png"
    fig.savefig(path)
    plt.close(fig)

    metrics = {
        "na": na,
        "wavelength_um": WAVELENGTH_UM,
        "first_zero_radius_um": ring_radius_um,
        "ideal_energy_outside_first_zero": ideal_halo_fraction,
        "weak_ghost_energy_outside_first_zero": coherent_halo_fraction,
        "synthetic_image_p5_p95_difference_modulation": diff_modulation,
        "ghost_amplitude_peak": 0.12,
        "partial_coherence_visibility": partial_coherence_visibility,
        "fringe_period_um": fringe_period_um,
    }
    return path, metrics


def write_report(
    metrics: pd.DataFrame,
    resolution_plot: Path,
    psf_plot: Path,
    sim_metrics: dict[str, float],
) -> Path:
    table = metrics.copy()
    table = table[
        [
            "NA",
            "rayleigh_lateral_um",
            "abbe_lateral_um",
            "airy_disk_diameter_um",
            "wave_dof_term_um",
            "max_sensor_pitch_for_rayleigh_nyquist_um",
        ]
    ]
    table_md = table.to_markdown(index=False, floatfmt=".3f")

    rel_resolution_plot = resolution_plot.relative_to(OUT_DIR).as_posix()
    rel_psf_plot = psf_plot.relative_to(OUT_DIR).as_posix()

    report = f"""# Olympus 20X + 525 nm 同轴绿光成像的衍射极限与干涉晕影验证

## 结论先行

在把“525mm LED 绿光”按 **525 nm LED 绿光**理解、且 Olympus 20X 物镜型号尚未完全确认的前提下，若物镜 NA 落在 Olympus/Evident 常见 20X 明场/反射物镜的 **0.40-0.45** 区间，则横向 Rayleigh 衍射极限为 **0.71-0.80 um**，Abbe 尺度为 **0.58-0.66 um**。如果相机像元尺寸不超过约 **7.1-8.0 um**，20X 放大后的物方采样满足 Rayleigh Nyquist 条件，系统分辨率主要受物镜 NA 和波长限制。因此，对于 1 um 量级纹理、刃边和微缺陷，当前成像链可以被合理表述为“接近衍射限制”。

干涉晕影部分应写成“物理上可解释且可由模拟复现”，不能只凭 20X、525 nm 和同轴照明三个参数断言真实图像中已经发生了波前干涉。525 nm LED 属于有限带宽、部分相干光源；在同轴照明、反光样本、保护玻璃/分光镜/物镜前表面存在弱寄生反射时，主波前与弱反射波前可形成低对比度调制。该调制叠加 Airy 旁瓣、场边缘照度下降和离焦高亮扩散后，会表现为环状/条纹状 halo、局部 vignette-like shading 或高亮边缘拖尾。

## 参数假设

- 波长：lambda = 525 nm = 0.525 um。
- 物镜：Olympus/Evident 20X 的完整 SKU 和铭牌 NA 尚未提供。本文按常见 20X 空气物镜的 NA=0.40-0.45 区间估算，并把 NA=0.40 和 NA=0.45 作为主分析点；拿到具体型号后只需替换 NA 即可重算。
- 介质：空气，n≈1。
- 采样：相机像元未知，因此给出满足 Nyquist 的最大传感器像元尺寸阈值。
- 光源：按中心波长 525 nm 的窄带 LED 估算；LED FWHM 未实测，报告只做 15-50 nm 量级的部分相干讨论。

## 1. 横向衍射极限推导

显微横向分辨率常用两个尺度：

$$
r_{{Rayleigh}} = \\frac{{0.61\\lambda}}{{NA}},
\\qquad
r_{{Abbe}} = \\frac{{\\lambda}}{{2NA}}.
$$

Airy 斑第一暗环直径为：

$$
d_{{Airy}} = \\frac{{1.22\\lambda}}{{NA}}.
$$

代入 lambda=0.525 um 得：

{table_md}

关键判断：

- NA=0.40 时，Rayleigh 横向极限约 0.801 um，Airy 斑第一暗环直径约 1.601 um。
- NA=0.45 时，Rayleigh 横向极限约 0.712 um，Airy 斑第一暗环直径约 1.423 um。
- 20X 系统若使用 3.45 um 像元，相当于物方 0.173 um/pixel；若使用 5.86 um 像元，相当于 0.293 um/pixel，均细于 Rayleigh Nyquist 阈值。因此多数工业相机配置下，限制项更可能来自物镜衍射/像差/照明，而非像元采样。

![Resolution versus NA]({rel_resolution_plot})

## 2. 轴向景深与焦栈影响

MicroscopyU 给出的显微景深近似式为：

$$
d_{{tot}} = \\frac{{\\lambda n}}{{NA^2}} + \\frac{{n}}{{M\\cdot NA}}e,
$$

其中第一项是波动光学项，第二项与探测器可分辨距离和放大率有关。对 lambda=0.525 um：

- NA=0.40 时，波动项约 3.28 um；若用更宽松的轴向尺度 2lambda/NA^2，则约 6.56 um。
- NA=0.45 时，波动项约 2.59 um；2lambda/NA^2 约 5.19 um。

这说明 20X、NA≈0.40-0.45 的焦栈成像对微米级 z 方向变化非常敏感；高亮边缘、弱纹理、局部饱和和离焦扩散都可能改变 focus measure 的峰值位置。

## 3. 波前干涉与晕影的可解释模型

把主成像场记作 E0，弱寄生反射或弱相干散射项记作 Eg，则传感器强度可写为：

$$
I = |E_0 + \\gamma E_g e^{{i\\Delta\\phi}}|^2
  = |E_0|^2 + \\gamma^2|E_g|^2
  + 2\\gamma |E_0||E_g|\\cos(\\Delta\\phi).
$$

相位差可近似写为：

$$
\\Delta\\phi(x,y) =
\\frac{{2\\pi}}{{\\lambda}}OPD(x,y)
+ k_x x + k_y y + \\phi_0.
$$

因此，只要反光样本、分光镜、保护玻璃或物镜表面引入小比例寄生场，即使 gamma 只有 0.05-0.15，也会在局部形成百分之几到十几的强度调制。LED 的有限带宽会降低相干可见度，但若光程差落在数微米到十余微米量级，部分相干项仍可能保留。

对两束小角度波前，条纹周期近似为：

$$
p \\approx \\frac{{\\lambda}}{{2\\sin(\\alpha/2)}}.
$$

当 alpha=2°、5°、10° 时，p 分别约为 15.0 um、6.0 um、3.0 um。这些尺度与显微图中的慢变晕影、环状 halo 或局部条纹调制处在同一数量级。

## 4. 模拟结果

模拟采用：

- 理想 Airy PSF：$[2J_1(u)/u]^2$，$u=2\\pi NA r/\\lambda$。
- 弱寄生相干项：峰值振幅比 0.12，条纹周期 4 um。
- 部分相干可见度：0.10。
- 场照度衰减：约 18% 的二次 vignetting-like roll-off。
- 高反射边缘 halo：对高亮结构做弱宽核扩散。

![PSF interference halo simulation]({rel_psf_plot})

模拟指标：

- Airy 第一暗环半径：{sim_metrics["first_zero_radius_um"]:.3f} um。
- 理想 Airy PSF 第一暗环外能量比例：{sim_metrics["ideal_energy_outside_first_zero"]:.3f}。
- 加入弱寄生场后的第一暗环外能量比例：{sim_metrics["weak_ghost_energy_outside_first_zero"]:.3f}。
- 合成图中弱干涉/晕影项引入的 p5-p95 强度调制幅度：{sim_metrics["synthetic_image_p5_p95_difference_modulation"]:.3f}。

这个结果支持一个更稳妥的表述：衍射旁瓣提供了 halo 的基础空间扩散，弱相干寄生反射提供了条纹/低频调制，场边缘照度下降和反光边缘扩散会让这种调制在真实图像中表现得像晕影或局部光晕。

## 5. 投稿中可使用的表述

建议写法：

> With a 525 nm coaxial green LED and a common Olympus/Evident 20X dry objective (NA≈0.40-0.45), the Rayleigh lateral diffraction limit is estimated to be 0.71-0.80 um. For typical industrial camera pixel pitches below 7-8 um, the object-side sampling after 20X magnification is finer than the Rayleigh Nyquist requirement, indicating that micrometer-scale reflective features are close to the optical diffraction-limited regime. Under coaxial reflective imaging, weak parasitic reflections and partial coherence can further modulate the Airy-limited image, producing halo-like or vignette-like artifacts around high-contrast reflective edges.

中文写法：

> 在 525 nm 同轴绿光和常见 Olympus/Evident 20X 干式物镜 NA≈0.40-0.45 的条件下，Rayleigh 横向衍射极限约为 0.71-0.80 um。若工业相机像元尺寸低于 7-8 um，20X 放大后的物方采样已经细于 Rayleigh Nyquist 要求，微米级反光纹理的可分辨性主要受物镜 NA、波长、像差和照明条件限制。在同轴反射成像中，弱寄生反射和部分相干性会调制 Airy 受限图像，使高对比反光边缘附近出现 halo-like 或 vignette-like artifacts。

## 6. 还需要补充的真实验证

1. 记录物镜完整型号与 NA、相机像元尺寸、管镜倍率、曝光时间、LED 带宽和照明孔径。
2. 用 USAF 1951、chrome-on-glass 分辨率板或亚分辨率荧光/反射微珠测量真实 PSF 和 MTF。
3. 拍摄平场反射样本或镜面样本，检查是否存在固定位置环状晕影、条纹和低频照度场。
4. 改变 LED 波长、光阑/NA、曝光、偏振或轻微倾斜样本。如果 halo 尺度随 lambda/NA 缩放，支持衍射解释；如果条纹周期随倾角或光程改变，支持干涉解释；如果只随视场位置变化，更多指向照明/光路 vignetting。

## 参考来源

- Nikon MicroscopyU, Resolution: https://www.microscopyu.com/microscopy-basics/resolution
- Nikon MicroscopyU, Depth of Field and Depth of Focus: https://www.microscopyu.com/microscopy-basics/depth-of-field-and-depth-of-focus
- Evident Scientific, Objective Finder: https://evidentscientific.com/en/objective-finder
- LightSource.tech, Monochromatic fiber-coupled LED light sources: https://www.lightsource.tech/en/fiber-coupled-light-sources/monochromatic-fiber-coupled-led/
"""

    path = OUT_DIR / "diffraction_interference_validation_report.md"
    path.write_text(report, encoding="utf-8", newline="\n")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = diffraction_metrics()
    metrics_path = OUT_DIR / "diffraction_resolution_metrics_525nm.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    resolution_plot = save_resolution_plot(metrics)
    psf_plot, sim_metrics = save_psf_interference_plot(na=0.40)

    sim_metrics_path = OUT_DIR / "interference_simulation_metrics_525nm_na040.json"
    sim_metrics_path.write_text(
        json.dumps(sim_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    report_path = write_report(metrics, resolution_plot, psf_plot, sim_metrics)
    print(report_path)
    print(metrics_path)
    print(resolution_plot)
    print(psf_plot)
    print(sim_metrics_path)


if __name__ == "__main__":
    main()
