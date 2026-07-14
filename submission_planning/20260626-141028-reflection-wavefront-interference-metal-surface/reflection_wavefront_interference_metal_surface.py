from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from surface_sample_generator import SurfaceConfig, generate_surface  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent

WAVELENGTH_UM = 0.525
NA = 0.40
SENSOR_PIXEL_UM = 3.45
MAGNIFICATION = 20.0
OBJECT_PIXEL_UM = SENSOR_PIXEL_UM / MAGNIFICATION
SENSOR_W = 960
SENSOR_H = 540
SUPER = 2
HIGH_W = SENSOR_W * SUPER
HIGH_H = SENSOR_H * SUPER
HIGH_PIXEL_UM = OBJECT_PIXEL_UM / SUPER
FOV_WIDTH_UM = SENSOR_W * OBJECT_PIXEL_UM
FOV_HEIGHT_UM = SENSOR_H * OBJECT_PIXEL_UM


@dataclass
class MetalOptics:
    name: str
    n: float
    k: float
    roughness_rms_nm: float
    parasitic_reflection_amp: float
    spatial_coherence: float
    diffuse_fraction: float


@dataclass
class ReflectionCase:
    case_id: str
    label: str
    surface: SurfaceConfig
    metal: MetalOptics


CASES = [
    ReflectionCase(
        "polished_a_ridge",
        "Polished metal A-ridge",
        SurfaceConfig(
            name="polished_a_ridge",
            width=HIGH_W,
            height=HIGH_H,
            depth_range_um=75.0,
            baseline_type="a_ridge",
            noise_type="perlin",
            seed=4101,
            feature_amp_um=68.0,
            noise_amp_um=0.55,
            ridge_width=0.082,
            perlin_octaves=6,
            perlin_grid=72,
            perlin_persistence=0.54,
            orientation_deg=-8.0,
        ),
        MetalOptics(
            name="generic polished steel-like metal",
            n=2.70,
            k=3.30,
            roughness_rms_nm=18.0,
            parasitic_reflection_amp=0.075,
            spatial_coherence=0.45,
            diffuse_fraction=0.08,
        ),
    ),
    ReflectionCase(
        "rough_v_valley",
        "Rough metal V-valley",
        SurfaceConfig(
            name="rough_v_valley",
            width=HIGH_W,
            height=HIGH_H,
            depth_range_um=88.0,
            baseline_type="v_valley",
            noise_type="perlin",
            seed=4102,
            feature_amp_um=82.0,
            noise_amp_um=1.35,
            valley_width=0.54,
            valley_floor=0.15,
            valley_sharpness=0.72,
            perlin_octaves=7,
            perlin_grid=86,
            perlin_persistence=0.60,
            orientation_deg=-15.0,
        ),
        MetalOptics(
            name="generic rough steel-like metal",
            n=2.70,
            k=3.30,
            roughness_rms_nm=62.0,
            parasitic_reflection_amp=0.055,
            spatial_coherence=0.26,
            diffuse_fraction=0.22,
        ),
    ),
    ReflectionCase(
        "micro_step_edge",
        "Metal step edge with micro texture",
        SurfaceConfig(
            name="micro_step_edge",
            width=HIGH_W,
            height=HIGH_H,
            depth_range_um=66.0,
            baseline_type="step",
            noise_type="perlin",
            seed=4103,
            feature_amp_um=60.0,
            noise_amp_um=0.95,
            step_position=-0.06,
            perlin_octaves=6,
            perlin_grid=64,
            perlin_persistence=0.56,
        ),
        MetalOptics(
            name="generic semi-polished metal",
            n=2.70,
            k=3.30,
            roughness_rms_nm=36.0,
            parasitic_reflection_amp=0.065,
            spatial_coherence=0.34,
            diffuse_fraction=0.14,
        ),
    ),
]


def normalize01(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    h, w = arr.shape[:2]
    trimmed = arr[: h - h % factor, : w - w % factor]
    if np.iscomplexobj(trimmed):
        shaped = trimmed.reshape(trimmed.shape[0] // factor, factor, trimmed.shape[1] // factor, factor)
    else:
        shaped = trimmed.reshape(trimmed.shape[0] // factor, factor, trimmed.shape[1] // factor, factor)
    return shaped.mean(axis=(1, 3))


def normals_from_height(z_um: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dzdy, dzdx = np.gradient(z_um.astype(np.float32), HIGH_PIXEL_UM, HIGH_PIXEL_UM)
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(z_um, dtype=np.float32)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    return nx / norm, ny / norm, nz / norm


def fresnel_unpolarized_complex(cos_theta: np.ndarray, n: float, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cos_theta = np.clip(cos_theta.astype(np.float64), 1e-4, 1.0)
    sin2 = 1.0 - cos_theta * cos_theta
    m = complex(n, k)
    q = np.sqrt(m * m - sin2)
    rs = (cos_theta - q) / (cos_theta + q)
    rp = (m * m * cos_theta - q) / (m * m * cos_theta + q)
    r_field = 0.5 * (rs + rp)
    reflectance = 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)
    phase = np.angle(r_field)
    return r_field.astype(np.complex64), reflectance.astype(np.float32), phase.astype(np.float32)


def reflection_acceptance(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    view_z = 1.0
    reflected_z = np.clip(2.0 * nz * nz - view_z, -1.0, 1.0)
    reflected_angle = np.arccos(reflected_z)
    theta_na = math.asin(NA)
    acceptance = np.exp(-((reflected_angle / max(theta_na, 1e-6)) ** 4))
    return acceptance.astype(np.float32), reflected_angle.astype(np.float32)


def high_frequency_micro_relief(z_um: np.ndarray, rms_nm: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    smooth = gaussian_filter(z_um.astype(np.float32), sigma=18.0)
    highpass = z_um.astype(np.float32) - smooth
    highpass = highpass / (float(np.std(highpass)) + 1e-9)
    grain = rng.normal(0.0, 1.0, z_um.shape).astype(np.float32)
    grain = gaussian_filter(grain, sigma=0.85)
    grain = grain / (float(np.std(grain)) + 1e-9)
    relief = 0.55 * highpass + 0.45 * grain
    relief = relief / (float(np.std(relief)) + 1e-9) * (rms_nm * 1e-3)

    local_rms = gaussian_filter(relief * relief, sigma=8.0)
    local_rms = np.sqrt(np.maximum(local_rms, 0))
    return relief.astype(np.float32), local_rms.astype(np.float32)


def simulate_reflection_interference(case: ReflectionCase) -> dict[str, np.ndarray | float]:
    z_um, _ = generate_surface(case.surface)
    nx, ny, nz = normals_from_height(z_um)
    cos_theta = np.clip(nz, 1e-4, 1.0)
    acceptance, reflected_angle = reflection_acceptance(nx, ny, nz)
    r_field, fresnel_reflectance, fresnel_phase = fresnel_unpolarized_complex(
        cos_theta, case.metal.n, case.metal.k
    )
    micro_relief, local_rms = high_frequency_micro_relief(z_um, case.metal.roughness_rms_nm, case.surface.seed)

    roughness_phase_arg = 4.0 * np.pi * local_rms * cos_theta / WAVELENGTH_UM
    coherent_attenuation = np.exp(-0.5 * roughness_phase_arg * roughness_phase_arg).astype(np.float32)
    rough_scatter = case.metal.diffuse_fraction * fresnel_reflectance * (1.0 - coherent_attenuation**2)

    phase = (4.0 * np.pi * micro_relief * cos_theta / WAVELENGTH_UM + fresnel_phase).astype(np.float32)
    field_amp = np.sqrt(np.maximum(fresnel_reflectance * acceptance, 0.0)) * coherent_attenuation
    surface_field = field_amp.astype(np.float32) * np.exp(1j * phase)

    intensity_incoherent_hr = np.abs(surface_field) ** 2 + rough_scatter
    base = block_mean(intensity_incoherent_hr, SUPER).astype(np.float32)
    coherent_surface = block_mean(surface_field, SUPER)
    incoherent_scatter = block_mean(rough_scatter, SUPER).astype(np.float32)

    yy, xx = np.mgrid[0:SENSOR_H, 0:SENSOR_W].astype(np.float32)
    ref_phase = 2.0 * np.pi * (xx / 145.0 + yy / 310.0) + 0.65
    ref_amp = case.metal.parasitic_reflection_amp * math.sqrt(float(np.mean(base)) + 1e-9)
    parasitic_field = ref_amp * np.exp(1j * ref_phase)

    coherent_only = np.abs(coherent_surface) ** 2 + incoherent_scatter
    with_reference = (
        case.metal.spatial_coherence * np.abs(coherent_surface + parasitic_field) ** 2
        + (1.0 - case.metal.spatial_coherence) * base
        + incoherent_scatter * case.metal.spatial_coherence
    )
    artifact = with_reference.astype(np.float32) - base

    return {
        "height_um": z_um,
        "micro_relief_um": micro_relief,
        "acceptance": acceptance,
        "reflected_angle_deg": np.degrees(reflected_angle).astype(np.float32),
        "fresnel_reflectance": fresnel_reflectance,
        "coherent_attenuation": coherent_attenuation,
        "base": base,
        "coherent_only": coherent_only.astype(np.float32),
        "with_reference": with_reference.astype(np.float32),
        "artifact": artifact.astype(np.float32),
        "artifact_norm": artifact / (float(np.mean(base)) + 1e-9),
    }


def save_case(case: ReflectionCase) -> dict[str, float | str]:
    case_dir = OUT_DIR / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    sim = simulate_reflection_interference(case)

    for key in [
        "height_um",
        "micro_relief_um",
        "acceptance",
        "fresnel_reflectance",
        "coherent_attenuation",
        "base",
        "coherent_only",
        "with_reference",
        "artifact_norm",
    ]:
        np.save(case_dir / f"{key}.npy", np.asarray(sim[key]).astype(np.float32))

    (case_dir / "case_config.json").write_text(
        json.dumps(
            {
                "case_id": case.case_id,
                "label": case.label,
                "surface": asdict(case.surface),
                "metal": asdict(case.metal),
                "wavelength_um": WAVELENGTH_UM,
                "na": NA,
                "object_pixel_um": OBJECT_PIXEL_UM,
                "super_sampling": SUPER,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )

    z_sensor = block_mean(np.asarray(sim["height_um"]), SUPER)
    acceptance_sensor = block_mean(np.asarray(sim["acceptance"]), SUPER)
    fresnel_sensor = block_mean(np.asarray(sim["fresnel_reflectance"]), SUPER)
    coherence_sensor = block_mean(np.asarray(sim["coherent_attenuation"]), SUPER)
    material_map = normalize01(0.55 * fresnel_sensor + 0.45 * coherence_sensor)
    base = np.asarray(sim["base"])
    with_reference = np.asarray(sim["with_reference"])
    artifact_norm = np.asarray(sim["artifact_norm"])
    artifact_p95 = float(np.percentile(np.abs(artifact_norm), 95))
    vmax = max(artifact_p95, 0.02)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), dpi=180)
    panels = [
        (z_sensor, "Macro surface height (um)", "viridis", None, None),
        (acceptance_sensor, "NA acceptance of specular reflection", "magma", 0, 1),
        (material_map, "Fresnel reflectance + coherence", "cividis", 0, 1),
        (normalize01(base), "Incoherent reflected intensity baseline", "gray", 0, 1),
        (normalize01(with_reference), "Reflection wavefront interference image", "gray", 0, 1),
        (artifact_norm, f"Artifact: interference - baseline (±p95={artifact_p95:.3f})", "coolwarm", -vmax, vmax),
    ]
    for ax, (img, title, cmap, vmin, vmax_panel) in zip(axes.ravel(), panels):
        im = ax.imshow(
            img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_panel,
            extent=[0, FOV_WIDTH_UM, FOV_HEIGHT_UM, 0],
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("object x (um)")
        ax.set_ylabel("object y (um)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"{case.label}: reflection-only wavefront interference", y=0.995)
    fig.tight_layout()
    panel_path = case_dir / f"{case.case_id}_reflection_interference_panel.png"
    fig.savefig(panel_path)
    plt.close(fig)

    abs_artifact = np.abs(artifact_norm)
    high_accept = acceptance_sensor > np.percentile(acceptance_sensor, 80)
    metrics = {
        "case_id": case.case_id,
        "label": case.label,
        "metal_model": case.metal.name,
        "roughness_rms_nm": case.metal.roughness_rms_nm,
        "spatial_coherence": case.metal.spatial_coherence,
        "parasitic_reflection_amp": case.metal.parasitic_reflection_amp,
        "height_range_um": float(np.max(z_sensor) - np.min(z_sensor)),
        "mean_fresnel_reflectance": float(np.mean(fresnel_sensor)),
        "mean_coherent_attenuation": float(np.mean(coherence_sensor)),
        "mean_acceptance": float(np.mean(acceptance_sensor)),
        "artifact_p95_abs_relative_to_mean": float(np.percentile(abs_artifact, 95)),
        "artifact_p99_abs_relative_to_mean": float(np.percentile(abs_artifact, 99)),
        "artifact_rms_relative_to_mean": float(np.sqrt(np.mean(artifact_norm**2))),
        "high_acceptance_artifact_mean_abs": float(np.mean(abs_artifact[high_accept])),
        "panel": str(panel_path.relative_to(OUT_DIR)).replace("\\", "/"),
    }
    return metrics


def write_report(metrics: pd.DataFrame) -> Path:
    table = metrics[
        [
            "case_id",
            "roughness_rms_nm",
            "spatial_coherence",
            "mean_fresnel_reflectance",
            "mean_coherent_attenuation",
            "mean_acceptance",
            "artifact_p95_abs_relative_to_mean",
            "artifact_p99_abs_relative_to_mean",
        ]
    ].to_markdown(index=False, floatfmt=".4f")
    figure_lines = "\n".join(f"- {row.case_id}: ![{row.case_id}]({row.panel})" for row in metrics.itertuples())

    report = f"""# 金属表面反射波前干涉伪影模拟

## 结论

本轮重新限定问题：**不使用 PSF 卷积，不模拟透镜衍射像差，只模拟金属表面反射复振幅的相干叠加是否会产生伪影。** 模型使用项目已有 `src/surface_sample_generator.py` 生成 A 型刃脊、V 谷和阶跃三类微结构；宏观高度用于计算局部法线、入射角、物镜 NA 接收权重，纳米到亚微米级残余微起伏用于反射相位。

在 525 nm、NA={NA:.2f}、20X、物方采样 {OBJECT_PIXEL_UM:.3f} um/pixel 的设置下，反射干涉图相对非相干反射基线产生了 **{metrics['artifact_p95_abs_relative_to_mean'].min():.3f}-{metrics['artifact_p95_abs_relative_to_mean'].max():.3f}** 的 p95 相对强度扰动。伪影主要出现在镜面反射能进入物镜接收锥的位置，以及微粗糙度仍保留一定相干性的区域。这说明：即使不考虑成像 PSF，仅金属表面反射波前的相干积分和弱寄生反射，也足以产生条纹状、斑点状或局部亮暗调制伪影。

## 物理模型

### 1. 金属 Fresnel 复反射系数

金属复折射率写作：

$$
\\tilde n = n + i\\kappa.
$$

对局部入射角 $\\theta$，s/p 偏振 Fresnel 系数为：

$$
r_s = \\frac{{\\cos\\theta - q}}{{\\cos\\theta + q}}, \\qquad
r_p = \\frac{{\\tilde n^2\\cos\\theta - q}}{{\\tilde n^2\\cos\\theta + q}},
$$

$$
q = \\sqrt{{\\tilde n^2 - \\sin^2\\theta}}.
$$

代码中使用未偏振近似：

$$
r = \\frac{{r_s+r_p}}{{2}}, \\qquad
R = \\frac{{|r_s|^2+|r_p|^2}}{{2}}.
$$

### 2. 粗糙度导致的相干衰减

局部 RMS 粗糙度记为 $\\sigma$，反射相干振幅衰减近似为：

$$
C_\\sigma = \\exp\\left[-\\frac12\\left(\\frac{{4\\pi\\sigma\\cos\\theta}}{{\\lambda}}\\right)^2\\right].
$$

表面越粗糙，镜面相干反射越弱，漫散射/非相干成分越强。

### 3. 反射波前相位

固定同轴照明下，局部残余高度 $h_r(x,y)$ 造成的反射相位为：

$$
\\phi(x,y)=\\frac{{4\\pi h_r(x,y)\\cos\\theta}}{{\\lambda}} + \\arg(r).
$$

对应反射复场：

$$
E_r(x,y)=r(\\theta)\\sqrt{{A_{{NA}}(x,y)}}C_\\sigma(x,y)e^{{i\\phi(x,y)}}.
$$

其中 $A_{{NA}}$ 表示镜面反射方向落入物镜 NA 接收锥的权重。

### 4. 传感器像素内相干积分

非相干反射基线为：

$$
I_{{base}}(P)=\\frac1N\\sum_{{x\\in P}} |E_r(x)|^2 + I_{{scatter}}.
$$

反射波前干涉图为：

$$
I_{{int}}(P)=\\mu\\left|\\frac1N\\sum_{{x\\in P}}E_r(x)+E_g(P)\\right|^2
+(1-\\mu)I_{{base}}(P)+I_{{scatter}},
$$

其中 $E_g$ 是弱寄生反射场，$\\mu$ 是空间相干权重。伪影定义为：

$$
A(P)=\\frac{{I_{{int}}(P)-I_{{base}}(P)}}{{\\mathrm{{mean}}(I_{{base}})}}.
$$

## 指标

{table}

## 图像结果

{figure_lines}

## 与上一版的区别

上一版把 Airy PSF、pupil 像差和弱干涉放在同一个系统级模拟中，适合说明“成像链可能接近衍射限制并出现 halo”。本轮去掉了 PSF 和透镜像差，只保留金属表面反射相关项：Fresnel 复反射、NA 接收锥、粗糙度相干衰减、表面残余高度相位、像素内相干积分和弱寄生反射。因此，本轮更直接对应你的问题：**反射本身的波前干涉是否能导致伪影。**

## 后续建议

1. 若能确认金属材料，可把 `n,k` 替换为对应材料在 525 nm 的实测或文献光学常数。
2. 若能测得表面粗糙度 Ra/Rq，可把 `roughness_rms_nm` 从假设量改成实测量。
3. 若要贴近 LED，可用实测 LED 带宽和照明孔径估计 `spatial_coherence`，再做相干性扫描。
4. 若要和真实图比对，应优先比较 artifact map 的空间频率、局部亮暗调制幅度和高反射边缘的条纹方向。
"""
    path = OUT_DIR / "reflection_wavefront_interference_metal_surface_report.md"
    path.write_text(report, encoding="utf-8", newline="\n")
    return path


def main() -> None:
    rows = [save_case(case) for case in CASES]
    metrics = pd.DataFrame(rows)
    metrics_path = OUT_DIR / "reflection_wavefront_interference_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    report_path = write_report(metrics)
    print(report_path)
    print(metrics_path)
    for row in rows:
        print(OUT_DIR / row["panel"])


if __name__ == "__main__":
    main()
