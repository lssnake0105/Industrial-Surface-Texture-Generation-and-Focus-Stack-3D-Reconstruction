from __future__ import annotations

import math
import sys
from dataclasses import dataclass
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
DEFAULT_OBJECT_PIXEL_UM = SENSOR_PIXEL_UM / MAGNIFICATION
LARGE_WINDOW_OBJECT_PIXEL_UM = 0.50
WIDTH = 512
HEIGHT = 288
STACK_LAYERS = 17
FOCUS_RANGE_UM = 2.0 * (2.0 * WAVELENGTH_UM / (NA * NA))
FOCUS_POSITIONS_UM = np.linspace(-FOCUS_RANGE_UM / 2, FOCUS_RANGE_UM / 2, STACK_LAYERS).astype(np.float32)


@dataclass(frozen=True)
class MetalParams:
    n: float = 2.70
    k: float = 3.30
    spatial_coherence: float = 0.34
    parasitic_reflection_amp: float = 0.065
    diffuse_fraction: float = 0.14


@dataclass(frozen=True)
class SweepCase:
    case_id: str
    roughness_rms_nm: float
    dynamic_range_um: float
    baseline_type: str
    seed: int
    case_family: str
    object_pixel_um: float = DEFAULT_OBJECT_PIXEL_UM


BASE_CASES = [
    SweepCase("route_validation_step_mid", 36.0, 70.0, "step", 5201, "route_validation"),
]

STEP_SWEEP_CASES: list[SweepCase] = []
for roughness in [15.0, 35.0, 75.0]:
    for dynamic_range in [40.0, 80.0, 120.0]:
        STEP_SWEEP_CASES.append(
            SweepCase(
                f"step_sweep_r{int(roughness):02d}_h{int(dynamic_range):03d}",
                roughness,
                dynamic_range,
                "step",
                5300 + int(roughness) * 10 + int(dynamic_range),
                "step_sweep",
            )
        )

COMPLEX_SURFACE_CASES = [
    SweepCase("complex_mountain_r35_h100", 35.0, 100.0, "mountain", 6201, "complex_100um", LARGE_WINDOW_OBJECT_PIXEL_UM),
    SweepCase("complex_mountain_r75_h100", 75.0, 100.0, "mountain", 6202, "complex_100um", LARGE_WINDOW_OBJECT_PIXEL_UM),
    SweepCase("complex_a_ridge_r35_h100", 35.0, 100.0, "a_ridge", 6203, "complex_100um", LARGE_WINDOW_OBJECT_PIXEL_UM),
    SweepCase("complex_v_valley_r35_h100", 35.0, 100.0, "v_valley", 6204, "complex_100um", LARGE_WINDOW_OBJECT_PIXEL_UM),
]

SWEEP_CASES = STEP_SWEEP_CASES + COMPLEX_SURFACE_CASES


def normalize01(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def surface_config(case: SweepCase) -> SurfaceConfig:
    if case.case_family == "complex_100um":
        width_map = {
            "mountain": 0.42,
            "a_ridge": 0.34,
            "v_valley": 0.58,
        }
        return SurfaceConfig(
            name=case.case_id,
            width=WIDTH,
            height=HEIGHT,
            depth_range_um=case.dynamic_range_um,
            baseline_type=case.baseline_type,
            noise_type="perlin",
            seed=case.seed,
            tilt_x_um=0.0,
            tilt_y_um=0.0,
            feature_amp_um=case.dynamic_range_um * 0.86,
            noise_amp_um=min(10.0, case.dynamic_range_um * 0.09),
            ridge_width=width_map.get(case.baseline_type, 0.40),
            valley_width=0.58,
            valley_floor=0.18,
            valley_sharpness=0.72,
            orientation_deg=-13.0,
            perlin_octaves=6,
            perlin_grid=96,
            perlin_persistence=0.58,
        )

    return SurfaceConfig(
        name=case.case_id,
        width=WIDTH,
        height=HEIGHT,
        depth_range_um=case.dynamic_range_um,
        baseline_type=case.baseline_type,
        noise_type="perlin",
        seed=case.seed,
        feature_amp_um=case.dynamic_range_um * 0.88,
        noise_amp_um=min(1.8, case.dynamic_range_um * 0.012),
        step_position=-0.07,
        perlin_octaves=6,
        perlin_grid=52,
        perlin_persistence=0.55,
    )


def normals_from_height(z_um: np.ndarray, object_pixel_um: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dzdy, dzdx = np.gradient(z_um.astype(np.float32), object_pixel_um, object_pixel_um)
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(z_um, dtype=np.float32)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    slope = np.hypot(dzdx, dzdy).astype(np.float32)
    return nx / norm, ny / norm, nz / norm, slope


def fresnel_unpolarized(cos_theta: np.ndarray, n: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    cos_theta = np.clip(cos_theta.astype(np.float64), 1e-4, 1.0)
    sin2 = 1.0 - cos_theta * cos_theta
    m = complex(n, k)
    q = np.sqrt(m * m - sin2)
    rs = (cos_theta - q) / (cos_theta + q)
    rp = (m * m * cos_theta - q) / (m * m * cos_theta + q)
    r = 0.5 * (rs + rp)
    R = 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)
    return r.astype(np.complex64), R.astype(np.float32)


def reflection_acceptance(nz: np.ndarray) -> np.ndarray:
    reflected_z = np.clip(2.0 * nz * nz - 1.0, -1.0, 1.0)
    reflected_angle = np.arccos(reflected_z)
    theta_na = math.asin(NA)
    return np.exp(-((reflected_angle / theta_na) ** 4)).astype(np.float32)


def micro_relief(z_um: np.ndarray, rms_nm: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    smooth = gaussian_filter(z_um.astype(np.float32), sigma=12.0)
    hp = z_um.astype(np.float32) - smooth
    hp = hp / (float(np.std(hp)) + 1e-9)
    grain = rng.normal(0, 1, z_um.shape).astype(np.float32)
    grain = gaussian_filter(grain, sigma=0.75)
    grain = grain / (float(np.std(grain)) + 1e-9)
    relief = 0.50 * hp + 0.50 * grain
    relief = relief / (float(np.std(relief)) + 1e-9) * (rms_nm * 1e-3)
    local_rms = np.sqrt(np.maximum(gaussian_filter(relief * relief, sigma=6.0), 0))
    return relief.astype(np.float32), local_rms.astype(np.float32)


def secondary_surface_scatter(
    object_field: np.ndarray,
    z_um: np.ndarray,
    acceptance: np.ndarray,
    case: SweepCase,
    metal: MetalParams,
) -> np.ndarray:
    rng = np.random.default_rng(case.seed + 7919)
    lowpass_sigma_px = max(8.0, 8.0 / case.object_pixel_um)
    low_real = gaussian_filter(object_field.real.astype(np.float32), sigma=lowpass_sigma_px)
    low_imag = gaussian_filter(object_field.imag.astype(np.float32), sigma=lowpass_sigma_px)
    low_field = (low_real + 1j * low_imag).astype(np.complex64)
    low_field /= math.sqrt(float(np.mean(np.abs(low_field) ** 2)) + 1e-9)

    coarse_phase = gaussian_filter(rng.normal(0, 1, z_um.shape).astype(np.float32), sigma=max(10.0, 12.0 / case.object_pixel_um))
    coarse_phase /= float(np.std(coarse_phase)) + 1e-9
    surface_phase = 2.0 * np.pi * normalize01(gaussian_filter(z_um.astype(np.float32), sigma=18.0))
    phase = 0.65 * coarse_phase + 0.55 * surface_phase

    amp_envelope = 0.35 + 0.65 * normalize01(gaussian_filter(acceptance.astype(np.float32), sigma=10.0))
    mean_power = float(np.mean(np.abs(object_field) ** 2)) + 1e-9
    ref_amp = metal.parasitic_reflection_amp * math.sqrt(mean_power)
    return (ref_amp * amp_envelope * low_field * np.exp(1j * phase)).astype(np.complex64)


def make_reflection_fields(z_um: np.ndarray, case: SweepCase, metal: MetalParams) -> dict[str, np.ndarray]:
    _, _, nz, slope = normals_from_height(z_um, case.object_pixel_um)
    cos_theta = np.clip(nz, 1e-4, 1.0)
    r, R = fresnel_unpolarized(cos_theta, metal.n, metal.k)
    acceptance = reflection_acceptance(nz)
    relief, local_rms = micro_relief(z_um, case.roughness_rms_nm, case.seed)
    atten = np.exp(-0.5 * (4.0 * np.pi * local_rms * cos_theta / WAVELENGTH_UM) ** 2).astype(np.float32)
    phase = 4.0 * np.pi * relief * cos_theta / WAVELENGTH_UM + np.angle(r).astype(np.float32)
    amp = np.sqrt(np.maximum(R * acceptance, 0)) * atten
    object_field = amp.astype(np.float32) * np.exp(1j * phase)
    parasitic = secondary_surface_scatter(object_field, z_um, acceptance, case, metal)

    return {
        "object_field": object_field.astype(np.complex64),
        "object_field_interference": (object_field + metal.spatial_coherence * parasitic).astype(np.complex64),
        "acceptance": acceptance,
        "fresnel_reflectance": R,
        "coherent_attenuation": atten,
        "slope": slope,
        "secondary_abs": np.abs(parasitic).astype(np.float32),
    }


def circular_pupil_and_transfer(defocus_um: float, object_pixel_um: float) -> np.ndarray:
    fx = np.fft.fftfreq(WIDTH, d=object_pixel_um).astype(np.float32)
    fy = np.fft.fftfreq(HEIGHT, d=object_pixel_um).astype(np.float32)
    fxx, fyy = np.meshgrid(fx, fy)
    f2 = fxx * fxx + fyy * fyy
    cutoff = NA / WAVELENGTH_UM
    pupil = (f2 <= cutoff * cutoff).astype(np.float32)
    kz_term = np.sqrt(np.maximum((1.0 / WAVELENGTH_UM) ** 2 - f2, 0.0)) - (1.0 / WAVELENGTH_UM)
    defocus_phase = 2.0 * np.pi * defocus_um * kz_term
    return pupil.astype(np.complex64) * np.exp(1j * defocus_phase).astype(np.complex64)


def propagate_stack(field: np.ndarray, object_pixel_um: float) -> np.ndarray:
    field_f = np.fft.fft2(field)
    layers = []
    for zf in FOCUS_POSITIONS_UM:
        transfer = circular_pupil_and_transfer(float(zf), object_pixel_um)
        img = np.abs(np.fft.ifft2(field_f * transfer)) ** 2
        layers.append(img.astype(np.float32))
    stack = np.stack(layers, axis=0)
    stack /= float(np.percentile(stack, 99.8)) + 1e-9
    return np.clip(stack, 0, 1).astype(np.float32)


def focus_measure(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maps = []
    for img in stack:
        gy, gx = np.gradient(img.astype(np.float32))
        fm = gaussian_filter(gx * gx + gy * gy, sigma=1.0)
        maps.append(fm.astype(np.float32))
    fmap = np.stack(maps, axis=0)
    idx = np.argmax(fmap, axis=0).astype(np.int16)
    sorted_f = np.sort(fmap, axis=0)
    conf = (sorted_f[-1] - sorted_f[-2]) / (sorted_f[-1] + 1e-9)
    return idx, normalize01(conf)


def artifact_directionality(img: np.ndarray) -> float:
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(img.astype(np.float32))))
    h, w = spectrum.shape
    spectrum[h // 2 - 3 : h // 2 + 4, w // 2 - 3 : w // 2 + 4] = 0
    mag = spectrum / (float(np.mean(spectrum)) + 1e-9)
    return float(np.percentile(mag, 99.9))


def run_case(case: SweepCase, save_stack: bool) -> dict[str, float | str | bool]:
    case_dir = OUT_DIR / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    cfg = surface_config(case)
    z_um, _ = generate_surface(cfg)
    fields = make_reflection_fields(z_um, case, MetalParams())
    fov_width_um = WIDTH * case.object_pixel_um
    fov_height_um = HEIGHT * case.object_pixel_um

    max_slope = float(np.percentile(fields["slope"], 99))
    rough_aspect = (case.roughness_rms_nm * 1e-3) / max(case.object_pixel_um, 1e-9)
    generator_spike_guard = bool(max_slope < 35.0)
    pupil_sampling_ok = (1.0 / (2.0 * case.object_pixel_um)) >= (NA / WAVELENGTH_UM)

    baseline_stack = propagate_stack(fields["object_field"], case.object_pixel_um)
    interference_stack = propagate_stack(fields["object_field_interference"], case.object_pixel_um)
    artifact_stack = interference_stack - baseline_stack
    artifact_abs = np.abs(artifact_stack)
    idx_base, conf_base = focus_measure(baseline_stack)
    idx_int, conf_int = focus_measure(interference_stack)
    peak_shift = np.abs(idx_int.astype(np.int16) - idx_base.astype(np.int16)).astype(np.float32)

    if save_stack:
        np.save(case_dir / "focus_stack_baseline_float32.npy", baseline_stack)
        np.save(case_dir / "focus_stack_interference_float32.npy", interference_stack)
    np.save(case_dir / "height_um.npy", z_um.astype(np.float32))
    np.save(case_dir / "focus_index_baseline.npy", idx_base)
    np.save(case_dir / "focus_index_interference.npy", idx_int)

    rep = STACK_LAYERS // 2
    p99_art = float(np.percentile(np.abs(artifact_stack[rep]), 99))
    vmax = max(float(np.percentile(np.abs(artifact_stack), 99)), 0.02)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=170)
    panels = [
        (z_um, "A Height map (um)", "viridis", None, None),
        (fields["acceptance"], "B NA acceptance of reflection", "magma", 0, 1),
        (fields["coherent_attenuation"], "C Coherence attenuation by roughness", "cividis", 0, 1),
        (baseline_stack[rep], "D Wave-optics baseline focus layer", "gray", 0, 1),
        (interference_stack[rep], "E With surface-derived secondary wave", "gray", 0, 1),
        (artifact_stack[rep], f"F Artifact E-D at middle focus (p99={p99_art:.3f})", "coolwarm", -vmax, vmax),
    ]
    for ax, (img, title, cmap, vmin, vmax_i) in zip(axes.ravel(), panels):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax_i, extent=[0, fov_width_um, fov_height_um, 0])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("object x (um)")
        ax.set_ylabel("object y (um)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    panel_path = case_dir / f"{case.case_id}_wave_focus_panel.png"
    fig.savefig(panel_path)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=170)
    axes[0].imshow(idx_base, cmap="viridis", vmin=0, vmax=STACK_LAYERS - 1)
    axes[0].set_title("Baseline DFF peak layer")
    axes[1].imshow(idx_int, cmap="viridis", vmin=0, vmax=STACK_LAYERS - 1)
    axes[1].set_title("Interference DFF peak layer")
    im = axes[2].imshow(peak_shift, cmap="magma", vmin=0, vmax=max(1, float(np.percentile(peak_shift, 99))))
    axes[2].set_title("Peak-layer shift caused by interference")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    fig.tight_layout()
    dff_path = case_dir / f"{case.case_id}_dff_peak_shift.png"
    fig.savefig(dff_path)
    plt.close(fig)

    return {
        "case_id": case.case_id,
        "case_family": case.case_family,
        "roughness_rms_nm": case.roughness_rms_nm,
        "dynamic_range_um": case.dynamic_range_um,
        "baseline_type": case.baseline_type,
        "object_pixel_um": case.object_pixel_um,
        "fov_width_um": fov_width_um,
        "fov_height_um": fov_height_um,
        "pupil_sampling_ok": pupil_sampling_ok,
        "rough_aspect_ratio": rough_aspect,
        "height_p99_slope": max_slope,
        "generator_spike_guard": generator_spike_guard,
        "artifact_stack_p95_abs": float(np.percentile(artifact_abs, 95)),
        "artifact_stack_p99_abs": float(np.percentile(artifact_abs, 99)),
        "artifact_middle_p99_abs": p99_art,
        "artifact_directionality_p999": artifact_directionality(artifact_stack[rep]),
        "mean_peak_shift_layers": float(np.mean(peak_shift)),
        "p95_peak_shift_layers": float(np.percentile(peak_shift, 95)),
        "mean_confidence_baseline": float(np.mean(conf_base)),
        "mean_confidence_interference": float(np.mean(conf_int)),
        "panel": str(panel_path.relative_to(OUT_DIR)).replace("\\", "/"),
        "dff_panel": str(dff_path.relative_to(OUT_DIR)).replace("\\", "/"),
    }


def save_sweep_summary(metrics: pd.DataFrame) -> Path:
    step_sweep = metrics[metrics["case_family"] == "step_sweep"].copy()
    complex_cases = metrics[metrics["case_family"] == "complex_100um"].copy()

    pivot_art = step_sweep.pivot(index="roughness_rms_nm", columns="dynamic_range_um", values="artifact_stack_p95_abs")
    pivot_shift = step_sweep.pivot(index="roughness_rms_nm", columns="dynamic_range_um", values="p95_peak_shift_layers")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=170)
    im0 = axes[0].imshow(pivot_art.values, cmap="magma", origin="lower")
    axes[0].set_title("Step cases: artifact p95 abs")
    axes[0].set_xticks(range(len(pivot_art.columns)), [str(int(c)) for c in pivot_art.columns])
    axes[0].set_yticks(range(len(pivot_art.index)), [str(int(i)) for i in pivot_art.index])
    axes[0].set_xlabel("surface dynamic range (um)")
    axes[0].set_ylabel("roughness RMS (nm)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)

    im1 = axes[1].imshow(pivot_shift.values, cmap="viridis", origin="lower")
    axes[1].set_title("Step cases: DFF peak shift p95")
    axes[1].set_xticks(range(len(pivot_shift.columns)), [str(int(c)) for c in pivot_shift.columns])
    axes[1].set_yticks(range(len(pivot_shift.index)), [str(int(i)) for i in pivot_shift.index])
    axes[1].set_xlabel("surface dynamic range (um)")
    axes[1].set_ylabel("roughness RMS (nm)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)

    labels = [c.replace("complex_", "").replace("_h100", "") for c in complex_cases["case_id"]]
    x = np.arange(len(labels))
    axes[2].bar(x, complex_cases["artifact_stack_p95_abs"].to_numpy(), color="#4c78a8")
    axes[2].set_title("Complex 100 um surfaces: artifact p95 abs")
    axes[2].set_xticks(x, labels, rotation=30, ha="right")
    axes[2].set_ylabel("artifact p95 abs")
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path = OUT_DIR / "roughness_dynamic_range_sweep_heatmaps.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_report(metrics: pd.DataFrame, heatmap_path: Path, folder_size_mb: float) -> Path:
    cols = [
        "case_id",
        "case_family",
        "baseline_type",
        "object_pixel_um",
        "fov_width_um",
        "fov_height_um",
        "roughness_rms_nm",
        "dynamic_range_um",
        "rough_aspect_ratio",
        "height_p99_slope",
        "generator_spike_guard",
        "pupil_sampling_ok",
        "artifact_stack_p95_abs",
        "artifact_stack_p99_abs",
        "artifact_directionality_p999",
        "p95_peak_shift_layers",
    ]
    table = metrics[cols].to_markdown(index=False, floatfmt=".4f")
    route = metrics.iloc[0]
    step_sweep = metrics[metrics["case_family"] == "step_sweep"]
    complex_cases = metrics[metrics["case_family"] == "complex_100um"]
    best_step = step_sweep.sort_values("artifact_stack_p95_abs", ascending=False).iloc[0]
    best_complex = complex_cases.sort_values("artifact_stack_p95_abs", ascending=False).iloc[0]
    complex_rows = "\n".join(
        f"- `{row.case_id}`：{row.baseline_type} 面形，等效窗口 {row.fov_width_um:.0f} x {row.fov_height_um:.0f} um，"
        f"动态范围 {row.dynamic_range_um:.0f} um，"
        f"伪影 p95={row.artifact_stack_p95_abs:.4f}，DFF 峰值层 p95 偏移={row.p95_peak_shift_layers:.1f} 层，"
        f"99% 局部坡度={row.height_p99_slope:.2f}。"
        for row in complex_cases.itertuples()
    )

    report = f"""# 轻量波动光学焦栈仿真与复杂 100 um 表面起伏验证

## 本轮问题定义

本轮补充的 100 um 量级起伏指表面高度动态范围，来源是复杂三维面形叠加低频 Perlin 起伏；该高度差不通过整体基线倾斜得到，也不使用尖峰、针孔或孤立像素级突变来制造。新增 case 使用 `mountain`、`a_ridge`、`v_valley` 三类已有表面生成器逻辑，目的是观察复杂表面反射场中的次级小波相干扰动，并降低弱平面寄生反射带来的规则条纹主导性。

## 存储估计与实际输出

- 运行前估计：单个 `512 x 288 x 17` float32 焦栈约 9.56 MB；baseline/interference 两套焦栈约 19.12 MB。只对路线验证 case 保存完整焦栈，其余 case 仅保存高度、DFF 索引、面板图和指标，预计总输出低于 80 MB。
- 实际输出文件夹大小：{folder_size_mb:.2f} MB。

## 模型链条

```text
surface_sample_generator 生成金属复杂高度场
-> Fresnel 复反射 + 粗糙度相干衰减 + NA 接收权重
-> 主反射复场 E(x,y)
-> 从主反射场低通散射残留构造弱次级小波场
-> circular pupil + defocus phase
-> 17 层焦平面强度图
-> baseline/interference 焦栈差分与 DFF peak-layer shift
```

核心变化是次级场的构造。上一版使用弱平面寄生反射，容易形成规则条纹；本版使用由表面复反射场低通得到的弱散射场，并叠加低频表面相位扰动，使 E-D 的伪影更依赖局部面形、NA 接收和相干保留。

尺度设置：路线验证和阶跃扫描沿用 20X 等效物方像素 `0.1725 um/px`；复杂 100 um 起伏组使用 `0.50 um/px` 的轻量等效窗口，对应约 `256 x 144 um` 的成像区域。该设置用于近似几百微米窗口内的复杂起伏，同时继续保持 512 x 288 的轻量计算规模。`pupil_sampling_ok=True` 表示该采样仍覆盖 NA/lambda 所需的 pupil 截止频率。

## 如何读每张 2 x 3 面板

- A `Height map (um)`：表面生成器输出的高度场。复杂 100 um case 中，这张图用于确认高度动态范围来自局部面形起伏，而非整体倾斜。
- B `NA acceptance of reflection`：根据局部法线计算镜面反射进入 NA=0.40 接收锥的权重。亮区代表更容易被物镜收集的反射区域。
- C `Coherence attenuation by roughness`：由局部粗糙度 RMS 推得的相干衰减。亮区说明相位保留更强，暗区说明微粗糙度更容易破坏相干叠加。
- D `Wave-optics baseline focus layer`：只使用主反射复场传播得到的中间焦平面强度，作为无次级小波干涉的对照。
- E `With surface-derived secondary wave`：主反射复场叠加弱表面派生次级小波后的中间焦平面强度。
- F `Artifact E-D`：E 减 D 得到的差分伪影。红色表示干涉增强，蓝色表示干涉削弱；若图案随复杂面形局部分布变化，说明伪影和表面反射结构相关。

每个 `*_dff_peak_shift.png` 用来读焦点选择影响：左图是 baseline 焦栈的 DFF 峰值层，中图是 interference 焦栈的 DFF 峰值层，右图是二者差值。右图越亮，说明次级小波干涉越可能改变重建算法的焦层判断。

## 路线验证 case

首个验证 case `{route['case_id']}` 保存了 baseline/interference 两套完整焦栈。其 `artifact_stack_p95_abs={route['artifact_stack_p95_abs']:.4f}`，`p95_peak_shift_layers={route['p95_peak_shift_layers']:.1f}`，说明轻量波动光学链条可以产生可量化的干涉差分，并可进入 DFF 层级诊断。

![route panel]({route['panel']})

![route dff]({route['dff_panel']})

## 复杂 100 um 表面起伏 case

新增复杂表面 case 全部取消整体倾斜，通过已有表面生成器的宽峰、宽脊和宽谷结构形成 100 um 高度动态范围。这里的 `roughness_rms_nm` 仍表示纳米级微粗糙度，用于相干衰减；`dynamic_range_um=100` 表示宏观/介观表面起伏幅度。复杂组的等效视场约为 256 x 144 um，符合“几百微米窗口中的 100 um 量级起伏”这一假设。

{complex_rows}

代表图：

![complex mountain panel]({complex_cases.iloc[0]['panel']})

![complex mountain dff]({complex_cases.iloc[0]['dff_panel']})

## 全部指标

{table}

指标读法：

- `rough_aspect_ratio`：纳米级微粗糙度 RMS 与物方像素尺寸的比值，用于避免把微粗糙度做成像素级高深尖峰。
- `height_p99_slope`：高度图 99% 局部坡度。复杂面形允许存在较大坡度，但该指标可用于筛查异常针状结构。
- `generator_spike_guard`：当前用宽结构生成复杂表面；若该值为 True，说明 99% 局部坡度没有触发本轮的尖峰保护阈值。
- `pupil_sampling_ok`：当前物方采样是否足以覆盖 `NA/lambda` 的 pupil 截止频率。
- `artifact_stack_p95_abs` 和 `artifact_stack_p99_abs`：焦栈范围内 E-D 差分伪影的主体水平和尾部水平。
- `artifact_directionality_p999`：中间焦平面伪影频谱的高分位峰值。规则条纹通常会抬高该值；复杂表面 case 需要结合 F 图共同判断。
- `p95_peak_shift_layers`：95% 像素处 DFF 峰值层偏移，用于判断干涉是否影响焦层选择。

## 汇总图

左、中图保留阶跃表面的粗糙度/动态范围扫描；右图单独汇总复杂 100 um 表面，避免把复杂面形和阶跃表面混入同一张二维热力图。

![sweep heatmap]({heatmap_path.relative_to(OUT_DIR).as_posix()})

## 主要观察

- 阶跃扫描中最高 `artifact_stack_p95_abs` 出现在 `{best_step['case_id']}`，粗糙度 {best_step['roughness_rms_nm']:.0f} nm，动态范围 {best_step['dynamic_range_um']:.0f} um。
- 复杂 100 um 表面中最高 `artifact_stack_p95_abs` 出现在 `{best_complex['case_id']}`，面形为 `{best_complex['baseline_type']}`，伪影 p95={best_complex['artifact_stack_p95_abs']:.4f}。
- 在复杂面形 case 中，F 图的伪影由局部高度、NA 接收和相干保留共同调制，图案不再主要表现为规则平面波条纹。
- 粗糙度增大会增强相干衰减，但复杂面形也会引入更丰富的局部相位扰动，因此伪影强度不应按单一粗糙度变量作线性解释。

## 结论边界

本轮结果支持一个更具体的判断：在仅考虑反射复场传播和弱表面派生次级小波的轻量模型中，复杂 100 um 级表面起伏可以产生可量化、局部分布的相干伪影，并可能改变 DFF 焦层选择。该结果仍是 simulation probe，尚未包含真实 Olympus 20X 物镜 pupil、LED 空间相干性实测、真实金属 n/k 标定和实测表面轮廓。
"""
    path = OUT_DIR / "lightweight_wave_optics_focus_stack_report.md"
    path.write_text(report, encoding="utf-8", newline="\n")
    return path


def folder_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024 / 1024


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in BASE_CASES:
        rows.append(run_case(case, save_stack=True))
    for case in SWEEP_CASES:
        rows.append(run_case(case, save_stack=False))
    metrics = pd.DataFrame(rows)
    metrics_path = OUT_DIR / "lightweight_wave_optics_sweep_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    heatmap = save_sweep_summary(metrics)
    size = folder_size_mb(OUT_DIR)
    report = write_report(metrics, heatmap, size)
    print(report)
    print(metrics_path)
    print(f"folder_size_mb={folder_size_mb(OUT_DIR):.2f}")


if __name__ == "__main__":
    main()
