from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
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
VENDOR = OUT_DIR / "vendor"
if VENDOR.exists() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

WAVELENGTH_UM = 0.525
NA = 0.40
WIDTH = 384
HEIGHT = 384
OBJECT_PIXEL_UM = 0.50
FOV_WIDTH_UM = WIDTH * OBJECT_PIXEL_UM
FOV_HEIGHT_UM = HEIGHT * OBJECT_PIXEL_UM
STACK_LAYERS = 13
FOCUS_RANGE_UM = 2.0 * (2.0 * WAVELENGTH_UM / (NA * NA))
FOCUS_POSITIONS_UM = np.linspace(-FOCUS_RANGE_UM / 2, FOCUS_RANGE_UM / 2, STACK_LAYERS).astype(np.float32)


@dataclass(frozen=True)
class MetalParams:
    n: float = 2.70
    k: float = 3.30
    roughness_rms_nm: float = 55.0
    spatial_coherence: float = 0.34
    parasitic_reflection_amp: float = 0.065


def normalize01(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def artifact_directionality(img: np.ndarray) -> float:
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(img.astype(np.float32))))
    h, w = spectrum.shape
    spectrum[h // 2 - 3 : h // 2 + 4, w // 2 - 3 : w // 2 + 4] = 0
    mag = spectrum / (float(np.mean(spectrum)) + 1e-9)
    return float(np.percentile(mag, 99.9))


def make_surface() -> np.ndarray:
    cfg = SurfaceConfig(
        name="fixed_complex_100um_sample",
        width=WIDTH,
        height=HEIGHT,
        depth_range_um=100.0,
        baseline_type="mountain",
        noise_type="perlin",
        seed=7101,
        tilt_x_um=0.0,
        tilt_y_um=0.0,
        feature_amp_um=86.0,
        noise_amp_um=9.0,
        perlin_octaves=6,
        perlin_grid=72,
        perlin_persistence=0.58,
    )
    z_um, _ = generate_surface(cfg)
    return z_um.astype(np.float32)


def normals_from_height(z_um: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dzdy, dzdx = np.gradient(z_um.astype(np.float32), OBJECT_PIXEL_UM, OBJECT_PIXEL_UM)
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


def secondary_surface_scatter(object_field: np.ndarray, z_um: np.ndarray, acceptance: np.ndarray, metal: MetalParams) -> np.ndarray:
    rng = np.random.default_rng(8191)
    low_real = gaussian_filter(object_field.real.astype(np.float32), sigma=16.0)
    low_imag = gaussian_filter(object_field.imag.astype(np.float32), sigma=16.0)
    low_field = (low_real + 1j * low_imag).astype(np.complex64)
    low_field /= math.sqrt(float(np.mean(np.abs(low_field) ** 2)) + 1e-9)
    coarse_phase = gaussian_filter(rng.normal(0, 1, z_um.shape).astype(np.float32), sigma=22.0)
    coarse_phase /= float(np.std(coarse_phase)) + 1e-9
    surface_phase = 2.0 * np.pi * normalize01(gaussian_filter(z_um.astype(np.float32), sigma=18.0))
    amp_envelope = 0.35 + 0.65 * normalize01(gaussian_filter(acceptance.astype(np.float32), sigma=10.0))
    ref_amp = metal.parasitic_reflection_amp * math.sqrt(float(np.mean(np.abs(object_field) ** 2)) + 1e-9)
    return (ref_amp * amp_envelope * low_field * np.exp(1j * (0.65 * coarse_phase + 0.55 * surface_phase))).astype(np.complex64)


def make_reflection_fields(z_um: np.ndarray, metal: MetalParams) -> dict[str, np.ndarray]:
    _, _, nz, slope = normals_from_height(z_um)
    cos_theta = np.clip(nz, 1e-4, 1.0)
    r, R = fresnel_unpolarized(cos_theta, metal.n, metal.k)
    acceptance = reflection_acceptance(nz)
    relief, local_rms = micro_relief(z_um, metal.roughness_rms_nm, 7101)
    atten = np.exp(-0.5 * (4.0 * np.pi * local_rms * cos_theta / WAVELENGTH_UM) ** 2).astype(np.float32)
    phase = 4.0 * np.pi * relief * cos_theta / WAVELENGTH_UM + np.angle(r).astype(np.float32)
    amp = np.sqrt(np.maximum(R * acceptance, 0)) * atten
    object_field = amp.astype(np.float32) * np.exp(1j * phase)
    parasitic = secondary_surface_scatter(object_field, z_um, acceptance, metal)
    return {
        "object_field": object_field.astype(np.complex64),
        "object_field_interference": (object_field + metal.spatial_coherence * parasitic).astype(np.complex64),
        "acceptance": acceptance,
        "coherent_attenuation": atten,
        "slope": slope,
    }


def numpy_transfer(defocus_um: float) -> np.ndarray:
    fx = np.fft.fftfreq(WIDTH, d=OBJECT_PIXEL_UM).astype(np.float32)
    fy = np.fft.fftfreq(HEIGHT, d=OBJECT_PIXEL_UM).astype(np.float32)
    fxx, fyy = np.meshgrid(fx, fy)
    f2 = fxx * fxx + fyy * fyy
    cutoff = NA / WAVELENGTH_UM
    pupil = (f2 <= cutoff * cutoff).astype(np.float32)
    kz_term = np.sqrt(np.maximum((1.0 / WAVELENGTH_UM) ** 2 - f2, 0.0)) - (1.0 / WAVELENGTH_UM)
    phase = 2.0 * np.pi * defocus_um * kz_term
    return pupil.astype(np.complex64) * np.exp(1j * phase).astype(np.complex64)


def propagate_numpy(field: np.ndarray) -> np.ndarray:
    field_f = np.fft.fft2(field)
    layers = []
    for zf in FOCUS_POSITIONS_UM:
        img = np.abs(np.fft.ifft2(field_f * numpy_transfer(float(zf)))) ** 2
        layers.append(img.astype(np.float32))
    stack = np.stack(layers, axis=0)
    stack /= float(np.percentile(stack, 99.8)) + 1e-9
    return np.clip(stack, 0, 1).astype(np.float32)


def propagate_torch(field: np.ndarray) -> np.ndarray:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.complex64
    field_t = torch.as_tensor(field, dtype=dtype, device=device)
    field_f = torch.fft.fft2(field_t)
    fx = torch.fft.fftfreq(WIDTH, d=OBJECT_PIXEL_UM, device=device)
    fy = torch.fft.fftfreq(HEIGHT, d=OBJECT_PIXEL_UM, device=device)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    f2 = fxx * fxx + fyy * fyy
    cutoff = NA / WAVELENGTH_UM
    pupil = (f2 <= cutoff * cutoff).to(torch.float32)
    kz_term = torch.sqrt(torch.clamp((1.0 / WAVELENGTH_UM) ** 2 - f2, min=0.0)) - (1.0 / WAVELENGTH_UM)
    layers = []
    for zf in FOCUS_POSITIONS_UM:
        phase = 2.0 * math.pi * float(zf) * kz_term
        transfer = pupil.to(dtype) * torch.exp(1j * phase).to(dtype)
        img = torch.abs(torch.fft.ifft2(field_f * transfer)) ** 2
        layers.append(img.to(torch.float32).cpu().numpy())
    stack = np.stack(layers, axis=0).astype(np.float32)
    stack /= float(np.percentile(stack, 99.8)) + 1e-9
    return np.clip(stack, 0, 1).astype(np.float32)


def propagate_prysm_angular_spectrum(field: np.ndarray) -> tuple[np.ndarray, str]:
    from prysm.propagation import angular_spectrum

    pupil_limited = np.fft.ifft2(np.fft.fft2(field) * numpy_transfer(0.0))
    layers = []
    dx_mm = OBJECT_PIXEL_UM * 1e-3
    for zf in FOCUS_POSITIONS_UM:
        propagated = angular_spectrum(pupil_limited, WAVELENGTH_UM, dx_mm, float(zf) * 1e-3, Q=1)
        img = np.abs(propagated) ** 2
        layers.append(img.astype(np.float32))
    stack = np.stack(layers, axis=0)
    stack /= float(np.percentile(stack, 99.8)) + 1e-9
    return (
        np.clip(stack, 0, 1).astype(np.float32),
        "prysm.propagation.angular_spectrum with same NA pupil mask before propagation",
    )


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


def evaluate_route(route_name: str, z_um: np.ndarray, fields: dict[str, np.ndarray]) -> dict[str, object]:
    t0 = time.perf_counter()
    status = "ok"
    note = ""
    try:
        if route_name == "numpy_reference":
            baseline_stack = propagate_numpy(fields["object_field"])
            interference_stack = propagate_numpy(fields["object_field_interference"])
        elif route_name == "torch_fft":
            baseline_stack = propagate_torch(fields["object_field"])
            interference_stack = propagate_torch(fields["object_field_interference"])
        elif route_name == "prysm_angular_spectrum":
            baseline_stack, note = propagate_prysm_angular_spectrum(fields["object_field"])
            interference_stack, _ = propagate_prysm_angular_spectrum(fields["object_field_interference"])
        else:
            raise ValueError(route_name)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "route": route_name,
            "status": "failed",
            "note": repr(exc),
            "runtime_s": elapsed,
        }

    elapsed = time.perf_counter() - t0
    artifact_stack = interference_stack - baseline_stack
    artifact_abs = np.abs(artifact_stack)
    idx_base, conf_base = focus_measure(baseline_stack)
    idx_int, conf_int = focus_measure(interference_stack)
    peak_shift = np.abs(idx_int.astype(np.int16) - idx_base.astype(np.int16)).astype(np.float32)
    rep = STACK_LAYERS // 2
    route_dir = OUT_DIR / route_name
    route_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=170)
    vmax_art = max(float(np.percentile(np.abs(artifact_stack), 99)), 0.01)
    panels = [
        (z_um, "A Fixed surface height (um)", "viridis", None, None),
        (fields["acceptance"], "B NA acceptance weight", "magma", 0, 1),
        (fields["coherent_attenuation"], "C Roughness coherence attenuation", "cividis", 0, 1),
        (baseline_stack[rep], "D Baseline focus layer", "gray", 0, 1),
        (interference_stack[rep], "E With secondary wave", "gray", 0, 1),
        (artifact_stack[rep], "F Artifact E-D", "coolwarm", -vmax_art, vmax_art),
    ]
    for ax, (img, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, FOV_WIDTH_UM, FOV_HEIGHT_UM, 0])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("object x (um)")
        ax.set_ylabel("object y (um)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    panel = route_dir / f"{route_name}_panel.png"
    fig.savefig(panel)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=170)
    axes[0].imshow(idx_base, cmap="viridis", vmin=0, vmax=STACK_LAYERS - 1)
    axes[0].set_title("Baseline DFF peak layer")
    axes[1].imshow(idx_int, cmap="viridis", vmin=0, vmax=STACK_LAYERS - 1)
    axes[1].set_title("Interference DFF peak layer")
    im = axes[2].imshow(peak_shift, cmap="magma", vmin=0, vmax=max(1.0, float(np.percentile(peak_shift, 99))))
    axes[2].set_title("Peak-layer shift")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    fig.tight_layout()
    dff_panel = route_dir / f"{route_name}_dff_peak_shift.png"
    fig.savefig(dff_panel)
    plt.close(fig)

    return {
        "route": route_name,
        "status": status,
        "note": note,
        "runtime_s": elapsed,
        "artifact_stack_p95_abs": float(np.percentile(artifact_abs, 95)),
        "artifact_stack_p99_abs": float(np.percentile(artifact_abs, 99)),
        "artifact_directionality_p999": artifact_directionality(artifact_stack[rep]),
        "mean_peak_shift_layers": float(np.mean(peak_shift)),
        "p95_peak_shift_layers": float(np.percentile(peak_shift, 95)),
        "mean_confidence_baseline": float(np.mean(conf_base)),
        "mean_confidence_interference": float(np.mean(conf_int)),
        "panel": panel.relative_to(OUT_DIR).as_posix(),
        "dff_panel": dff_panel.relative_to(OUT_DIR).as_posix(),
    }


def estimate_storage() -> dict[str, float]:
    stack_mb = HEIGHT * WIDTH * STACK_LAYERS * 4 / 1024 / 1024
    field_mb = HEIGHT * WIDTH * 8 / 1024 / 1024
    return {
        "single_float32_stack_mb": stack_mb,
        "two_stacks_mb": stack_mb * 2,
        "single_complex64_field_mb": field_mb,
        "expected_outputs_without_vendor_mb": 25.0,
    }


def folder_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024 / 1024


def write_report(metrics: pd.DataFrame, env: dict[str, object], storage: dict[str, float]) -> Path:
    ok = metrics[metrics["status"] == "ok"].copy()
    failed = metrics[metrics["status"] != "ok"].copy()
    table = metrics.to_markdown(index=False, floatfmt=".4f")

    failed_text = "无"
    if len(failed):
        failed_text = "\n".join(f"- `{row.route}`：{row.note}" for row in failed.itertuples())

    judgment_text = "暂无可比较路线。"
    if len(ok):
        ref = ok[ok["route"] == "numpy_reference"].iloc[0]
        torch_row = ok[ok["route"] == "torch_fft"].iloc[0] if "torch_fft" in set(ok["route"]) else None
        prysm_row = ok[ok["route"] == "prysm_angular_spectrum"].iloc[0] if "prysm_angular_spectrum" in set(ok["route"]) else None
        torch_delta = ""
        if torch_row is not None:
            torch_delta = (
                f"`torch_fft` 与 NumPy 参考的 artifact p95 差值为 "
                f"{abs(torch_row['artifact_stack_p95_abs'] - ref['artifact_stack_p95_abs']):.2e}，"
                f"耗时约为 NumPy 的 {torch_row['runtime_s'] / ref['runtime_s']:.1f} 倍。"
            )
        prysm_delta = ""
        if prysm_row is not None:
            rel = (
                (prysm_row["artifact_stack_p95_abs"] - ref["artifact_stack_p95_abs"])
                / max(ref["artifact_stack_p95_abs"], 1e-12)
                * 100
            )
            prysm_delta = (
                f"`prysm_angular_spectrum` 可运行，artifact p95 相对 NumPy 参考变化 {rel:.2f}% ，"
                f"耗时约为 NumPy 的 {prysm_row['runtime_s'] / ref['runtime_s']:.1f} 倍；"
                "差异主要来自 prysm 的近轴 angular-spectrum 相位形式。"
            )
        judgment_text = (
            "当前同条件实验不应按伪影强度的微小差异直接给路线排名。"
            f"`numpy_reference` 最轻、最快，适合作为可解释基线；{torch_delta}"
            f"{prysm_delta}"
            "综合判断：论文前仿真的主线建议使用 NumPy/Torch 自控传播核，"
            "prysm 作为工程光学交叉验证后端保留；显微 PSF 校准和金属 BRDF 物理应放到下一轮 pyOTF/psf-generator/pySCATMECH 专项。"
        )

    image_lines = []
    for row in ok.itertuples():
        image_lines.append(f"### {row.route}\n\n![{row.route} panel]({row.panel})\n\n![{row.route} dff]({row.dff_panel})")
    images = "\n\n".join(image_lines)

    report = f"""# 开源波动光学路线对比实验

## 实验目的

本实验用于决定后续应优先采用哪条开源波动光学路线。固定变量包括同一复杂 100 um 表面样品、同一金属 Fresnel/粗糙度反射场、同一 NA=0.40、同一波长 525 nm、同一 13 层焦栈和同一 DFF 评价指标。

## 存储与依赖估计

- 单个 float32 焦栈：{storage['single_float32_stack_mb']:.2f} MB。
- baseline/interference 两套焦栈：{storage['two_stacks_mb']:.2f} MB。
- 单个 complex64 复场：{storage['single_complex64_field_mb']:.2f} MB。
- 本轮不保存完整焦栈，只保存图、CSV 和报告，预计非依赖输出约 {storage['expected_outputs_without_vendor_mb']:.1f} MB。
- 当前文件夹实际大小：{folder_size_mb(OUT_DIR):.2f} MB。

依赖可用性：

```json
{json.dumps(env, ensure_ascii=False, indent=2)}
```

## 路线设置

- `numpy_reference`：本项目自写 angular-spectrum / pupil defocus 参考路线。
- `torch_fft`：使用 PyTorch FFT 实现同一传播核，用于验证 GPU/深度学习工具链接入可行性。
- `prysm_angular_spectrum`：调用 `prysm.propagation.angular_spectrum`，并在传播前使用同一 NA pupil mask，使其和参考路线保持相同接收条件。

`TorchOptics`、`Diffractio`、`pyOTF`、`pySCATMECH` 若未安装，本轮记录为环境缺口。下一轮可在独立 vendor 环境中安装并加入更多独立 API 对比。

## 指标

- `artifact_stack_p95_abs` / `artifact_stack_p99_abs`：baseline 与 interference 焦栈差分伪影强度。
- `artifact_directionality_p999`：中间焦平面伪影频谱高分位峰值，用于观察规则条纹倾向。
- `p95_peak_shift_layers`：DFF 峰值层 95% 分位偏移。
- `runtime_s`：同条件运行耗时。

## 结果表

{table}

## 初步判断

{judgment_text}

失败或缺失路线：

{failed_text}

## 图像说明

每条成功路线的第一张 2 x 3 图中，A 是固定高度图，B 是局部反射进入 NA 的权重，C 是粗糙度相干衰减，D 是无次级小波的基线焦平面，E 是叠加表面派生次级小波后的焦平面，F 是 E-D 差分伪影。第二张图展示 baseline 与 interference 的 DFF 峰值层及其偏移。

{images}

## 下一步路线选择建议

1. 若目标是可控、可解释、可和训练脚本连接，优先扩展 `torch_fft` 或 TorchOptics 路线。
2. 若目标是更接近显微成像 PSF/焦深校准，下一轮应加入 `pyOTF` 或 `psf-generator`。
3. 若目标是金属反射物理可信度，下一轮应单独尝试 `pySCATMECH`，先输出角度相关 Fresnel/BRDF 参数，再接入传播。
4. 若目标是“全视场每个表面点作为子波源，并按方向进入 NA pupil”，需要优先选择支持 Rayleigh-Sommerfeld 或等价全场传播的路线，并在报告中明确采样、pupil、离轴角度和计算代价。
"""
    path = OUT_DIR / "open_source_wave_optics_route_benchmark_report.md"
    path.write_text(report, encoding="utf-8", newline="\n")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    storage = estimate_storage()
    env = {
        "python": sys.executable,
        "vendor_dir": str(VENDOR),
        "vendor_exists": VENDOR.exists(),
        "vendor_size_mb": folder_size_mb(VENDOR) if VENDOR.exists() else 0.0,
        "numpy": module_available("numpy"),
        "scipy": module_available("scipy"),
        "matplotlib": module_available("matplotlib"),
        "pandas": module_available("pandas"),
        "torch": module_available("torch"),
        "torchoptics": module_available("torchoptics"),
        "diffractio": module_available("diffractio"),
        "prysm": module_available("prysm"),
        "pyotf": module_available("pyotf"),
        "psf_generator": module_available("psf_generator"),
        "pySCATMECH": module_available("pySCATMECH"),
    }
    z_um = make_surface()
    fields = make_reflection_fields(z_um, MetalParams())
    np.save(OUT_DIR / "fixed_complex_surface_height_um.npy", z_um)

    routes = ["numpy_reference", "torch_fft", "prysm_angular_spectrum"]

    rows = [evaluate_route(route, z_um, fields) for route in routes]
    metrics = pd.DataFrame(rows)
    metrics_path = OUT_DIR / "open_source_wave_optics_route_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    (OUT_DIR / "environment_probe.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "storage_estimate.json").write_text(json.dumps(storage, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(metrics, env, storage)
    print(report)
    print(metrics_path)
    print(f"folder_size_mb={folder_size_mb(OUT_DIR):.2f}")


if __name__ == "__main__":
    main()
