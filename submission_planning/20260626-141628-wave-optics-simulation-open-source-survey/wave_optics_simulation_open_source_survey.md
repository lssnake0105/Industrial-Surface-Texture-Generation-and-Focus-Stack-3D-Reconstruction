# 波动光学显微成像与金属反射仿真开源资料调研

## 结论

公开资料里可以找到足够多的基础工具，但没有一个项目能开箱即用地完整覆盖“金属粗糙表面反射复场 + Olympus 20X 显微物镜 + 525 nm LED 部分相干 + 多焦平面焦栈 + DFF 伪峰分析”。更可行的路线是组合使用：

- **波动传播/显微成像骨架**：TorchOptics、Diffractio、prysm。
- **显微 PSF / OTF / 焦深校准**：pyOTF、psf-generator、PSFmodels。
- **金属反射、BRDF、粗糙表面散射**：pySCATMECH / SCATMECH。
- **严格电磁散射上限验证**：Meep、RCWA/S4、SMUTHI。

对本项目投稿前最实用的组合是：

```text
surface_sample_generator
-> Fresnel / roughness / microfacet reflection field
-> TorchOptics 或 Diffractio 标量传播
-> pyOTF / psf-generator 校准焦深和 PSF
-> pySCATMECH 辅助校验金属 BRDF / 粗糙散射参数
```

## 文献与项目矩阵

| 资源 | 类型 | 能解决什么 | 对本项目价值 | 局限 | 推荐级别 |
|---|---|---|---|---|---|
| [TorchOptics](https://pypi.org/project/torchoptics/) | Python / PyTorch / Fourier optics | 可微分波动光学、GPU、Zernike、偏振与空间相干 | 适合实现端到端焦栈仿真，并和深度学习训练衔接 | 需要自己写金属表面反射场和显微物镜几何 | 高 |
| [Diffractio](https://github.com/aocg-ucm/diffractio) | Python / scalar & vector diffraction | FFT、Rayleigh-Sommerfeld、BPM、WPM、CZT，支持标量和矢量模块 | 适合快速搭建波前传播、离焦传播、干涉场模拟 | API 与项目训练管线需要额外适配 | 高 |
| [prysm](https://github.com/brandondube/prysm) | Python / numerical optics | pupil-to-focus、focus-to-pupil、free-space angular spectrum、Zernike、MTF | 适合做显微物镜 pupil、焦深、像差、MTF 快速原型 | 偏工程光学，金属反射/粗糙散射需自建 | 高 |
| [POPPY](https://poppy-optics.readthedocs.io/en/latest/) | Python / physical optics | Fraunhofer/Fresnel 传播、pupil/image planes、Zernike 像差、多波长 PSF | 可作为物理光学传播参考实现 | 主要面向天文望远镜，明确不做矢量传播与探测器噪声 | 中 |
| [pyOTF](https://github.com/david-hoffman/pyotf) | Python / microscope OTF/PSF | 显微镜 OTF/PSF 模拟 | 可用于校准 NA、波长、焦深、PSF 与 DFF 焦曲线宽度 | 偏 PSF/OTF 计算，无法直接模拟金属反射复场 | 高 |
| [psf-generator](https://github.com/Biomedical-Imaging-Group/psf_generator) | PyTorch / microscope PSF | 高性能显微 PSF 物理模型 | 可用于生成精确显微 PSF 或校验自写 defocus phase | 重点是点扩散函数，需外接表面反射场 | 高 |
| [PSFmodels](https://github.com/tlambert03/PSFmodels) | Python bindings / scalar & vector PSF | 标量和矢量显微 PSF 模型 | 可作为 pyOTF 的补充，特别是验证焦深/轴向 PSF | 同样偏点源模型 | 中 |
| [pySCATMECH](https://github.com/usnistgov/pySCATMECH) | Python interface / NIST SCATMECH | Fresnel、BRDF、粗糙表面散射、偏振、薄膜、RCW | 最适合补足金属表面反射物理；可用于拟合/校验 roughness BRDF 参数 | 安装和模型选择可能比纯 Python 复杂 | 高 |
| [SCATMECH](https://pages.nist.gov/SCATMECH/docs/index.htm) | C++ / polarized light scattering | 表面散射、BRDF、粒子散射、光栅反射/透射/衍射 | 作为金属反射与粗糙散射理论/代码基准 | C++ 库，直接集成成本更高 | 中高 |
| [Meep](https://meep.readthedocs.io/en/latest/FAQ/) | FDTD / full electromagnetic simulation | 任意结构 Maxwell 方程时域仿真，支持材料色散等 | 可作为小 ROI / 单一微结构的严格电磁验证 | 对 960×540 显微视场代价很高，不适合整幅焦栈 | 中 |
| [rcwa](https://pypi.org/project/rcwa/) / [S4](https://fan.group.stanford.edu/S4) | RCWA / Fourier modal method | 薄膜、周期结构、1D/2D 光栅、金属镜反射 | 对周期加工纹理、条纹粗糙度和薄膜反射有帮助 | 要求周期性或层状结构，复杂随机表面不方便 | 中 |
| [SMUTHI](https://github.com/KMCzajkowski/smuthi) | T-matrix + scattering matrix | 层状介质附近多粒子散射、近场/远场 | 可用于颗粒/污染物/表面附着物导致的散射伪影 | 与连续粗糙金属表面焦栈距离较远 | 低中 |
| [diffractsim](https://github.com/rafael-fuente/diffractsim) | Python/JAX scalar diffraction | 直观物理光学可视化、JAX 后端 | 适合教学和快速可视化干涉/衍射 | 不直接面向显微物镜或金属 BRDF | 中 |

## 推荐路线

### 路线 A：最适合投稿前快速落地

```text
现有 surface_sample_generator
-> 自写金属 Fresnel + roughness coherence + NA acceptance
-> Diffractio 或 TorchOptics 做标量传播
-> 扫描 z_f 生成 17 层焦栈
-> 用 pyOTF / psf-generator 校准轴向 PSF 与焦深
```

优势：实现速度较快，可解释性强，能直接服务论文中的 Simulation-to-Real 主线。  
风险：部分相干 LED 和真实物镜 pupil 仍需要近似。

### 路线 B：金属反射物理更严谨

```text
现有 surface_sample_generator
-> pySCATMECH 查询/拟合 Fresnel、BRDF、粗糙散射
-> 生成角度相关反射振幅和偏振项
-> 接入 TorchOptics / Diffractio 传播
```

优势：金属反射与粗糙度参数更有文献/开源模型支撑。  
风险：BRDF 到复场传播的接口需要自己设计，不能只拿强度 BRDF 替代相位。

### 路线 C：严格电磁小样本验证

```text
选取 5-20 um 小 ROI
-> Meep 或 RCWA/S4
-> 输入金属材料 n,k 和周期/局部表面结构
-> 输出反射场/散射角谱
-> 与标量模型的反射场做数量级对比
```

优势：物理严谨度最高。  
风险：计算代价高，只适合小 ROI 或周期结构，不适合整幅 960×540 焦栈。

## 对焦深/景深复现的建议

完整波动光学焦栈不应只靠经验 blur radius。推荐用 pupil defocus phase 扫描焦平面：

$$
P_z(f_x,f_y)=P_0(f_x,f_y)\exp[i\Phi_{defocus}(f_x,f_y;z)].
$$

标量角谱近似：

$$
\Phi_{defocus}(f_x,f_y;z)
=
2\pi z
\left[
\sqrt{\frac{1}{\lambda^2}-f_x^2-f_y^2}
-\frac{1}{\lambda}
\right].
$$

小角度近似：

$$
\Phi_{defocus}\approx -\pi\lambda z(f_x^2+f_y^2).
$$

对 525 nm、NA=0.40：

$$
\frac{\lambda}{NA^2}\approx 3.28\,\mu m,\qquad
\frac{2\lambda}{NA^2}\approx 6.56\,\mu m.
$$

因此，焦栈层间距应围绕 3-6 µm 的轴向尺度设计或至少在报告中解释。如果真实焦层步长远大于这个尺度，DFF 峰会有欠采样风险；如果步长小于这个尺度，焦点曲线会更接近连续波动光学响应。

## 当前项目可直接采用的实现方案

建议下一步在当前任务文件夹中新增：

```text
full_wave_focus_stack_simulator.py
```

模块结构：

1. `make_surface()`：调用现有表面生成器。
2. `make_reflection_field()`：Fresnel + roughness coherence + NA acceptance。
3. `make_pupil()`：NA cutoff + optional Zernike。
4. `propagate_to_focus(z_f)`：pupil defocus phase + FFT propagation。
5. `source_average()`：多波长/多照明角度加权，近似 LED 部分相干。
6. `sensor_integrate()`：超采样后 block average 到相机像素。
7. `export_focus_stack()`：输出 `.npy`、代表层图、焦点曲线和 DFF 峰偏移图。

## 来源链接

- TorchOptics: https://pypi.org/project/torchoptics/
- Diffractio: https://github.com/aocg-ucm/diffractio
- prysm: https://github.com/brandondube/prysm
- POPPY: https://poppy-optics.readthedocs.io/en/latest/
- pyOTF: https://github.com/david-hoffman/pyotf
- psf-generator: https://github.com/Biomedical-Imaging-Group/psf_generator
- pySCATMECH: https://github.com/usnistgov/pySCATMECH
- SCATMECH docs: https://pages.nist.gov/SCATMECH/docs/index.htm
- Meep: https://meep.readthedocs.io/en/latest/FAQ/
- rcwa: https://pypi.org/project/rcwa/
- S4: https://fan.group.stanford.edu/S4
- SMUTHI: https://github.com/KMCzajkowski/smuthi
- diffractsim: https://github.com/rafael-fuente/diffractsim
