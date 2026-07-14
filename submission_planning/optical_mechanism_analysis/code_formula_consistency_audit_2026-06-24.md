# 模拟与重建代码的成像公式一致性校对

日期：2026-06-24  
项目：SRTP 反光工业表面缺陷焦栈三维重建  
对象：`src/` 主代码与 `submission_planning/tools/` 中和模拟、重建、反光风险、focus confidence、prior loss 相关的脚本  
参照：`literature_matrix_focus_imaging_mechanism_analysis_2026-06-23.md` 中整理的 thin-lens CoC、PSF/blur radius、DFF focus measure、NA/microfacet、DFV probability / uncertainty 思路。

## 0. 总结论

当前代码与主流对焦成像原理在“定性机制”上基本一致：高度差决定焦向清晰度，DFF 通过 Laplacian/Tenengrad 类 focus measure 选择最佳焦层，反光风险来自局部法线、镜面反射、饱和高亮和离焦 halo，网络输入也包含焦栈、焦向差分、DFF/GADFF 先验与置信度。

但当前仿真主线并未实现 DDFS / Focus on Defocus / DEReD 文献中的物理 CoC 公式：

$$
c=b\frac{|d-d_f|}{d}\frac{f^2}{N(d_f-f)},\qquad
\sigma=\frac{1}{2p}\frac{|d_o-F|}{d_o}\frac{f^2}{N(F-f)}.
$$

代码实际使用的是“归一化高度差或微米高度差 -> Gaussian focus weight -> 固定尺度 blur 图像混合”的工程近似。它可以支撑 simulation-to-real 方向的受控数据生成和机制诊断，但论文方法部分不能把当前渲染写成严格 thin-lens CoC renderer。更准确的表述应是：**a focus-response simulator with CoC-inspired Gaussian defocus weighting and glare-aware degradation**。

另一个需要优先校对的是高度方向约定。合成主线使用 `focus_index_to_relative_height()` 将第 0 层映射为高平面；部分真实样本脚本直接使用 `idx/(K-1)`，可能与项目统一约定相反。若真实结果只做相对可视化，问题可控；若写入论文定量或与模型输入合并，需要统一。

## 1. 代码中的核心成像链条

### 1.1 高度图与焦层坐标

项目统一约定写在 `src/dff_depth_direction.py`：

```python
STACK_ORDER = "high_to_low"
relative_layer = index.astype(np.float32) / float(denom)
return 1.0 - relative_layer
```

对应公式为：

$$
\hat H(x)=1-\frac{\hat k(x)}{K-1}.
$$

合成焦层坐标由：

```python
focus_positions_um(height_range_um, layer_count)
```

生成，本质为：

$$
z_k = H_{\max}\left(1-\frac{k}{K-1}\right).
$$

这与项目文档中“第 0 张图是高焦平面，后续向下扫描”的约定一致。

### 1.2 高度图到法线

主仿真使用：

```python
dy_um, dx_um = np.gradient(z_um, camera.object_pixel_um, camera.object_pixel_um)
normals = np.dstack((-dx_um, -dy_um, np.ones_like(z_um)))
normals /= np.linalg.norm(normals, axis=2, keepdims=True)
```

对应公式为：

$$
\mathbf n(x,y)=
\frac{(-\partial z/\partial x,\ -\partial z/\partial y,\ 1)}
{\sqrt{(\partial z/\partial x)^2+(\partial z/\partial y)^2+1}}.
$$

这一点与主流 surface normal from height field 一致，并且单位处理比早期 prototype 更清楚，因为 `z_um` 与 `object_pixel_um` 同为微米尺度。

### 1.3 反光 / glare 生成

主仿真 `render_coaxial_metal()` 中使用：

```python
nz = normals[:, :, 2]
diffuse = albedo * (0.36 + 0.64 * nz)
shininess = 38 + 240 * (1 - roughness) ** 2
specular = scenario.f0 * np.power(nz, shininess) * (0.55 + 0.45 * nz)
radiance = normalize01(0.10 + 0.52 * diffuse + 2.2 * specular + edge_boost)
hard_seed = np.where(specular > np.percentile(specular, 97.5), specular, 0)
bloom = GaussianBlur(hard_seed)
glare = normalize01(hard_seed + 1.8 * bloom)
```

它不是完整 Cook-Torrance BRDF：

$$
f_r=\frac{D(\mathbf h)F(\mathbf v,\mathbf h)G(\mathbf l,\mathbf v,\mathbf h)}
{4(\mathbf n\cdot \mathbf l)(\mathbf n\cdot \mathbf v)}.
$$

它更接近同轴视角下的简化 Phong / Blinn-Phong 风格近似：

$$
R_{\mathrm{spec}}(x)\propto F_0\,n_z(x)^{s(x)}(0.55+0.45n_z(x)),
$$

其中 $s(x)$ 由 roughness 控制。这个近似与“平缓局部法线更容易在同轴成像中形成强反光”的故事一致，但没有显式使用：

$$
\mathbf r=2(\mathbf n\cdot\mathbf l)\mathbf n-\mathbf l,\qquad
\arccos(\mathbf r\cdot\mathbf v)\leq \arcsin(\mathrm{NA}/n).
$$

`submission_planning/tools/glare_risk_microfacet_demo.py` 和相关研究脚本实现了这个 NA 接收锥版本：

```python
light = [sin(tilt), 0, -cos(tilt)]
view = [0, 0, 1]
reflected = light - 2.0 * dot * n
theta = arccos(reflected · view)
acceptance = arcsin(na)
risk = max(0.65 * hard, exp(-(theta/acceptance)^2))
```

这与文献机理更一致。建议论文中把 NA/microfacet 公式作为物理解释和辅助仿真，把主数据生成器的 glare 写成简化同轴 specular-bloom model。

### 1.4 焦栈渲染与 blur 半径

主仿真 `synthesize_focus_stack()` 使用：

```python
dist = np.abs(z_um - focus_z)
focus_weight = np.exp(-0.5 * (dist / dof_um) ** 2)
mid_weight = np.exp(-0.5 * (dist / (dof_um * 2.4)) ** 2)
base = focus_weight * sharp + (1 - focus_weight) * (mid_weight * blur_soft + (1 - mid_weight) * blur_mid)
far = dist > dof_um * 2.8
base[far] = 0.65 * base[far] + 0.35 * blur_heavy[far]
```

其中：

```python
dof_um = max(45.0, scenario.depth_range_um / 14.0)
blur_soft = GaussianBlur(sharp, sigma=max(0.8, 0.0025*max(h,w)))
blur_mid = GaussianBlur(sharp, sigma=max(1.5, 0.0060*max(h,w)))
blur_heavy = GaussianBlur(sharp, sigma=max(2.8, 0.0130*max(h,w)))
```

对应的工程模型是：

$$
w_f(x;k)=\exp\left[-\frac{1}{2}
\left(\frac{|H(x)-z_k|}{\mathrm{DoF}}\right)^2\right],
$$

$$
I_k(x)=w_f\,I_{\mathrm{sharp}}(x)+(1-w_f)
\left(w_m\,G_{\sigma_s}(I_{\mathrm{sharp}})(x)+(1-w_m)G_{\sigma_m}(I_{\mathrm{sharp}})(x)\right),
$$

并在远离焦层时混入 heavy blur。

与主流 thin-lens CoC 的关系：

$$
c(d,d_f)=b\frac{|d-d_f|}{d}\frac{f^2}{N(d_f-f)}
$$

当前代码只保留了“$|H-z_k|$ 越大，越模糊”的单调关系，没有显式使用 $d$ 分母、$f^2$、$N$、$d_f-f$ 或像素尺寸换算。因此：

| 项目 | 主流 CoC | 当前代码 | 一致性 |
|---|---|---|---|
| 深度差影响模糊 | $|d-d_f|$ 增大，CoC 增大 | `dist` 增大，`focus_weight` 下降 | 一致 |
| 相机参数影响 | $f,N,d_f,p,b$ 显式参与 | `f_number` 存在但未参与渲染 | 不一致 |
| blur 半径局部变化 | 每个像素由 $d(x)$ 决定 CoC | 使用固定 `blur_soft/mid/heavy` 图混合，非局部半径卷积 | 部分近似 |
| 焦内峰形 | CoC 最小时最清晰 | Gaussian focus weight 在 $H=z_k$ 最大 | 一致 |
| 物理单位 | 米/像素尺度可校准 | 主要是微米高度差与经验 DoF | 工程近似 |

结论：该仿真可以写作 defocus-inspired simulator，不能写作 calibrated thin-lens CoC renderer。

### 1.5 glare / stray / saturation 退化

主仿真还加入：

```python
layer_glare = glare * (0.65 + 0.30 * focus_weight) * random_gain
layer_stray = stray * random_gain
image = clip(base + 0.62 * layer_glare + layer_stray, 0, 1)
prnu, row_bias, col_bias, shot noise, read noise, 8-bit quantization
```

这与主流论文中常见的理想 defocus model 有明显差异。文献中的简化模型多写作：

$$
I_k=I_{\mathrm{AiF}}\ast h_{c_k}+\eta.
$$

本项目更接近：

$$
I_k=\mathrm{clip}\left[
\mathcal D_k(T+R_{\mathrm{spec}})
+G_k+S_k+N_k
\right],
$$

其中 $\mathcal D_k$ 是工程离焦混合，$G_k$ 是 layer glare，$S_k$ 是 stray/ghost，$N_k$ 是 PRNU/DSNU/Poisson-Gaussian 噪声。这个差异反而是项目应用侧的特色：它更贴近反光工业表面的图像退化，但需要承认其参数是经验式。

### 1.6 DFF / GADFF 重建

主 DFF 使用：

```python
lap = abs(Laplacian(GaussianBlur(layer)))
tenengrad = Sobel_x^2 + Sobel_y^2
fm = boxFilter(lap) + 0.0018 * boxFilter(tenengrad)
idx = argmax(focus, axis=0)
depth = focus_index_to_relative_height(idx, K)
```

对应公式为：

$$
F_k(x)=\mathrm{Box}\left(|\nabla^2 G(I_k)|\right)
+\lambda\,\mathrm{Box}\left(\|\nabla G(I_k)\|^2\right),
$$

$$
\hat H(x)=1-\frac{\arg\max_k F_k(x)}{K-1}.
$$

这与 Pertuz / Nayar 的 focus-measure DFF 一致，是项目中最贴近经典 SFF/DFF 的部分。

GADFF 使用：

```python
focus = focus * clip(1.0 - 0.70 * risk_layers, 0.20, 1.0)
```

对应：

$$
F'_k(x)=F_k(x)\cdot \mathrm{clip}(1-0.70R_k(x),0.20,1.0).
$$

这与“高亮/饱和区域可能产生伪清晰峰，因此应降低其 focus response 权重”的机理一致，但它是启发式风险降权，并非由 BRDF 或 CoC 直接推导。

### 1.7 focus confidence

主代码使用：

```python
confidence = (peak - second) / (peak + 1e-6)
confidence = clip(confidence / percentile(confidence, 98.5), 0, 1)
```

真实样本原型还使用 peak sharpness：

```python
ratio_conf = (peak - second) / peak
sharp_conf = (2*peak_f - prev_f - next_f) / peak_f
confidence = (0.46*ratio_conf + 0.34*sharp_conf + 0.20*texture) * (1 - risk penalty)
```

这些与 DFV 的 probability uncertainty 思路方向一致，但不是概率分布标准差：

$$
\phi_j=\sqrt{\sum_i p_j^i(l_i-\hat d_j)^2}.
$$

当前 confidence 更接近峰值间隔与局部尖锐度指标：

$$
C_{\mathrm{margin}}=\frac{F_{(1)}-F_{(2)}}{F_{(1)}+\epsilon}.
$$

它适合写作 focus-response reliability indicator。若想与 DFV 更一致，后续可以把 focus volume 归一化成：

$$
p_i(x)=\frac{\exp(\beta F_i(x))}{\sum_j \exp(\beta F_j(x))}
$$

再计算期望深度和方差。

### 1.8 网络输入与 prior loss

`train_focus_resunet_loss_experiment.py` 中模型输入为：

```python
stack, diffs=np.diff(stack), priors=[risk, dff, conf, gadff, ga_conf]
```

这对应：

$$
X=\mathrm{concat}(I_{1:K},\Delta I_{1:K-1},R,P_{\mathrm{DFF}},C_{\mathrm{DFF}},P_{\mathrm{GADFF}},C_{\mathrm{GADFF}}).
$$

旧版 `HybridDFFLoss` 使用：

```python
glare_weight = 1.0 + 0.80 * risk
data = mean(glare_weight * charbonnier(pred - target))
prior_weight = (0.65*conf+0.35*ga_conf) * (1-risk)^1.5
prior_target = 0.45*dff + 0.55*gadff
```

对应：

$$
\mathcal L_{\mathrm{data}}=
\mathbb E[(1+0.80R)\rho(\hat H-H)],
$$

$$
W_{\mathrm{prior}}=(0.65C_{\mathrm{DFF}}+0.35C_{\mathrm{GADFF}})(1-R)^{1.5}.
$$

后续对比脚本中更稳妥的 confidence-gated 版本为：

```python
confidence_prior_weight = focus_conf.pow(1.5) * (1.0 - 0.45 * risk)
actual_data_weight = ones_like(...)
```

对应：

$$
W_{\mathrm{prior}}=
\mathrm{clip}\left(C_{\mathrm{focus}}^{1.5}(1-0.45R),0.02,1.0\right),
$$

且不再直接把 high-risk 区域的数据项加权。这个版本更符合当前机理证据：focus confidence 是更稳定的 prior reliability 信号，risk 应保持弱条件项或诊断项。

## 2. 与主流成像原理的一致性

### 2.1 一致部分

| 代码模块 | 公式/机制 | 与主流原理关系 |
|---|---|---|
| `focus_index_to_relative_height` | $\hat H=1-\hat k/(K-1)$ | 符合 DFF 从最佳焦层映射深度的基本逻辑 |
| `surface_normals_from_um` | $\mathbf n=(-z_x,-z_y,1)/\|\cdot\|$ | 符合高度场法线定义 |
| `focus_weight` | $\exp[-0.5(|H-z_k|/\mathrm{DoF})^2]$ | 保留对焦层附近最清晰的单峰响应 |
| `focus_maps_from_stack` | Laplacian + Tenengrad | 符合经典 focus measure family |
| `dff_depth` | $\arg\max_k F_k$ | 符合 Nayar/Pertuz 的传统 DFF |
| `specular_risk` tools | $\angle(\mathbf r,\mathbf v)\leq\arcsin(\mathrm{NA})$ | 与 NA 接收锥物理解释一致 |
| `risk_layers` | 高亮/局部高亮/bloom/saturation | 与反光伪峰和饱和裁剪问题一致 |
| `confidence` | peak margin / sharpness | 与 DFF reliability / uncertainty 思路一致 |
| `np.diff(stack)` | 焦向差分 | 与 DFV 的 differential focus volume 思路一致 |

### 2.2 差异部分

| 差异点 | 当前代码 | 主流物理模型 | 影响 |
|---|---|---|---|
| CoC 公式 | 未显式计算 CoC | $c=b|d-d_f|f^2/[dN(d_f-f)]$ | 不能声称物理标定 CoC 渲染 |
| blur radius | 固定 `blur_soft/mid/heavy` 混合，部分 tools 用全图 mean defocus 控制 radius | 每像素 CoC 半径随深度变化 | 局部深度边界的真实 PSF 过渡不够物理 |
| f-number | `CameraConfig.f_number` 存在但主渲染未使用 | $N$ 直接控制 CoC | 不能讨论 f-number 参数扫描，除非新增实现 |
| NA | 主仿真未显式使用 NA，tools 使用 NA | NA 控制接收锥 | 主实验中的 risk 主要是强度启发式 |
| BRDF | 使用 $n_z^{shininess}$ + bloom | Cook-Torrance microfacet | 可解释但非完整材质模型 |
| 光照方向 | 主仿真近似同轴 $n_z$，prototype 有随机 light | 需要明确 $\mathbf l,\mathbf v$ | 写作时应避免过度声称真实同轴照明标定 |
| 真实样本高度方向 | 部分脚本直接 `idx/(K-1)` | 项目统一 high_to_low 为 $1-idx/(K-1)$ | 需统一或说明只是可视化相对深度 |
| confidence | 峰值间隔启发式 | DFV 可用概率期望与方差 | 当前可写 reliability indicator，不写 calibrated uncertainty |

## 3. 高优先级校对点

### 3.1 真实样本脚本的高度方向需要统一

`src/real_sample_glare_prior_fusion_validation.py` 中：

```python
depth = idx.astype(np.float32) / max(focus.shape[0] - 1, 1)
```

没有调用 `focus_index_to_relative_height()`。这与 `src/dff_depth_direction.py` 的 high-to-low 约定不同。若该脚本输出只用于相对纹理可视化，风险较低；若要进入论文与合成结果同一套方法说明，应改成：

$$
\hat H=1-\frac{\hat k}{K-1}
$$

或在报告中声明真实样本图只使用相对层号显示，并已按图示方向统一。

### 3.2 论文中不能把主仿真写成严格 CoC 渲染

建议写法：

> We use a CoC-inspired focus-response simulator in which axial defocus is modeled by a Gaussian function of the distance between the surface height and the focal plane, while glare, bloom, veiling stray light, sensor noise, and clipping are added to reproduce reflective focus-stack degradations.

不建议写法：

> We render focal stacks using the thin-lens CoC formula.

除非后续把 CoC 公式真正加入 `synthesize_focus_stack()`。

### 3.3 `CameraConfig.f_number` 当前没有进入成像公式

`CameraConfig` 里有 `f_number=5.6`，但主渲染的 blur 和 DoF 没有使用它。若论文要写 f-number 或 aperture 对结果的影响，需要新增：

$$
\sigma_k(x)=\gamma\frac{|d(x)-d_{f,k}|}{d(x)}
\frac{f^2}{N(d_{f,k}-f)}
$$

或至少将：

$$
\mathrm{DoF}\propto N
$$

显式接入 `dof_um`。

### 3.4 tools 中的 `halo_radius` 使用全图平均 defocus

`focus_confidence_risk_study.py`、`superres_downsample_glare_probe.py`、`superres_dff_resolution_study.py` 里有：

```python
halo_radius = 1.0 + 6.0 * mean(defocus)
radius = blur_base + blur_gain * mean(defocus)
```

这适合做轻量机制 demo，但不是局部 PSF。写报告时应把这些脚本称为 diagnostic toy simulation / mechanism probe，不作为最终物理渲染器。

### 3.5 旧版 loss 与当前最佳叙事存在差异

旧版 `HybridDFFLoss` 直接上调 high-risk 区域 supervised data loss：

$$
\mathcal L_{\mathrm{data}}=\mathbb E[(1+0.80R)\rho(\hat H-H)].
$$

后续证据更支持取消直接 glare data upweight，把机制集中在 confidence-gated prior consistency。投稿时应以 ABL-07 / confidence-gated 版本为主线，旧版 loss 可作为 development history 或 baseline，不宜作为最终方法主表达。

## 4. 建议的后续修正

### 4.1 最小代码修正

1. 在真实样本 `dff_depth_from_focus()` 中调用统一的 `focus_index_to_relative_height()`，或输出字段明确命名为 `relative_layer_index`。
2. 在 `synthesize_focus_stack()` 的 docstring 中明确：当前是 CoC-inspired Gaussian focus response simulator。
3. 在 `CameraConfig` 中若保留 `f_number`，要么接入 `dof_um`，要么在报告中不要声称它参与成像。

### 4.2 中等强度物理升级

在主仿真中新增可选 CoC 半径：

$$
\sigma_k(x)=\sigma_0+\gamma\frac{|z(x)-z_k|}{z_{\mathrm{offset}}+z(x)}
\frac{f^2}{N(z_k+z_{\mathrm{offset}}-f)}.
$$

由于显微高度范围相对工作距离很小，可先采用局部线性化：

$$
\sigma_k(x)=\sigma_0+\gamma\frac{|H(x)-z_k|}{\Delta z}.
$$

再用分桶或多尺度混合实现空间变化 blur，避免每像素卷积成本过高。

### 4.3 论文表述建议

建议将当前代码定位为三层模型：

1. **Physical proxy layer**：height -> normal -> simplified coaxial specular / glare / saturation risk。
2. **Focus-response layer**：height-focal-plane distance -> Gaussian focus weight -> defocused image mixture。
3. **Reconstruction layer**：Laplacian/Tenengrad DFF -> glare-aware weighted DFF -> confidence-gated prior network。

这样既保留物理合理性，也避免把工程参数说成严格光学标定。

## 5. 可直接放入论文方法边界的表述

当前可安全写：

> The simulator does not aim to be a fully calibrated lens renderer. Instead, it preserves the causal structure required by the task: surface height determines axial focus response, local normals and reflective priors generate glare-prone regions, and saturation/stray-light degradations create focus-measure failures observed in real stacks.

当前应避免写：

> The proposed simulator computes the physical CoC of each pixel using focal length, f-number, focus distance, and pixel size.

只有在主代码显式加入 CoC 后，后一类表述才成立。

## 6. 代码证据索引

| 机制 | 文件 | 代码位置 |
|---|---|---|
| 高低焦层方向约定 | `src/dff_depth_direction.py` | lines 6-40 |
| 高度图法线 | `src/simulate_antiglare_highres_samples.py` | lines 336-340 |
| 同轴金属反光近似 | `src/simulate_antiglare_highres_samples.py` | lines 343-367 |
| 焦栈渲染与 Gaussian focus weight | `src/simulate_antiglare_highres_samples.py` | lines 389-442 |
| DFF/GADFF | `src/simulate_antiglare_highres_samples.py` | lines 445-470 |
| 模型输入特征 | `src/simulate_antiglare_highres_samples.py` | lines 473-481 |
| 真实样本 DFF 方向差异 | `src/real_sample_glare_prior_fusion_validation.py` | lines 134-144 |
| 真实样本 risk layers | `src/real_sample_glare_prior_fusion_validation.py` | lines 118-131 |
| 真实样本 feature 构造 | `src/real_sample_glare_prior_fusion_validation.py` | lines 237-260 |
| 焦向差分输入 | `src/train_focus_resunet_loss_experiment.py` | lines 46-54 |
| HybridDFFLoss | `src/train_focus_resunet_loss_experiment.py` | lines 147-172 |
| NA 接收锥风险 | `submission_planning/tools/glare_risk_microfacet_demo.py` | lines 57-77 |
| focus confidence toy probe | `submission_planning/tools/focus_confidence_risk_study.py` | lines 87-125 |
| super-resolution microfacet probe | `submission_planning/tools/superres_downsample_glare_probe.py` | lines 65-112 |
| confidence-gated prior 版本对照 | `submission_planning/tools/compare_confidence_gated_prior_loss.py` | lines 64-78 |
