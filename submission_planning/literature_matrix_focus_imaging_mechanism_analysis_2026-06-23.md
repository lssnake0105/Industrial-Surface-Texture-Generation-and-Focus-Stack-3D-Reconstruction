# Literature Matrix 对焦成像机理深度分析

日期：2026-06-23  
项目：SRTP 反光工业表面缺陷焦栈三维重建  
输入来源：`literature_matrix.md`、`related_work_core_15.bib`、本地 Zotero PDF、既有光学机理与真实焦曲线诊断报告  
分析口径：以对焦成像机理为主，保留关键公式，比较各路线的成像假设、创新点、失效模式与对本项目的启发。

## 0. 总判断

这组文献可以压缩成一条主线：焦栈深度恢复的核心，是从物距、焦距、光圈和局部表面反射共同决定的轴向清晰度变化中恢复高度。传统 SFF/DFF 把该问题写成“在焦栈上寻找最大 focus measure”；学习型 DFF 把该问题写成“从焦堆体或焦向差分体中学习最佳对焦概率”；Defocus / S2R 文献把该问题写成“用 thin-lens defocus model 将深度、相机参数和模糊半径显式连接”；反光表面相关机理进一步指出，真实工业样本中还存在局部法线、NA 接收锥、镜面反射、饱和裁剪和离焦亮斑边缘，这些因素会把普通纹理的对焦响应改写成高亮驱动的伪峰。

对本项目最关键的判断是：DFF/GADFF 不应被当作处处可靠的深度标签，更适合被建模为带置信度的物理先验观测。真正有投稿价值的机制表述，是在 simulation-to-real 条件下，用可控合成真值学习高度恢复，同时用 focus confidence 调节传统焦栈先验的训练强度；glare/risk cue 当前更适合作为辅助诊断与软化信号。

## 1. 统一成像模型：从薄透镜到反光焦栈

### 1.1 薄透镜散焦半径

DDFS、Focus on Defocus、DEReD 都围绕同一个光学事实展开：当物体深度 $d$ 与对焦距离 $d_f$ 不一致时，点不再成像为理想点，而会形成 ==circle of confusion (CoC)==。DDFS 采用的形式为：

$$
c=b\frac{|d-d_f|}{d}\frac{f^2}{N(d_f-f)},
$$

其中 $f$ 为焦距，$N$ 为 f-number，$b$ 将物理长度转换为像素尺度。Focus on Defocus 写作：

$$
c=\frac{|S_2-S_1|}{S_2}\frac{f^2}{N(S_1-f)},
$$

其中 $S_1$ 是对焦距离，$S_2$ 是物体距离。DEReD 进一步把 CoC 转成用于渲染的半径：

$$
\sigma=\frac{\mathrm{CoC}}{2p}
=\frac{1}{2p}\frac{|d_o-F|}{d_o}\frac{f^2}{N(F-f)},
$$

其中 $p$ 是 CMOS 像素尺寸。三者的共同点是：焦栈中的模糊程度并非只由深度决定，还依赖焦距、光圈、对焦距离、像素尺度和相机标定。

### 1.2 焦栈图像生成

对一组焦层 $k=1,\dots,K$，可将焦栈抽象为：

$$
I_k(x)=\mathrm{clip}\left((A(x)\ast h_{\sigma_k(H(x))})(x)+\eta_k(x)\right),
$$

其中 $H(x)$ 是高度或深度，$A(x)$ 是 all-in-focus radiance / reflectance，$h_{\sigma}$ 是由 CoC 半径决定的 PSF，$\eta_k$ 是噪声，`clip` 表示曝光饱和。普通 DFF 往往默认 $A(x)$ 含有稳定纹理，且 $I_k$ 的清晰度峰值由 $H(x)$ 决定。

反光工业表面需要将 $A(x)$ 写得更细：

$$
A(x)=T(x)+R(\mathbf n(x),\mathbf l,\mathbf v,\mathrm{NA}),
$$

其中 $T(x)$ 是漫反射或纹理项，$R$ 是由局部法线、入射方向、观察方向和物镜接收孔径共同决定的镜面或高亮项。当 $R$ 强到触发饱和时，focus measure 看到的可能是高亮边缘、饱和平台边界和焦向亮度漂移，而非真实几何纹理。

### 1.3 从 focus measure 到深度

传统 SFF/DFF 的基本选择规则为：

$$
\hat{k}(x)=\arg\max_k F_k(x),\qquad \hat{H}(x)=z_{\hat{k}(x)},
$$

其中 $F_k(x)=\Phi(I_k,\Omega_x)$ 是局部窗口 $\Omega_x$ 上的 focus measure。常见例子包括：

$$
F_{\mathrm{TEN}}(x)=\sum_{u\in\Omega_x}\left(G_x(u)^2+G_y(u)^2\right),
$$

$$
F_{\mathrm{SML}}(x)=\sum_{u\in\Omega_x}\left(|I_{xx}(u)|+|I_{yy}(u)|\right),
$$

$$
F_{\mathrm{VAR}}(x)=\frac{1}{|\Omega_x|}\sum_{u\in\Omega_x}\left(I(u)-\bar I_{\Omega_x}\right)^2.
$$

这类公式的隐含假设是：真实对焦层会产生更高的高频能量、梯度、二阶导数或局部方差。'反光样本中的麻烦在于，离焦亮斑边缘也能产生高梯度和高 Laplacian，持续饱和会压缩纹理动态范围，弱纹理和多峰响应会降低 $\arg\max$ 的可靠性。

### 1.4 从最大层选择到概率回归

学习型 DFF 将硬选择扩展为概率分布。DFV 中，网络输出 focus probability volume：

$$
\sum_{i=1}^{N}p_j^i=1,\qquad
\hat{d}_j=\sum_{i=1}^{N}p_j^i l_i,
$$

其中 $p_j^i$ 表示像素 $x_j$ 在第 $i$ 个焦层最佳对焦的概率，$l_i$ 是第 $i$ 层焦距或归一化焦层坐标。DFV 还用加权标准差定义不确定性：

$$
\phi_j=\sqrt{\sum_{i=1}^{N}p_j^i(l_i-\hat d_j)^2}.
$$

这与本项目的 focus confidence 逻辑高度一致：==焦堆并不只输出一个深度值，还应输出该深度值的可信度==。对反光表面而言，置信度比单一深度值更接近真实问题本质。

### 1.5 反光与 NA 接收锥

物镜数值孔径定义为：

$$
\mathrm{NA}=n\sin\theta,
$$

其中 $n$ 为介质折射率，$\theta$ 为接收光锥半角。局部镜面反射方向可写为：

$$
\mathbf r = 2(\mathbf n\cdot \mathbf l)\mathbf n-\mathbf l.
$$

若满足：

$$
\arccos(\mathbf r\cdot \mathbf v)\leq \arcsin(\mathrm{NA}/n_{\mathrm{medium}}),
$$

==该微表面更可能把反射光送入成像路径==。更完整的 microfacet / Cook-Torrance 类 BRDF 可抽象为：

$$
f_r(\mathbf l,\mathbf v)=
\frac{D(\mathbf h)F(\mathbf v,\mathbf h)G(\mathbf l,\mathbf v,\mathbf h)}
{4(\mathbf n\cdot\mathbf l)(\mathbf n\cdot\mathbf v)}.
$$

本项目当前不需要完整拟合 BRDF，但这个公式提供了一个重要思想：反光来自微表面法线分布、菲涅尔项、遮蔽项和几何项的共同作用。因此，反光焦栈仿真应至少保留法线、NA、曝光裁剪和焦向 PSF 的耦合。

## 2. 逐篇机制分析

### 2.1 Nayar and Nakagawa 1994, Shape from Focus

**成像机理。**  
该文奠定了 SFF/DFF 的基本范式：采集一组不同焦平面的图像，对每个像素计算局部清晰度，利用清晰度峰值对应的焦平面恢复三维形状。它把成像机理简化成一个可操作假设：真实深度处的局部纹理最清晰，因此 focus curve 应在真实焦层附近达到最大值。

**关键公式。**

$$
\hat{k}(x)=\arg\max_k F(I_k,x),\qquad \hat{z}(x)=z_{\hat{k}(x)}.
$$

为了获得亚焦层精度，可在最大响应附近做 Gaussian / quadratic interpolation。若把局部 Gaussian focus curve 取对数，可近似写成：

$$
\log F(z)\approx az^2+bz+c,\qquad \hat z=-\frac{b}{2a}.
$$

**创新点。**  
它把被动焦栈从“视觉清晰度判断”转化为几何测量问题，建立了后续 focus measure、focus volume 和 learning-based DFF 的共同语言。

**对本项目的启发。**  
Nayar 路线提供了最强可解释 baseline，但它默认 focus curve 的峰值可信。对反光表面，峰值可能由高亮边缘或饱和裁剪产生，所以本项目应保留 DFF 作为先验，同时引入 focus confidence 判断该先验能否被训练目标信任。

### 2.2 Pertuz et al. 2013, Analysis of Focus Measure Operators

**成像机理。**  
Pertuz 对梯度、Laplacian、wavelet、统计量、DCT 等 focus measure 做系统比较。它的价值在于==把“清晰”拆成不同可计算的频域或空间域响应，并说明 operator 性能会受到噪声、对比度、饱和和窗口尺寸影响。==

**关键公式。**

Tenengrad:

$$
F_{\mathrm{TEN}}(x)=\sum_{u\in\Omega_x}(G_x(u)^2+G_y(u)^2).
$$

Modified Laplacian / SML:

$$
\mathrm{ML}(u)=|2I(u)-I(u-\Delta x)-I(u+\Delta x)|
+|2I(u)-I(u-\Delta y)-I(u+\Delta y)|,
$$

$$
F_{\mathrm{SML}}(x)=\sum_{u\in\Omega_x}\mathrm{ML}(u).
$$

Gray-level variance:

$$
F_{\mathrm{VAR}}(x)=\frac{1}{|\Omega_x|}\sum_{u\in\Omega_x}(I(u)-\bar I_{\Omega_x})^2.
$$

**创新点。**  
该文的创新主要在评估框架而非单一算法：它让后续研究可以从 operator family、噪声敏感性、窗口大小、contrast / saturation 这些维度比较 SFF。

**对本项目的启发。**  
反光表面的 DFF 失效可以直接用 Pertuz 的 operator 语言解释：Laplacian / Tenengrad 会奖励高亮边缘，variance 会受到饱和平台和局部对比度影响，固定窗口会混合真实纹理、高亮扩散和边界伪影。该文支撑本项目把 focus curve morphology 分成 confident single peak、flat ambiguous、multi-peak、local spike、saturated highlight 等类别。

### 2.3 Lee et al. 2013, Adaptive Window Selection

**成像机理。**  
Lee 等将传统 focus measure 的局部窗口从固定尺度改为自适应尺度。第一阶段根据图像强度离散程度选择用于计算 focus measure 的窗口；第二阶段根据初始 focus value 的离散程度选择用于增强 focus measure 的窗口。

**关键表达。**

可抽象为：

$$
F_k(x;w)=\Phi(I_k,\Omega_x(w)),
$$

$$
w_{\mathrm{measure}}(x)=g_1(\mathrm{disp}(I_k|\Omega_x)),\qquad
w_{\mathrm{enhance}}(x)=g_2(\mathrm{disp}(F_k|\Omega_x)).
$$

**创新点。**  
该文指出 DFF 的误差并不只来自 operator，也来自窗口尺度。小窗口在弱纹理区域不稳，大窗口会过平滑边界或扭曲形状。

**对本项目的启发。**  
自适应窗口适合作为传统增强 baseline，但它主要解决空间支持域不足的问题。对反光表面，高亮伪峰具有明确的光学来源，扩大或缩小窗口可能改变伪峰形态，却难以判断伪峰是否代表真实高度。因此，它能改善普通弱纹理或噪声，但不能单独解决 flare-driven focus failure。

### 2.4 Li et al. 2019, Adaptive Window Iteration Algorithm

**成像机理。**  
Li 等进一步把 adaptive window 与 focus value iteration 结合，目标是缓解固定窗口导致的局部信息不足和一次 focus evaluation 难以获得全焦像素的问题。该文仍以 modified Laplacian / SML 类清晰度为基础。

**关键公式。**

其底层仍可用 SML 表达：

$$
F_{\mathrm{SML}}(x)=\sum_{u\in\Omega_x(w)}\left(|I_{xx}(u)|+|I_{yy}(u)|\right),
$$

并通过灰度差异选择窗口、通过迭代增强评价值：

$$
F^{(t+1)}(x)=\mathcal E(F^{(t)}(x), \Omega_x(w_x)).
$$

这里 $\mathcal E$ 表示局部 focus evaluation enhancement，精确实现依赖论文算法细节。

**创新点。**  
该文更接近工程显微测量问题，强调微观图像测量中 3D reconstruction distortion 与窗口尺度、focus evaluation 不充分有关。

**对本项目的启发。**  
Li2019 很适合作为工业显微传统基线。它也提醒本项目：若反光伪峰已经成为局部最强响应，迭代增强可能会强化错误峰值。因此，传统增强方法的结果需要分区看，尤其要看 saturated highlight、local peak spike 和 high-risk ROI。

### 2.5 DDFFNet / Deep Depth from Focus

**成像机理。**  
DDFFNet 将 DFF 从手工 focus measure 改为端到端学习问题。输入为 focal stack，输出为 disparity / depth。它仍利用“焦向清晰度变化携带深度信息”的物理事实，但不显式定义 Laplacian、CoC 或窗口尺度，而是让网络学习空间上下文与焦向变化。

**关键表达。**

$$
\hat D=f_\theta(I_1,I_2,\dots,I_K).
$$

监督训练可抽象为：

$$
\mathcal L_{\mathrm{depth}}=\ell(\hat D,D_{\mathrm{gt}}).
$$

**创新点。**  
DDFFNet 是学习型 DFF 的基础工作之一，关键贡献在于证明 CNN 可以从 focal stack 中学习比手工 focus measure 更稳健的深度估计，并用光场相机和 RGB-D 传感器构造了较大规模带真值数据。

**对本项目的启发。**  
==DDFFNet 适合作为 learning-based baseline 的下限==。它的局限也很清楚：网络可能把相机设置、数据集外观、纹理统计隐式吸收进参数，面对显微反光表面、合成到真实迁移和缺少真实高度真值时，单纯端到端训练的可解释性不足。

### 2.6 AiFDepthNet, All-in-Focus Supervision

**成像机理。**  
AiFDepthNet 同时估计 depth map 和 all-in-focus image，把焦层选择写成 attention。它的关键思想是：对深度估计，注意力分布应允许较平滑的概率回归；对 AiF 重建，注意力应更尖锐地选出最清晰像素。

**关键公式。**

Depth attention:

$$
M_{\mathrm{depth}}=\varsigma(M),
$$

$$
M^{\mathrm{depth}}_{i,j,1,t}
=\frac{\ln(1+\exp(M_{i,j,1,t}))}
{\sum_{n=1}^{F}\ln(1+\exp(M_{i,j,1,n}))},
$$

$$
D_{i,j,1}=\sum_{t=1}^{F}(M_{\mathrm{depth}}\cdot P)_{i,j,1,t}.
$$

AiF attention:

$$
M^{\mathrm{AiF}}_{i,j,1,t}
=\frac{\exp(M_{i,j,1,t})}{\sum_{n=1}^{F}\exp(M_{i,j,1,n})},
$$

$$
I^{\mathrm{AiF}}_{i,j,k}=\sum_{t=1}^{F}(M_{\mathrm{AiF}}\cdot S)_{i,j,k,t}.
$$

**创新点。**  
该文把监督和无监督之间的断层变成一个可训练的中间任务：即使缺少 depth GT，也可用 AiF image 提供焦层选择监督。

**对本项目的启发。**  
AiFDepthNet 对真实无高度真值很有启发，但反光显微样本中 AiF 本身也可能被饱和、高亮边缘和曝光策略污染。对本项目而言，AiF 更适合作为后续自监督或辅助输出，不宜直接替代高度真值。

### 2.7 DFV, Deep Depth from Focus with Differential Focus Volume

**成像机理。**  
DFV 是与本项目 focal-difference prior 最接近的学习型方法。它认为深度不仅体现在某一层是否清晰，也体现在跨焦层特征的一阶变化。通过 stacked features over focal distances 的 first-order derivative，DFV 同时捕捉 focus information 和 context information。

**关键公式。**

焦向差分可抽象为：

$$
\Delta \mathbf f_i(x)=\mathbf f_{i+1}(x)-\mathbf f_i(x),
$$

网络输出 focus probability volume：

$$
\hat i_j=\sum_{i=1}^{N}p_j^i i,
$$

$$
\hat d_j=\sum_{i=1}^{N}p_j^i l_i,
$$

$$
\phi_j=\sqrt{\sum_{i=1}^{N}p_j^i(l_i-\hat d_j)^2}.
$$

**创新点。**  
DFV 的创新在于把焦栈视为沿焦距轴的可微体数据，并用概率回归实现 sub-frame / sub-layer focus localization。它还自然给出 uncertainty，这对稀疏焦栈很关键。

**对本项目的启发。**  
DFV 是当前最强相关 SOTA。它证明焦向差分是合理表征，也支持本项目把 $\Delta I$ 或 feature-difference 作为输入。但 DFV 默认的 focus probability 仍主要来自普通焦栈数据；反光表面还需要额外处理饱和、高亮伪边缘和局部不可恢复区域。因此，本项目与 DFV 的差异应写成：DFV 建模焦向变化，本项目进一步对焦向变化的可靠性进行物理风险和 confidence-aware 调节。

### 2.8 Learning Depth from Focus in the Wild

**成像机理。**  
该文强调真实手机焦栈与合成焦栈之间的一个常被忽略问题：focal breathing 和硬件误差会导致不同焦层的视场变化与横向错位。若不同焦层的同一像素并不对应同一物点，传统 focus measure 和 CNN 都会把配准误差误当成焦向变化。

**关键公式。**

真实镜头对焦时，可由薄透镜关系写出传感器-镜头距离 $v_k$：

$$
\frac{1}{f}=\frac{1}{u_k}+\frac{1}{v_k}.
$$

视场角随 $v_k$ 改变，可抽象为：

$$
\mathrm{FoV}_k=2\arctan\left(\frac{s}{2v_k}\right),
$$

其中 $s$ 是传感器尺寸。不同焦层的相对缩放与平移需要被估计和校正。

**创新点。**  
该文把真实焦栈中的配准、focus breathing、弱纹理和相机仿真器放到学习框架中，补上了公开合成数据与真实设备之间的重要缺口。

**对本项目的启发。**  
本项目的真实焦栈诊断已经发现多数样本全局 median shift 较小，但局部高亮和边缘区域仍可能有外观变化导致的伪错位。因此，真实无真值验证应同时报告配准检查、focus curve morphology 和 high-glare ROI，而不能只展示最终高度图。

### 2.9 DDFS, Deep Depth from Focal Stack with Defocus Model

**成像机理。**  
DDFS 将 defocus model、plane sweep volume 和 camera setting 显式放入网络。它指出 blur size 同时依赖 scene depth 和 camera settings；如果训练和测试的焦距、f-number、对焦距离不同，缺少光学模型的学习型方法可能无法输出正确深度。

**关键公式。**

CoC:

$$
c=b\frac{|d-d_f|}{d}\frac{f^2}{N(d_f-f)}.
$$

基于 CoC 的模糊成像：

$$
I_{d_f}=I_{\mathrm{AiF}}\ast h_{c(d,d_f,f,N)}.
$$

plane sweep / cost volume 可抽象为：

$$
C(x,d_m)=\Psi\left(I_1,\dots,I_K; d_m, f, N, d_{f,1:K}\right),
$$

深度可由 soft argmin 得到：

$$
\hat d(x)=\sum_m d_m\frac{\exp(-C(x,d_m))}{\sum_n\exp(-C(x,d_n))}.
$$

**创新点。**  
DDFS 的关键创新是把相机参数差异吸收到中间表示中，从而追求 camera-setting invariance 和 synthetic-to-real robustness。

**对本项目的启发。**  
DDFS 强烈支持本项目记录物镜 NA、焦层间距、曝光时间、焦距等 metadata。对显微反光表面，camera-setting invariance 还需要扩展为 illumination / reflectance / saturation invariance；也就是除了 $f,N,d_f$，还要关注 NA 接收锥、照明角、曝光裁剪和局部法线。

### 2.10 Focus on Defocus

**成像机理。**  
Focus on Defocus 将 defocus blur 作为 synthetic-to-real 更稳定的中间监督信号。它的核心判断是：相较于外观纹理，焦点差异和 defocus map 在合成域与真实域之间更具有结构一致性。

**关键公式。**

$$
c=\frac{|S_2-S_1|}{S_2}\frac{f^2}{N(S_1-f)}.
$$

defocus map 可由 CoC 裁剪并归一化：

$$
B(x)=\mathrm{norm}(\mathrm{clip}(c(x),0,c_{\max})).
$$

**创新点。**  
该文把 S2R 叙事从 photo-realistic rendering 转向 domain-invariant cue。它使用 defocus map 作为中间监督，让模型学焦点差异，而非过度依赖图像外观。

**对本项目的启发。**  
这篇文献是本项目 simulation-to-real 主线的重要支撑。对本项目而言，合成数据的目标不应只追求视觉逼真，更应复现真实焦栈中的 focus-response morphology：单峰、平坦、多峰、局部 spike、饱和高亮和暗弱信号。

### 2.11 DEReD, Fully Self-Supervised Depth Estimation from Defocus Clue

**成像机理。**  
DEReD 在只有 sparse focal stack、没有 depth 或 AiF ground truth 的情况下，同时预测 depth 和 AiF，再通过 optical model 重建输入焦栈。它的监督来自“预测深度 + 预测 AiF 经光学模型渲染后应回到原焦栈”。

**关键公式。**

CoC:

$$
\mathrm{CoC}=A\frac{|d_o-F|}{d_o}\frac{f}{F-f}.
$$

使用 f-number $N=f/A$ 并转成像素半径：

$$
\sigma=\frac{1}{2p}\frac{|d_o-F|}{d_o}\frac{f^2}{N(F-f)}.
$$

自监督重建损失可抽象为：

$$
\mathcal L_{\mathrm{recon}}
=\sum_{k=1}^{K}\left\|I_k-\mathcal R(\hat A,\hat D;F_k,f,N,p)\right\|_1.
$$

**创新点。**  
它将深度估计的监督从外部真值转移到焦栈自身的一致性，缓解真实场景中 depth/AiF GT 难获得的问题。

**对本项目的启发。**  
DEReD 为真实无高度真值提供了后续扩展路线：可以用当前模型、DFF/GADFF ensemble 和焦栈重建一致性构造自监督约束。但反光显微样本中，AIF 与 depth 的可辨识性会被饱和裁剪和镜面高亮破坏，因此自监督项也需要 confidence / saturation mask。

### 2.12 Dr.Bokeh, Differentiable Occlusion-Aware Bokeh Rendering

**成像机理。**  
Dr.Bokeh 研究的是从 RGBD 和镜头参数渲染真实 bokeh / defocus blur。它与 DFF 深度估计方向相反：DFF 从焦栈推深度，Dr.Bokeh 从深度渲染焦外图像。它的价值在于把 defocus rendering 做成可微模块，并显式处理遮挡与层间混合。

**关键公式。**

分层遮挡感知 bokeh rendering 可抽象为：

$$
B(x)=\sum_{l=1}^{n}V_l(x)\prod_{k=1}^{l-1}(1-V_k(x))
\frac{\sum_{y\in\Omega}I_l(y)w_l(y,x)O_l(y,x)}
{\sum_{y\in\Omega}w_l(y,x)O_l(y,x)}.
$$

其中 $V_l$ 表示层间可见性，$O_l$ 表示层内遮挡，$w_l$ 是由 CoC / bokeh kernel 决定的权重。其物理渲染部分还使用 lensmaker 关系：

$$
\frac{1}{f}=(\eta-1)\left(\frac{2}{R_c}+\frac{(\eta-1)d}{\eta R_c^2}\right).
$$

**创新点。**  
Dr.Bokeh 的创新是让焦外渲染既物理合理又可微，尤其处理普通卷积渲染容易出现的 color bleeding 和遮挡边界错误。

**对本项目的启发。**  
如果后续要把仿真器升级为更真实的 focus-stack renderer，Dr.Bokeh 提供了可微离焦、遮挡和层状场景的设计参考。当前阶段更现实的做法是借鉴其“物理渲染模块可进入训练闭环”的思想，而非完整实现其 bokeh 渲染。

### 2.13 Depth Anything / Depth Anything V2

**成像机理。**  
这两篇属于单目深度基础模型，不直接建模焦栈、focus measure 或 defocus PSF。其输入通常是单张 RGB 图像，输出相对或度量深度，因此它缺少焦栈的轴向响应。

**关键表达。**

其对本项目有价值的部分可抽象为 teacher-student / pseudo-label 策略：

$$
\tilde D_{\mathrm{real}}=T(x_{\mathrm{real}}),\qquad
\mathcal L_{\mathrm{student}}=\ell(S_\theta(x_{\mathrm{real}}),\tilde D_{\mathrm{real}}),
$$

并结合高质量合成标签：

$$
\mathcal L=\mathcal L_{\mathrm{synthetic\ GT}}+\lambda\mathcal L_{\mathrm{pseudo\ real}}.
$$

**创新点。**  
这一路线的创新不在对焦成像，而在数据规模、合成标签质量、伪标签真实数据和 teacher-student bridge。

**对本项目的启发。**  
Depth Anything V2 应放在训练策略和 auxiliary prior 位置。它支持“高质量合成真值 + 真实无标签样本 + 伪标签/一致性训练”的方法论，但不应进入 focus-stack 主数值 SOTA 表。

### 2.14 Surface Defect Detection Review

**成像机理。**  
工业表面缺陷综述主要讨论 2D 缺陷检测、传统机器视觉、深度学习检测、小样本、小目标、不平衡样本和实时性挑战。它并不直接提供焦栈成像公式。

**创新点。**  
该类综述的价值在于定义应用场景：工业缺陷研究长期偏向 detect / classify，但很多缺陷严重程度取决于 3D morphology，例如坑深、边缘塌陷、划痕深度、毛刺高度和局部粗糙度。

**对本项目的启发。**  
它支撑 introduction 中的应用动机：本项目的目标不是只定位缺陷，而是恢复反光微结构的相对形貌。它也提示评估指标应从分类准确率转向 MAE、edge MAE、profile consistency、spike count 和真实标定 ROI。

### 2.15 Att-PU-Net BDSIM 2026

**成像机理。**  
Att-PU-Net BDSIM 属于 bright-field / dark-field structured illumination microscopy 与点云上采样结合的工业缺陷三维重建路线。它与焦栈 DFF 的直接机理不同，但同样关注微纳缺陷的 3D morphology 和真实 WLI 验证。

**一般结构光表达。**

结构照明显微可抽象为：

$$
I_m(x)=A(x)+B(x)\cos(2\pi f_0 x+\phi_m),
$$

其中 $\phi_m$ 是相位步进。bright-field 更强调直射/近轴信息，dark-field 更强调散射、高角度或边缘相关信息。

**创新点。**  
该文的创新是把 bright-field 与 dark-field 的互补信息整合到缺陷点云重建，并用 Att-PU-Net 提升点云质量，再用 WLI 做真实深度验证。

**对本项目的启发。**  
它提供了两个投稿层面的标尺：第一，工业缺陷 3D reconstruction 是正在发展的相关方向；第二，真实 WLI 或轮廓仪标定会显著增强 claim。当前本项目若无法做大规模真实 GT，至少应争取小规模 ROI / profile 级标定。

### 2.16 Optical Reflection / NA / Microfacet Models

**成像机理。**  
这组理论不是单篇 DFF 文献，却是本项目反光表面故事线的物理根基。对同轴显微，局部法线决定镜面反射是否进入物镜接收锥；NA 决定接收锥半角；microfacet / BRDF 解释粗糙纹理、刃脊、孔边缘为何会产生局部高亮。

**关键公式。**

$$
\mathrm{NA}=n\sin\theta,
$$

$$
\mathbf r = 2(\mathbf n\cdot\mathbf l)\mathbf n-\mathbf l,
$$

$$
R_{\mathrm{glare}}(x)=\mathbb 1\left[
\arccos(\mathbf r(x)\cdot\mathbf v)\leq \arcsin(\mathrm{NA}/n_{\mathrm{medium}})
\right].
$$

soft risk 可写成：

$$
R_{\mathrm{soft}}(x)=\sigma\left(\alpha\left(\arcsin(\mathrm{NA}/n_{\mathrm{medium}})
-\arccos(\mathbf r(x)\cdot\mathbf v)\right)\right).
$$

**创新点。**  
对本项目而言，创新不在提出全新 BRDF，而在把最小反射-孔径模型转化为焦栈重建中的风险先验，并用它解释 DFF 伪峰、饱和持久性和真实焦曲线分型。

**对本项目的启发。**  
已有实验显示，focus confidence 是当前更稳定的门控变量，global multiplicative risk gate 的表现不如 no-risk 的 $C_{\mathrm{focus}}^{1.5}$ 方案。因此，论文中应把 NA / microfacet 风险写成机理解释、仿真变量和辅助诊断，避免把它夸大成已经被定量证明的主要收益来源。

### 2.17 扩展条目：Dense Depth from Event Focal Stack

**成像机理。**  
该文使用事件相机在扫焦过程中产生 event stream，并将其组织为 event focal stack。事件相机记录亮度变化事件：

$$
e=(x,y,t,p),\qquad
\Delta \log I(x,y,t)\geq C,
$$

其中 $p$ 是极性，$C$ 是事件触发阈值。扫焦时，不同深度位置的清晰度变化会触发不同时间-空间事件分布。

**创新点。**  
它把焦栈从帧图像扩展到事件流，利用事件相机高动态范围和高时间分辨率缓解传统相机在快速变化或过曝/欠曝条件下的限制。

**对本项目的启发。**  
该文不适合作为当前硬件条件下的直接 baseline，但它提示一个方向：反光表面中的动态范围问题可能需要从传感器层面解决。若后续硬件可扩展，事件焦栈或 HDR 焦栈可能比普通曝光焦栈更适合处理强反光。

## 3. 横向比较：这些文献到底差在哪里

### 3.1 对焦线索的数学对象不同

| 路线 | 核心对象 | 代表文献 | 本质优势 | 主要缺口 |
|---|---|---|---|---|
| 最大 focus measure | $F_k(x)$ 与 $\arg\max_k F_k(x)$ | Nayar, Pertuz, Lee, Li | 可解释、易实现、适合传统 baseline | 弱纹理、饱和、高亮伪峰、多峰响应不稳 |
| 学习型 focal stack | $f_\theta(I_{1:K})$ | DDFFNet | 可学习上下文和非线性模式 | 依赖数据，光学参数与失效原因常被黑箱吸收 |
| 焦向概率体 | $p_j^i$、$\hat d_j=\sum_i p_j^i l_i$ | DFV, AiFDepthNet | 可做 sub-layer 回归和不确定性分析 | 概率高不一定代表物理正确，反光区域仍需可靠性约束 |
| 显式 defocus model | $c(d,d_f,f,N)$、plane sweep volume | DDFS, Focus on Defocus, DEReD | 能解释相机设置和 S2R | 常假设可渲染的 AIF / diffuse radiance，反光饱和会破坏一致性 |
| 可微渲染 | $B(x)$、layer visibility、CoC kernel | Dr.Bokeh | 适合构建训练闭环与仿真器 | 实现复杂，显微反光材质仍需额外建模 |
| 反光 / NA 风险 | $\mathbf r,\mathrm{NA},f_r$ | NA, microfacet, BRDF | 能解释反光伪峰来源 | 当前最适合做辅助 prior，完整标定难度高 |

### 3.2 监督信号不同

| 监督方式 | 代表文献 | 适用条件 | 对本项目意义 |
|---|---|---|---|
| 深度 GT 监督 | DDFFNet, DFV | 有 dense depth/disparity GT | 合成数据主训练和主量化评估 |
| AiF supervision | AiFDepthNet | 有可信 all-in-focus 图像 | 可作为真实无高度真值的辅助路线 |
| defocus map supervision | Focus on Defocus | 可由相机参数和深度生成 CoC | 支撑 S2R 的 domain-invariant cue |
| 焦栈自重建 | DEReD | 有焦栈和可用光学模型 | 后续真实无 GT 自监督扩展 |
| pseudo-label / teacher-student | Depth Anything V2 | 有大规模无标签真实数据和教师模型 | 后续真实焦栈一致性训练参考 |
| no-reference diagnostics | 本项目真实焦曲线分析 | 缺少真实高度 GT | 用于真实域 claim 边界和机制一致性检查 |

### 3.3 对真实域问题的覆盖不同

| 问题 | 已有文献覆盖 | 尚未充分覆盖 | 本项目可切入点 |
|---|---|---|---|
| 相机参数变化 | DDFS | 显微系统 metadata 不完整时的鲁棒性 | 记录 NA、焦层间距、曝光与焦距 |
| 焦层错位 / focal breathing | DfF in the Wild | 反光区域导致的局部外观变化 | 配准诊断 + ROI focus curve |
| 合成到真实 | Focus on Defocus, DDFS, Depth Anything V2 | 反光显微焦栈的光学失效分布 | 用仿真复现 focus-response morphology |
| 无真实高度真值 | AiFDepthNet, DEReD | AiF 与反光饱和本身不可靠 | confidence-aware self-supervision |
| 反光 / 高亮 | NA / BRDF theory, industrial highlight work | 与 DFF 伪峰的直接连接 | glare-risk + saturation-persistence + focus confidence |
| 工业 3D 缺陷 | Att-PU-Net BDSIM | 焦栈 DFF 与反光微结构耦合 | 反光缺陷相对形貌重建 |

## 4. 对本项目最有用的创新对照

### 4.1 与传统 SFF/DFF 的差异

传统方法把 focus measure 视为深度选择的直接依据。本项目应将其降级为 prior observation：

$$
P_{\mathrm{DFF}}(x)=H(x)+\epsilon(x),
\qquad \epsilon(x)\sim q(C_{\mathrm{focus}}(x),R(x)).
$$

该表达的价值在于：DFF 的错误不再被视为随机噪声，而是与可观测的 focus confidence、saturation persistence、glare risk 和 focus curve morphology 相关。

### 4.2 与 DFV 的差异

DFV 的贡献是 differential focus volume：

$$
\Delta \mathbf f_i=\mathbf f_{i+1}-\mathbf f_i.
$$

本项目可以吸收该思想，但进一步强调“差分响应是否可靠”。反光伪边缘也会产生强焦向差分，因此仅有 $\Delta I$ 或 $\Delta f$ 不够，还需要：

$$
W_{\mathrm{prior}}(x)=g(C_{\mathrm{focus}}(x),R(x),S_{\mathrm{sat}}(x)).
$$

已有实验中最稳定的版本接近：

$$
W_{\mathrm{prior}}=\mathrm{clip}\left(C_{\mathrm{focus}}^{1.5}(1-0.45R),0.02,1.0\right),
$$

但风险项的主要作用应保持克制，focus confidence 才是当前更强的主变量。

### 4.3 与 DDFS / Focus on Defocus 的差异

DDFS 与 Focus on Defocus 强调：

$$
c=c(d,d_f,f,N).
$$

本项目需要扩展为：

$$
c=c(d,d_f,f,N),\qquad
I= \mathrm{clip}\left((T+R(\mathbf n,\mathbf l,\mathbf v,\mathrm{NA}))\ast h_c\right).
$$

也就是说，反光显微焦栈不只需要 depth-to-blur model，还需要 normal-to-glare、glare-to-saturation、saturation-to-pseudo-focus 的链条。

### 4.4 与 DEReD / Dr.Bokeh 的差异

DEReD 和 Dr.Bokeh 都把光学模型放进训练或渲染闭环。对本项目，直接引入完整可微渲染成本较高，短期更可行的是轻量化：

$$
\mathcal L
=\mathcal L_{\mathrm{GT}}
+\lambda_{\mathrm{prior}}\mathcal L_{\mathrm{confidence\ prior}}
+\lambda_{\mathrm{smooth}}\mathcal L_{\mathrm{shape}}.
$$

其中：

$$
\mathcal L_{\mathrm{confidence\ prior}}
=
\frac{\sum_x W_{\mathrm{prior}}(x)\rho(\hat H(x)-P_{\mathrm{prior}}(x))}
{\max(\sum_x W_{\mathrm{prior}}(x),1)}.
$$

这条路线比完整渲染更适合当前投稿节奏，因为它能直接利用已有合成 GT 和真实无参考诊断结果。

## 5. 适合写入论文的机制主线

### 5.1 可写成 Related Work 的逻辑

1. Classical SFF/DFF 建立了基于 focus measure 峰值恢复深度的范式，但其可靠性受纹理、噪声、饱和和窗口大小限制。
2. Learning-based DFF 通过 CNN、attention、differential focus volume 和 probability regression 提升了焦栈建模能力，但多数工作对反光显微表面的高亮伪峰缺少显式处理。
3. Defocus-model-based 与 S2R 工作将 CoC、相机参数和光学渲染引入训练，使焦栈深度估计从外观学习转向物理约束学习。
4. 工业缺陷 3D 重建工作说明真实应用正在从 2D 检测走向可量化形貌，但真实高度标定仍是 claim 强度的关键。
5. 本项目的切入点是反光显微焦栈中的 focus reliability：用合成真值训练高度恢复，用 focus confidence 和反光风险软化传统 DFF/GADFF 先验。

### 5.2 可写成 Method motivation 的逻辑

普通 DFF 假设：

$$
\hat H(x)=z_{\arg\max_k F_k(x)}.
$$

反光显微样本中，真实观测更接近：

$$
F_k(x)=F^{\mathrm{texture}}_k(x;H)+F^{\mathrm{glare}}_k(x;\mathbf n,\mathrm{NA},\mathrm{exposure})+\xi_k(x).
$$

当 $F^{\mathrm{glare}}_k$ 占主导时，$\arg\max$ 选择的焦层可能对应高亮伪峰。解决思路并非丢弃 DFF，而是估计其局部可靠性：

$$
P_{\mathrm{prior}}=\alpha P_{\mathrm{DFF}}+(1-\alpha)P_{\mathrm{GADFF}},
$$

$$
W_{\mathrm{prior}}=g(C_{\mathrm{focus}},R_{\mathrm{glare}},S_{\mathrm{sat}}),
$$

$$
\mathcal L_{\mathrm{prior}}
=
\frac{\sum_x W_{\mathrm{prior}}(x)\rho(\hat H(x)-P_{\mathrm{prior}}(x))}
{\max(\sum_x W_{\mathrm{prior}}(x),1)}.
$$

### 5.3 可写成 Discussion 的边界

- Depth Anything / V2 可支持合成标签、伪标签真实数据和 teacher-student 训练策略，不适合直接作为 focus-stack 主表基线。
- DEReD / Dr.Bokeh 支持未来引入自监督焦栈重建和可微 defocus rendering，但当前真实反光样本需要先解决饱和与低置信 mask。
- Att-PU-Net BDSIM 提示真实 WLI / profile 标定非常重要。本项目真实无 GT 指标只能证明诊断一致性和形貌稳定性，不能替代绝对高度误差。
- Risk cue 当前证据更适合作为辅助变量；focus confidence 是已更稳定支撑 prior weighting 的核心变量。

## 6. 推荐引用优先级

| 优先级 | 文献 | 在“对焦成像机理”中的作用 | 本项目写法 |
|---|---|---|---|
| P0 | Nayar1994 | 最大 focus response 恢复深度 | 传统 DFF 理论起点 |
| P0 | Pertuz2013 | focus measure family 与失效因素 | 解释高亮、饱和、窗口和噪声敏感性 |
| P0 | DFV2022 | differential focus volume 与 uncertainty | 最相关学习型 SOTA |
| P0 | DDFS2024 | CoC + camera setting + plane sweep | 支撑光学参数与 S2R |
| P0 | Focus on Defocus2020 | defocus cue domain invariance | 支撑仿真到真实迁移 |
| P1 | AiFDepthNet2021 | attention + AiF supervision | 支撑无 depth GT 的训练替代路径 |
| P1 | DfF in the Wild2022 | focal breathing / alignment | 支撑真实焦栈质量检查 |
| P1 | DEReD2023 | sparse focal stack self-supervision | 后续无 GT 自监督路线 |
| P1 | Dr.Bokeh2024 | differentiable defocus rendering | 后续仿真器升级路线 |
| P1 | Att-PU-Net BDSIM2026 | 工业缺陷 3D 与 WLI 验证 | 应用动机与真实标定标准 |
| P2 | Depth Anything / V2 | 大规模合成标签与伪标签策略 | 训练策略参考，辅助讨论 |
| P2 | Event Focal Stack2025 | 高动态范围焦栈传感器方向 | 未来硬件扩展讨论 |

## 7. 一段可直接放入论文前期笔记的总结

Existing focus-stack depth methods share the same optical root: depth is encoded by the axial change of defocus blur and local sharpness. Classical SFF/DFF converts this cue into handcrafted focus measures and selects the focal layer with maximal response, while learning-based methods such as DDFFNet, AiFDepthNet, and DFV learn the focal-stack volume or its differential representation and often regress a focus probability distribution. Recent defocus-model-based methods further reintroduce thin-lens physics through CoC, camera settings, plane-sweep volumes, or differentiable rendering. However, reflective microscopic surfaces introduce an additional mechanism: local microfacet normals and finite NA can drive specular light into the objective, producing saturated highlights and defocused highlight edges that generate false focus peaks. Therefore, in this project, DFF/GADFF should be treated as confidence-aware physical priors rather than uniformly reliable labels. The central contribution should be framed as simulation-to-real, confidence-gated focus-stack reconstruction for glare-prone reflective morphology.

## 8. 参考来源

- Nayar, S. K., and Nakagawa, Y. Shape from focus. IEEE TPAMI, 1994. https://doi.org/10.1109/34.308479
- Pertuz, S., Puig, D., and Garcia, M. A. Analysis of focus measure operators for shape-from-focus. Pattern Recognition, 2013. https://doi.org/10.1016/j.patcog.2012.11.011
- Lee, I. et al. Adaptive window selection for 3D shape recovery from image focus. Optics & Laser Technology, 2013. https://doi.org/10.1016/j.optlastec.2012.08.003
- Li, L. et al. Adaptive window iteration algorithm for enhancing 3D shape recovery from image focus. Chinese Optics Letters, 2019. https://doi.org/10.3788/COL201917.061001
- Hazirbas, C. et al. Deep Depth From Focus. ACCV, 2018/2019. https://arxiv.org/abs/1704.01085
- Wang, N.-H. et al. Bridging Unsupervised and Supervised Depth from Focus via All-in-Focus Supervision. ICCV, 2021. https://arxiv.org/abs/2108.10843
- Yang, F. et al. Deep Depth from Focus with Differential Focus Volume. CVPR, 2022. https://arxiv.org/abs/2112.01712
- Won, C., and Jeon, H.-G. Learning Depth from Focus in the Wild. ECCV, 2022. https://arxiv.org/abs/2207.09658
- Fujimura, Y. et al. Deep Depth from Focal Stack with Defocus Model for Camera-Setting Invariance. IJCV, 2024. https://arxiv.org/abs/2202.13055
- Maximov, M. et al. Focus on Defocus: Bridging the Synthetic to Real Domain Gap for Depth Estimation. CVPR, 2020. https://arxiv.org/abs/2005.09623
- Si, H. et al. Fully Self-Supervised Depth Estimation from Defocus Clue. CVPR, 2023. https://arxiv.org/abs/2303.10752
- Sheng, Y. et al. Dr.Bokeh: DiffeRentiable Occlusion-Aware Bokeh Rendering. CVPR, 2024. https://arxiv.org/abs/2308.08843
- Yang, L. et al. Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. CVPR, 2024. https://arxiv.org/abs/2401.10891
- Yang, L. et al. Depth Anything V2. 2024. https://arxiv.org/abs/2406.09414
- Horikawa, K. et al. Dense Depth from Event Focal Stack. WACV, 2025. https://arxiv.org/abs/2412.08120
- Nikon MicroscopyU. Numerical Aperture. https://www.microscopyu.com/microscopy-basics/numerical-aperture
- Cook, R. L., and Torrance, K. E. A Reflectance Model for Computer Graphics. ACM TOG, 1982. https://doi.org/10.1145/357290.357293
- Ren, Z. et al. Surface Defect Detection Methods for Industrial Products: A Review. Applied Sciences, 2021. https://doi.org/10.3390/app11167657
- Wang, Z. et al. Defect 3D reconstruction with integrated bright-field and dark-field structured illumination microscopy based on Att-PU-Net. Applied Optics, 2026. https://doi.org/10.1364/AO.587592
