# 完全波动光学显微成像仿真的可行性与实现路线

## 结论

完全可以把本项目的成像仿真升级为波动光学模型。更合适的建模链条是：

```text
金属表面高度/粗糙度
-> 局部法线与 Fresnel 复反射系数
-> 反射复场 E_obj(x,y)
-> 物镜 pupil / NA / 像差 / 离焦相位
-> Fourier optics 传播到像面
-> 不同焦平面 z_f 的强度图
-> LED 部分相干与像素积分
-> 焦栈图像
```

这样得到的焦栈不再依赖经验 PSF 后处理，而是从反射复场和物镜孔径传播推导出来。对当前 Olympus 20X、525 nm LED、NA 约 0.40-0.45 的系统，第一版可以使用标量 Fourier optics；若后续要达到物镜设计或高 NA 偏振精度，再升级到 Debye / Richards-Wolf 矢量衍射模型。

## 1. 物体面：金属反射复场

表面生成器给出宏观高度：

$$
h(x,y).
$$

局部法线由高度梯度得到：

$$
\mathbf n(x,y)=
\frac{(-\partial h/\partial x,-\partial h/\partial y,1)}
{\sqrt{(\partial h/\partial x)^2+(\partial h/\partial y)^2+1}}.
$$

金属复折射率为：

$$
\tilde n = n+i\kappa.
$$

根据入射角得到 Fresnel 复反射系数：

$$
r_s(\theta),\quad r_p(\theta),\quad
r(\theta)=\frac{r_s+r_p}{2}.
$$

考虑表面残余微起伏 \(h_r(x,y)\)、粗糙度相干衰减 \(C_\sigma\)、NA 接收锥权重 \(A_{NA}\)，物体面反射复场可写为：

$$
E_{obj}(x,y)
=
r(\theta)
\sqrt{A_{NA}(x,y)}
C_\sigma(x,y)
\exp\left[
i\left(
\frac{4\pi h_r(x,y)\cos\theta}{\lambda}
+\arg r(\theta)
\right)
\right].
$$

如果要模拟弱寄生反射，可加入：

$$
E_{obj}'(x,y)=E_{obj}(x,y)+\gamma E_g(x,y).
$$

## 2. 物镜 pupil 与 NA

物镜只接收有限角度内的空间频率。标量模型中，pupil 可写为：

$$
P(f_x,f_y)=
\mathrm{circ}\left(
\frac{\sqrt{f_x^2+f_y^2}}{NA/\lambda}
\right)
\exp[i\Phi(f_x,f_y)].
$$

其中：

- cutoff spatial frequency 为 \(f_c=NA/\lambda\)；对非相干强度成像，MTF cutoff 常写为 \(2NA/\lambda\)。
- \(\Phi\) 可放入 defocus、astigmatism、coma、spherical aberration 等 Zernike 项。
- 若只验证焦深/景深，先保留 defocus phase 即可。

## 3. 焦深/景深的波动光学复现

焦栈的核心是不同焦平面 \(z_f\) 下的离焦相位变化。pupil 中的 defocus phase 可写为近似形式：

$$
\Phi_{defocus}(\rho;z)
=
\pi W_{20}(z)\rho^2,
$$

其中 \(\rho\) 是归一化 pupil 半径。更物理的标量角谱近似可写为：

$$
\Phi_{defocus}(f_x,f_y;z)
=
2\pi z
\left[
\sqrt{\frac{1}{\lambda^2}-f_x^2-f_y^2}
-\frac{1}{\lambda}
\right].
$$

在小角度条件下：

$$
\Phi_{defocus}
\approx
-\pi \lambda z(f_x^2+f_y^2).
$$

因此，模拟焦栈时不需要人为指定 blur radius，只需要扫描 \(z_f\)，每一层使用不同的离焦相位：

$$
P_z(f_x,f_y)=P_0(f_x,f_y)\exp[i\Phi_{defocus}(f_x,f_y;z_f)].
$$

再传播得到该焦平面的图像。

## 4. 景深数量级校准

显微系统常用景深近似：

$$
DOF
\approx
\frac{\lambda n}{NA^2}
+
\frac{n e}{M NA},
$$

其中 \(e\) 是探测器允许弥散直径或像元尺度对应项，\(M\) 是放大倍率。

若仅看波动项：

$$
\frac{\lambda}{NA^2}.
$$

对 \(\lambda=0.525\,\mu m\)：

NA=0.40：\(\lambda/NA^2 \approx 3.28\,\mu m\)，宽松轴向尺度 \(2\lambda/NA^2 \approx 6.56\,\mu m\)。
- NA=0.45：\(\lambda/NA^2 \approx 2.59\,\mu m\)，宽松轴向尺度 \(2\lambda/NA^2 \approx 5.19\,\mu m\)。

这给焦栈层间距提供约束：如果层间距远大于 3-6 µm，焦点峰会被欠采样；如果层间距明显小于该尺度，焦曲线能更连续地反映离焦过程。

## 5. 推荐实现层级

### Level 1：标量 Fourier optics 焦栈

适合当前投稿前验证。输入为反射复场，pupil 为圆孔 + defocus phase + 少量 Zernike 像差。每一层输出：

$$
I_z(x,y)=|\mathcal F^{-1}\{
\mathcal F[E_{obj}]
P_z
\}|^2.
$$

优点：实现快、参数可解释、能复现焦深、离焦、干涉条纹、局部高反射伪影。  
限制：部分相干 LED、物镜矢量效应和真实多层镀膜只能近似。

### Level 2：部分相干 Hopkins / source integration

把 LED 看作有限角度和有限带宽的源。对多个照明角度、多个波长分别计算 \(I_z\)，再做加权平均：

$$
I_z = \sum_{\lambda}\sum_s w_{\lambda,s} I_z(\lambda,s).
$$

优点：更接近 LED 同轴照明，能解释为什么干涉伪影会被削弱但不一定消失。  
限制：计算量提高，需要估计 LED 带宽、照明 NA、光源尺寸。

### Level 3：Debye / Richards-Wolf 矢量衍射

当 NA 较高、偏振和金属反射方向性很重要时使用。该模型把 pupil 上每个角谱分量的偏振旋转和焦场矢量叠加考虑进去。  
对 NA≈0.40 的 20X 干式物镜，投稿前通常可以先不走到这一层。

## 6. 与当前项目故事线的关系

完整波动光学仿真可以支撑三个论点：

1. **Simulation-to-Real 更可信。** 合成焦栈不再只靠经验 blur，而是由表面高度、金属反射、NA、离焦相位和部分相干共同产生。
2. **反光伪影有物理来源。** 高亮/晕影/焦点评价异常可由反射复场相干叠加与有限 NA 接收解释。
3. **焦深约束模型训练策略。** 焦层间距、焦栈层数和 DFF/GADFF 先验可靠性可由 \(\lambda/NA^2\) 的轴向尺度解释。

## 7. 下一步建议

建议下一轮实现 `full_wave_focus_stack_simulator.py`：

- 复用当前任务中的金属反射复场；
- 添加 Fourier pupil；
- 扫描 17 个焦平面；
- 输出 `focus_stack_wave_optics.npy`、代表层 PNG、焦点曲线和 DFF 峰值偏移图；
- 对比经验 defocus simulator 与 wave-optics simulator 的焦点峰差异。

这样可以直接回答：在相同表面结构下，经验 PSF 仿真和波动光学仿真是否给出不同的伪焦点、晕影和高反射边缘响应。
