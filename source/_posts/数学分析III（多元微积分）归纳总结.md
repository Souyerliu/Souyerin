---
title: 数学分析III（多元微积分）归纳总结
date: 2026-01-08 10:08:46
categories:
 - study
 - [数学,微积分] 
tags: 
 - 数学分析
 - 微积分
 - 多元积分
 - 多元微分
cover: folder.png
quiz: true
---
> **说明：**（既然有了概率论，那么也顺便把数分III梳理一下~）考虑了一下，还是以题目汇编的形式进行总结复习，通过解题梳理知识点。（题目来源：课后习题、答案书题目、期中考试题目）   
> 另注：由于期末考试重点考后三章（二十~二十二章），所以前四章题目会相对缩减。
1. **多元函数极限与连续：** {.quiz}
2. 请写出下面这个平面点集的聚点和界点：{.quiz .fill}
   $$
   \{(x,y)\mid y=\sin\frac{1}{x},x>0\}
   $$
   > 聚点和界点均为$(x,y)\mid y=\sin\frac{1}{x},x>0$ 和 $(x,y)\mid x=0,-1\leq y\leq 1$  
   > 后面这个容易漏{.mistake}
3. 确定下列多元函数极限是否存在，若存在求出相应的极限值：{.quiz .fill}
   $$
   \begin{aligned}
      &(1)\hspace{1em}\lim_{(x,y)\to(0,0)}\frac{\sin xy}{\sqrt{x^2+y^2}}\\
      &(2)\hspace{1em}\lim_{(x,y)\to(0,0)}(x^2+y^2)^{x^2y^2}
   \end{aligned}
   $$
   > $(1)$：因为
   > $$
      \begin{aligned}
         \left|\frac{\sin xy}{\sqrt{x^2+y^2}}\right|&\leq\left|\frac{xy}{\sqrt{x^2+y^2}}\right|\\
         &\leq\frac{1}{2}\left|\frac{x^2+y^2}{\sqrt{x^2+y^2}}\right|=\frac{1}{2}\sqrt{x^2+y^2}\to 0\hspace{1em}((x,y)\to (0,0))
      \end{aligned}
   > $$
   > 所以$\lim_{(x,y)\to(0,0)}\frac{\sin xy}{\sqrt{x^2+y^2}}$存在且为$0$；  
   > $(2)$：设$S=(x^2+y^2)^{x^2y^2}$，则$\ln S=x^2y^2\ln(x^2+y^2)$。又因为
   > $$
      \begin{aligned}
         0\leq |x^2y^2\ln(x^2+y^2)|&\leq\frac{(x^2+y^2)^2}{2}|\ln(x^2+y^2)|\\
         &\xlongequal{\rho=\sqrt{x^2+y^2}}\rho^4|\ln\rho|\to 0(\rho\to 0)
      \end{aligned}
   > $$
   > 所以$\lim_{(x,y)\to(0,0)}\ln S$，存在且为$0$，$\lim_{(x,y)\to(0,0)}(x^2+y^2)^{x^2y^2}=1$.
3. 判断下列函数在$(0,0)$处的连续性、任意方向导数以及全微分的存在性，若存在则求解其相应的任意方向导数值以及全微分： {.quiz .fill}
   $$
   f(x,y)=\left\{\begin{aligned}
      &\frac{4xy^3}{x^2+y^4},&(x,y)\neq (0,0)\\
      &0,&(x,y)=(0,0)
   \end{aligned}
   \right.
   $$
   > 连续性：因为
   > $$
      \lim_{(x,y)\to(0,0)}\frac{|4xy^3|}{x^2+y^4}\leq\frac{4|xy^3|}{2|xy^2|}=2|y|\to 0=f(0,0)
   > $$
   > 所以$f(x,y)$在$(0,0)$处连续；   
   > 任意方向导数：
   > 设单位方向向量为$(\cos\theta,\sin\theta)$，则
   > $$
      \begin{aligned}
         D_u f(x,y)&=\lim_{t\to 0}\frac{f(t\cos\theta,t\sin\theta)-f(0,0)}{t}\\
         &=\lim_{t\to 0}{\frac{4t\cos\theta\sin^3\theta}{\cos^2\theta+t^2\sin^4\theta}}=0
      \end{aligned}
   > $$
   > 即任意方向方向导数为$0$。   
   > 全微分存在性：
   > 由于任意方向导数均为$0$，故$f_x(0,0)=f_y(0,0)=0$，若全微分存在（即$f$在$(0,0)$可微）,则
   > $f(\Delta x,\Delta y)=f_x(0,0)\Delta x+f_y(0,0)\Delta y+o(\sqrt{(\Delta x)^2+(\Delta y)^2})=o(\sqrt{(\Delta x)^2+(\Delta y)^2})$  
   > 而
   > $$
      \begin{aligned}
         &\lim_{(\Delta x,\Delta y)\to(0,0)}\frac{f(\Delta x,\Delta y)}{\sqrt{(\Delta x)^2+(\Delta y)^2}}\\
         &=\lim_{(\Delta x,\Delta y)\to(0,0)}\frac{4\Delta x(\Delta y)^3}{((\Delta x)^2+(\Delta y)^4)\sqrt{(\Delta x)^2+(\Delta y)^2}}
      \end{aligned}
   > $$
   > 取$\Delta x=(\Delta y)^2$，则
   > $$
   \begin{aligned}
      &\lim_{(\Delta x,\Delta y)\to(0,0)}\frac{4\Delta x(\Delta y)^3}{((\Delta x)^2+(\Delta y)^4)\sqrt{(\Delta x)^2+(\Delta y)^2}}\\
      &=\lim_{(\Delta x,\Delta y)\to(0,0)}\frac{2}{\sqrt{(\Delta y)^2+1}}=2\neq 0
   \end{aligned}
   > $$
   > 故$f(x,y)$在$(0,0)$处不可微，全微分不存在。
1. **多元函数微分学：** {.quiz}
2. 求函数$u=xyz$在沿点$A(5,1,2)$到点$B(9,4,14)$的方向$\overrightarrow{AB}$上的方向导数. [$\frac{98}{13}$]{.gap} {.quiz .fill}
   > 易知$u$在$A$点偏导数存在，又$\overrightarrow{AB}=(4,3,12)$，对应单位方向向量为$(\frac{4}{13},\frac{3}{13},\frac{12}{13})$，$f_x=yz,f_y=xz,f_z=xy$，故
   > $$
   \begin{aligned}
      f_{\overrightarrow{AB}}(A)&=f_x(A)\cos\alpha+f_y(A)\cos\beta +f_z(A)\cos\gamma\\
      &=2\times\frac{4}{13}+10\times\frac{3}{13}+5\times\frac{12}{13}=\frac{98}{13}.
   \end{aligned}
   > $$
3. 设$u=f(\frac{x}{y},\frac{y}{z})$，求$\frac{\partial u}{\partial x},\frac{\partial u}{\partial y},\frac{\partial u}{\partial z}$。{.quiz .fill}
   > 设$\frac{x}{y}=p,\frac{y}{z}=q$，则
   > $$
   \begin{aligned}
      \frac{\partial u}{\partial x}&=\frac{\partial u}{\partial p}\frac{\partial p}{\partial x}+\frac{\partial u}{\partial q}\frac{\partial q}{\partial x}\\
      &=\frac{1}{y}f_1,
   \end{aligned}
   > $$
   > $$
   \begin{aligned}
      \frac{\partial u}{\partial y}&=\frac{\partial u}{\partial p}\frac{\partial p}{\partial y}+\frac{\partial u}{\partial q}\frac{\partial q}{\partial y}\\
      &=-\frac{x}{y^2}f_1+\frac{1}{z}f_2,
   \end{aligned}
   > $$
   > $$
   \begin{aligned}
      \frac{\partial u}{\partial z}&=\frac{\partial u}{\partial p}\frac{\partial p}{\partial z}+\frac{\partial u}{\partial q}\frac{\partial q}{\partial z}\\
      &=-\frac{y}{z^2}f_2.
   \end{aligned}
   > $$
4. 已知函数$u=(xyz)e^{x+y+z}$，求$\frac{\partial^{p+q+r}u}{\partial x^p\partial y^q\partial z^r}$.  {.quiz .fill}
   > 注意到
   > $$
      \frac{\partial u}{\partial x}=(xyz+yz)e^{x+y+z}
   > $$
   > 所以
   > $$
      \frac{\partial^p u}{\partial x^p}=(xyz+pyz)e^{x+y+z}
   > $$
   > 同理可得
   > $$
      \frac{\partial^{p+q} u}{\partial x^p\partial y^q}=(xyz+pyz+qxz+pqz)e^{x+y+z},
   > $$
   > $$
      \frac{\partial^{p+q+r}u}{\partial x^p\partial y^q\partial z^r}=(xyz+pyz+qxz+pqz+rxy+rpy+rqx+pqr)e^{x+y+z}.
   > $$
5. 证明：{.quiz .fill}
   $$
   \begin{aligned}
   &(1)\hspace{1em} \operatorname*{grad}(uv)=u\operatorname*{grad}v+v\operatorname*{grad}u;\\
   &(2)\hspace{1em}\operatorname*{grad}f(u)=f'(u)\operatorname*{grad}u.
   \end{aligned}
   $$
   > 不失一般性，不妨设$u=u(x,y,z),v=v(x,y,z)$，则
   > $$
      \begin{aligned}
         \operatorname*{grad}(uv)&=(uv_x+vu_x,uv_y+vu_y,uv_z+vu_z)\\
         &=u(v_x,v_y,v_z)+v(u_x,u_y,u_z)\\
         &=u\operatorname*{grad}v+v\operatorname*{grad}u;
      \end{aligned}
   > $$
   > 此式推论：$\operatorname*{grad}\frac{1}{u}=-\frac{1}{u^2}\operatorname*{grad}u$.
   > $$
      \begin{aligned}
         \operatorname*{grad}f(u)&=(f'(u)u_x,f'(u)u_y,f'(u)u_z)\\
         &=f'(u)(u_x,u_y,u_z)\\
         &=f'(u)\operatorname*{grad}u;
      \end{aligned}
   > $$
6. 已知$z=f(x+y,xy,\frac{x}{y})$，求$z_x,z_{xx},z_{xy}$.{.quiz .fill}
   > $$
      z_x=f'_1+yf'_2+\frac{1}{y}f'_3,
   > $$
   > $$
      \begin{aligned}
         z_{xx}&=f''_{11}+yf''_{12}+\frac{1}{y}f''_{13}+y(f''_{21}+yf''_{22}+\frac{1}{y}f''_{23})+\frac{1}{y}(f''_{31}+yf''_{32}+\frac{1}{y}f''_{33})\\
         &=f''_{11}+2yf''_{12}+\frac{2}{y}f''_{13}+y^2f''_{22}+2f''_{23}+\frac{1}{y^2}f''_{33}
      \end{aligned}
   > $$
   > 同理可得
   > $$
      z_{xy}=f''_{11}+(x+y)f''_{12}+\frac{1}{y}(1-\frac{x}{y})f''_{13}+xyf''_{22}-\frac{x}{y^3}f''_{33}+f'_2-\frac{1}{y^2}f'_{3}
   > $$
1. **隐函数定理及其应用：** {.quiz}
2. 求由下列方程所确定的隐函数的导数：{.quiz .fill}
   $$
   \ln\sqrt{x^2+y^2}=\arctan\frac{y}{x},\text{求}\frac{dy}{dx}.
   $$
   > 两边对$x$求偏导得
   > $$
      \frac{x+y\frac{dy}{dx}}{x^2+y^2}=\frac{1}{1+(\frac{y}{x})^2}\left(-\frac{y}{x^2}+\frac{1}{x}\frac{dy}{dx}\right)
   > $$
   > 解得
   > $$
      \frac{dy}{dx}=\frac{x+y}{x-y}
   > $$
3. 已知$u=u(x,y),v=v(x,y)$满足下列方程组：{.quiz .fill}
   $$
   \left\{
      \begin{aligned}
         u=f(ux,v+y)\\
         v=g(u-x,v^2y)
      \end{aligned}
   \right.
   $$
   求$\frac{\partial u}{\partial x},\frac{\partial v}{\partial x}$.
   > 第一个方程两边对$x$求偏导得：
   > $$
      \frac{\partial u}{\partial x}=\left(u+x\frac{\partial u}{\partial x}\right)f_1+\frac{\partial v}{\partial x}f_2
   > $$
   > 第二个方程两边对$x$求偏导得：
   > $$
      \frac{\partial v}{\partial x}=\left(\frac{\partial u}{\partial x}-1\right)g_1+\left(2yv\frac{\partial v}{\partial x}\right)g_2
   > $$ 
   > 整理两个方程得
   > $$
      \begin{aligned}
         (1-f_1x)u_x-f_2v_x=f_1u\\
         -g_1u_x+(1-2yvg_2)v_x=-g_1
      \end{aligned}
   > $$
   > 解得
   > $$
      \begin{aligned}
         u_x=\frac{f_1u(1- 2yvg_2)- f_2g_1}{(1-f_1x)(1-2yvg_2) - f_2g_1}\\
         v_x=\frac{f_1ug_1- g_1(1-f_1x)}{(1-f_1x)(1-2yvg_2) - f_2g_1}\\
      \end{aligned}
   > $$
   > !!算起来确实比较烦!!
4. 求下列函数组所确定的反函数组的偏导数： {.quiz .fill}
   $$
   \left\{\begin{aligned}
      &x=u+v\\
      &y=u^2+v^2\\
      &z=u^3+v^3
   \end{aligned}
   \right.
   $$
   求$z_x=$[$-3uv$]{.gap}。
   > 法1：
      三个方程两边分别对$x$求偏导，得：
   > $$
   \left\{\begin{aligned}
      &1=u_x+v_x\\
      &0=2u_xu+2v_xv\\
      &z_x=3u_xu^2+3v_xv^2
   \end{aligned}
   \right.
   > $$
   > 解得
   > $$
      v_x=\frac{u}{u-v},u_x=-\frac{v}{u-v},z_x=\frac{-3u^2v+3v^2u}{u-v}=-3uv.
   > $$
   > 法2：!!严格来说上面的方法属于“投机取巧”，并未完全使用反函数组的性质!!  
   > 由二对二的隐函数组定理，我们可以推导三对二的隐函数组定理：由
   > $$
      \begin{aligned}
         dx=u_xdu+v_xdv\\
         dy=u_ydu+v_ydv\\
         dz=u_zdu+v_zdv
      \end{aligned}
   > $$  
   > 为求$z_x$，固定$y$不变，则$dy=0$，解前两个方程得
   > $$
      du=\frac{y_v}{\frac{\partial(x,y)}{\partial(u,v)}}dx,dv=-\frac{y_u}{\frac{\partial(x,y)}{\partial(u,v)}}dx
   > $$
   > 于是
   > $$
      dz=u_zdu+v_zdv=\frac{z_uy_v-z_vy_u}{\frac{\partial(x,y)}{\partial(u,v)}}dx=\frac{\frac{\partial(z,y)}{\partial(u,v)}}{\frac{\partial(x,y)}{\partial(u,v)}}dx
   > $$
   > 即$z_x=\frac{\frac{\partial(z,y)}{\partial(u,v)}}{\frac{\partial(x,y)}{\partial(u,v)}}$.又因为
   > $$
      \frac{\partial(x,y)}{\partial(u,v)}=\left|\begin{matrix}
         1&1\\
         2u&2v
      \end{matrix}
      \right|=2(v-u),
      \frac{\partial(z,y)}{\partial(u,v)}=\left|\begin{matrix}
         3u^2&3v^2\\
         2u&2v
      \end{matrix}
      \right|=6uv(u-v)
   > $$
   > 所以$z_x=\frac{6uv(u-v)}{2(v-u)}=-3uv.$
5. 已知空间中的一个平面 {.quiz .fill}
   $$
   \left\{\begin{aligned}
      x=a\cos\psi\cos\varphi\\
      y=b\cos\psi\sin\varphi\\
      z=\sin\psi
   \end{aligned}
   \right.
   $$
   求其过点$(\varphi_0,\psi_0)$的切平面与法线方程。
   > 由法向量公式：
   > $$
      \begin{aligned}
       \mathbf{n}&=\pm\left(\frac{\partial(y,z)}{\partial(\psi,\varphi)},\frac{\partial(z,x)}{\partial(\psi,\varphi)},\frac{\partial(x,y)}{\partial(\psi,\varphi)}\right)\\
       &=\pm(−b\cos^2\psi\cos\varphi,-a\cos^2\psi\sin\varphi,-ab\sin\psi\sin\varphi)
      \end{aligned}
   > $$
   > 故$\mathrm{n}(\varphi_0,\psi_0)=(b\cos^2\psi_0\cos\varphi_0,a\cos^2\psi_0\sin\varphi_0,ab\sin\psi_0\sin\varphi_0)$   
   > 切平面：
   > $$
      b\cos^2\psi_0\cos\varphi_0(x-x_0)+a\cos^2\psi_0\sin\varphi_0(y-y_0)+ab\sin\psi_0\sin\varphi_0(z-z_0)=0
   > $$
   > 法线：
   > $$
      \frac{x-x_0}{b\cos^2\psi_0\cos\varphi_0}=\frac{y-y_0}{a\cos^2\psi_0\sin\varphi_0}=\frac{z-z_0}{ab\sin\psi_0\sin\varphi_0(z-z_0)}
   > $$
6. 求函数$f(x,y,z)=4x^2+y^2+5z^2$在平面$2x+3y+5z=12$上的最小值点。{.quiz .fill}
   > 设$g(x,y,z)=2x+3y+5z-12=0$，构造拉格朗日函数
   > $$
      \mathcal{L}(x,y,z,\lambda)=4x^2+y^2+5z^2+\lambda(2x+3y+5z-12)=0
   > $$
   > 求偏导，构造方程组：
   > $$
      \left\{\begin{aligned}
         &\mathcal{L}_x=8x+2\lambda=0\\
         &\mathcal{L}_y=2y+3\lambda=0\\
         &\mathcal{L}_z=10z+5\lambda=0\\
         &2x+3y+5z-12=0
      \end{aligned}
      \right.
   > $$
   > 解得
   > $$
      \left\{\begin{aligned}
         &x=-\frac{\lambda}{4}=\frac{2}{5}\\
         &y=-\frac{3\lambda}{2}=\frac{12}{5}\\
         &z=-\frac{\lambda}{2}=\frac{4}{5}\\
         &\lambda=-\frac{8}{5}
      \end{aligned}
      \right.
   > $$
   > 得到稳定点$(\frac{2}{5},\frac{12}{5},\frac{4}{5})$，又因为Hessian矩阵$\mathrm{diag}(8,2,10)$为正定矩阵，所以也为最小值点。
1. **含参量积分：** {.quiz}
2. 设$F(x,y)=\int_{\frac{x}{y}}^{xy}(x-yz)f(z)dz$，其中$f(z)$为可微函数，求$F_{xy}(x,y)$。{.quiz .fill}
3. 运用对参量的微分法，求下列积分：{.quiz .fill}
   $$
   \int_{0}^{\frac{\pi}{2}}\ln(a^2\sin^2x+b^2\cos^2x)dx\hspace{1em}(a^2+b^2\neq 0)
   $$
   > 分情况讨论：  
   > 当$|a|=0,|b|>0$时，原式=
   > $$
      \begin{aligned}
        \int_{0}^{\frac{\pi}{2}}\ln(b^2\cos^2x)dx&=\pi\ln|b|+2\int_{0}^{\frac{\pi}{2}}\ln(\cos x)dx\\
        &=\pi\ln|b|-\pi\ln2=\pi\ln\frac{|b|}{2},
      \end{aligned}
   > $$
   > 其中
   > $$
      \begin{aligned}
         &2\int_{0}^{\frac{\pi}{2}}\ln(\cos x)dx\\
         &=\int_{0}^{\frac{\pi}{2}}\ln(\cos x)dx+\int_{0}^{\frac{\pi}{2}}\ln(\sin x)dx\\
         &=\int_{0}^{\frac{\pi}{2}}\ln(\sin 2x)dx-\frac{\pi}{2}\ln2\\
         &=\frac{1}{2}\int_{0}^{\pi}\ln(\sin t)dt-\frac{\pi}{2}\ln2\\
         &=\int_{0}^{\frac{\pi}{2}}\ln(\sin x)dx-\frac{\pi}{2}\ln2\\
         &\Longrightarrow \int_{0}^{\frac{\pi}{2}}\ln(\cos x)dx=-\frac{\pi}{2}\ln2
      \end{aligned}
   > $$
   > 同理可得$|a|>0,|b|=0$时，原式=$\pi\ln\frac{|a|}{2}$，
   > 当$|a|\neq 0,|b|\neq 0$时，设
   > $$
      I(b)=\int_0^{\frac{\pi}{2}}\ln(a^2\sin^2x+b^2\cos^2x)dx
   > $$
   > 则对$b$求导，
   > $$
      \begin{aligned}
         I'(b)&=\int_0^{\frac{\pi}{2}}\frac{2b\cos^2x}{a^2\sin^2x+b^2\cos^2x}dx\\
         &=\frac{2}{b}\int_0^{\frac{\pi}{2}}\frac{1}{(\frac{a}{b})^2\tan^2x+1}dx=\frac{2}{b}\cdot\frac{\pi}{2(1+|\frac{a}{b}|)}\\
         &=\frac{\pi}{|a|+|b|}
      \end{aligned}
   > $$
   > 又因为
   > $$
      I(0)=\int_{0}^{\frac{\pi}{2}}\ln(a^2\sin^2x)dx=\pi\ln\frac{|a|}{2},
   > $$
   > 所以
   > $$
      I(b)=I(0)+\int_0^b\frac{\pi}{|a|+t}dt=\pi\ln\frac{|a|+|b|}{2}
   > $$
   > 综上，原式值为$\pi\ln\frac{|a|+|b|}{2}$.
4. 计算 {.quiz .fill}
   $$
   J=\int_0^\infty e^{-px}\frac{\sin bx-\sin ax}{x}dx\hspace{1em}(p>0,b>a).
   $$
5. 证明下列函数关于$\alpha$在集合$-1<\alpha<1$上内闭一致收敛： {.quiz .fill}
   $$
   \int_{0}^{\infty}\frac{cos{x^2}}{x^\alpha}
   $$
