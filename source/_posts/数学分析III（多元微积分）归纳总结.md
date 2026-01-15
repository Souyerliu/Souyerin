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
>  ~~另注：由于期末考试重点考后三章（二十~二十二章），所以前四章题目会相对缩减。~~
>  sry，时间不够了，后面几章的题目量可能还会少些……
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
   （源自教材19.2 例5）
   > 因为
   > $$
      \frac{\sin bx-\sin ax}{x}=\int_a^b\cos xydy
   > $$
   > 所以
   > $$
     J=\int_0^\infty e^{-px}dx\int_a^b\cos xydy=\int_0^\infty dx\int_a^be^{-px}\cos xydy
   > $$
   > 又因为$|e^{-px}\cos xy|<e^{-px}$且$\int_0^\infty e^{-px}dx$收敛，故由M判别法，$\int_a^be^{-px}\cos xydx$在$[a,b]$上一致收敛。又因为$e^{-px}\cos xy$在$[0,+\infty)\times[a,b]$上连续，故可交换积分顺序，故
   > $$
      J=\int_a^bdy\int_0^\infty e^{-px}\cos xydx=\int_a^b\frac{p}{p^2+y^2}dy=\arctan\frac{b}{p}-\frac{a}{p}
   > $$
   > 特别地，当$p=1,a=0$时，$\int_0^\infty e^{-t}\frac{\sin xt}{t}dt=\arctan x$.
5. 证明下列函数关于$\alpha$在集合$-1<\alpha<1$上内闭一致收敛： {.quiz .fill}
   $$
   \int_{0}^{\infty}\frac{\cos{x^2}}{x^\alpha}dx
   $$
   > 设$x^2=t$，则$dx=\frac{1}{2\sqrt{t}}dt$，故
   > $$ 
      \begin{aligned}
         \int_{0}^{\infty}\frac{\cos{x^2}}{x^\alpha}dx&=\frac{1}{2}\int_{0}^{\infty}\frac{\cos t}{t^\frac{\alpha+1}{2}}dt\\
         &\xlongequal{\beta=\frac{\alpha+1}{2}}\frac{1}{2}\int_{0}^{\infty}t^{-\beta}\cos tdt\\
         &=\frac{1}{2}\left(\int_{0}^{1}t^{-\beta}\cos tdt+\int_{1}^{\infty}t^{-\beta}\cos tdt\right)\\
         &\triangleq J_1(\beta)+J_2(\beta)
      \end{aligned}
   > $$
   > 其中$\beta\in (0,1)$。   
   > 对于$J_1(\beta)$，因为$\forall 0<\beta_1<\beta_2<1$，
   > $$
      \begin{aligned}
         J_1(\beta)&=\int_{0}^{1}t^{-\beta}\cos tdt\\
         &\leq \int_{0}^{1}t^{-\beta_2}\cos tdt
         &\leq \int_{0}^{1}t^{-\beta_2}dt
      \end{aligned}
   > $$
   > 又$0<\beta_2<1$，故$\int_{0}^{1}t^{-\beta_2}dt$收敛，由M判别法得$J_1(\beta)$在$[\beta_1,\beta_2]$上一致收敛；
   > 对于$J_2(\beta)=\int_{1}^{\infty}t^{-\beta}\cos tdt$，因为
   > $$
      \int_{1}^{\infty}\cos tdt
   > $$
   > 关于$\beta$一致有界，而$t^{-\beta}$在$[1,\infty]$单调且关于$\beta$一致趋于$0$，故由狄利克雷判别法，$J_2(\beta)$在$[\beta_1,\beta_2]$上一致收敛。   
   > 综上，$J_1(\beta)+J_2(\beta)$在$\beta\in(0,1)$上内闭一致收敛，即原积分在$-1<\alpha<1$上内闭一致收敛。
1. **曲线积分：** {.quiz}
2. 若曲线以极坐标$\rho=\rho(\theta)(\theta_1\leq\theta\leq\theta_2)$表示，试给出计算$\int_L f(x,y)ds$的公式，并用此公式计算下列曲线积分：{.quiz .fill}
   $$
   \int_L xds,\text{其中}L\text{为对数螺线}\rho=ae^{k\theta}(k>0)\text{在圆}r=a\text{内的部分}.
   $$
   > 公式推导：由极坐标方程得$x=\rho(\theta)\cos\theta,y=\rho(\theta)\sin\theta$，故
   > $$
      \begin{aligned}
         ds&=\sqrt{\left(\frac{dx}{d\theta}\right)^2+\left(\frac{dy}{d\theta}\right)^2}\\
         &=\sqrt{(-\rho(\theta)\sin\theta+\rho'(\theta)\cos\theta)^2+(\rho(\theta)\cos\theta+\rho'(\theta)\sin\theta)^2}\\
         &=\sqrt{\rho^2(\theta)+(\rho'(\theta))^2}
      \end{aligned}
   > $$
   > 所以
   > $$
      \int_L f(x,y)ds=\int_{\theta_1}^{\theta_2}f(\rho(\theta)\cos\theta,\rho(\theta)\sin\theta)\sqrt{\rho^2(\theta)+(\rho'(\theta))^2}
   > $$
   > 计算：
   > $$
   \begin{aligned}
      \int_L xds&=\int_{-\infty}^0 ae^{k\theta}\cos\theta\sqrt{(ae^{k\theta})^2+(ake^{k\theta})^2}d\theta\\
      &=a^2\sqrt{1+k^2}\int_{\infty}^0 e^{2k\theta}\cos\theta d\theta\\
      &=a^2\sqrt{1+k^2}\cdot e^{2k\theta}\left.\frac{\sin\theta+2k\cos\theta}{4k^2+1}\right|_{\infty}^0\\
      &=\frac{2ka^2\sqrt{1+k^2}}{4k^2+1}
   \end{aligned}
   > $$
2. 计算下述曲线积分： [$-\frac{\pi}{4}a^3$]{.gap} {.quiz .fill}
   $$
   \int_L y^2dx+z^2dy+x^2dz
   $$
   $L$是维维安尼曲线$x^2+y^2+z^2=a^2,x^2+y^2=ax(z\geq 0,a>0)$；若从$x$轴正向看去，$L$是逆时针方向进行的。
   > 设$x=a\sin^2t,y=a\sin t\cos t,z=a\cos t$，其中$t:\frac{\pi}{2}\to-\frac{\pi}{2}$ **（关键一步）**
   > 于是
   > $$
      \begin{aligned}
         &\int_L y^2dx+z^2dy+x^2dz\\
         =&\int_{\frac{\pi}{2}}^{-\frac{\pi}{2}}a^2\sin^2t\cos^2t\cdot 2a\cos t\sin t+a^2\cos^2 t\cdot a\cos 2t-a^2\sin^4 t\cdot a\sin tdt\\
         =&a^3\int_{\frac{\pi}{2}}^{-\frac{\pi}{2}}(2\sin^3t\cos^3t+\cos^4 t-\cos^2 t\sin^2 t-\sin^5 t)dt\\
         =&-2a^3\int_0^{-\frac{\pi}{2}}(\cos^4 t-\cos^2 t\sin^2 t)dt\\
         =&-\frac{\pi}{4}a^3
      \end{aligned}
   > $$ 
3. **多重积分：** {.quiz}
4. 设平面区域$D$在$x$轴和$y$轴的投影长度分别为和$l_x$和$l_y$，$D$的面积为$S_D$，$(\alpha,\beta)$为$D$内任一点，证明：     
   （1）{.quiz .fill} 
   $$
   \left|\iint_D (x-\alpha)(y-\beta)d\sigma\right|\leq l_xl_yS_D
   $$
   > 证明：设D在$x$轴与$y$轴投影分别为$[a,b]$与$[c,d]$，则$l_x=b-a,l_y=d-c$，所以
   > $$
      \left|\iint_D (x-\alpha)(y-\beta)d\sigma\right|\leq\iint_D |x-\alpha||y-\beta|d\sigma\leq l_xl_yS_D.
   > $$
   （2）{.quiz .fill} 
   $$
   \left|\iint_D (x-\alpha)(y-\beta)d\sigma\right|\leq\frac{l_x^2l_y^2}{4}
   $$
   > 证明：因为
   > $$
      \left|\iint_D (x-\alpha)(y-\beta)d\sigma\right|\leq\int_a^b|x-\alpha|dx\cdot\int_c^d|y-\beta|dy,
   > $$
   > 对于$\int_a^b|x-\alpha|dx$，因为
   > $$
      \begin{aligned}
         \int_a^b|x-\alpha|dx&=\int_a^\alpha(x-\alpha)dx+\int_\alpha^b(\alpha-x)dx\\
         &=\left.\frac{1}{2}x^2-\alpha x\right|_a^\alpha-\left.\frac{1}{2}x^2-\alpha x\right| _\alpha^b\\
         &=\frac{(\alpha-a)^2+(b-\alpha)^2}{2}\\
         &\xlongequal{u=\alpha-a}\frac{u^2+(b-a-u)^2}{2}\\
         &\leq \frac{l_x^2}{2}
      \end{aligned}
   > $$
   > 同理有$\int_a^b|y-\beta|dy\leq\frac{l_y^2}{2}$，故得证。

5. 运用格林公式计算双纽线$(x^2+y^2)^2=a^2(x^2-y^2)$所围的平面面积。{.quiz .fill}
   > 先转化为极坐标：设$x=r\cos\theta,y=r\sin\theta$，则
   > $r^4=a^2r^2(\cos^2\theta-\sin^2\theta)\Longrightarrow r^2=a^2\cos2\theta$（当$-\frac{\pi}{4}\leq\theta\leq\frac{\pi}{4}$时，对应右半片），故
   > $$
      \begin{aligned}
         S&=\frac{1}{2}\oint xdy-ydx\\
         &=\int_{-\frac{\pi}{4}}^{\frac{\pi}{4}}\left[a \sqrt{\cos2\theta}\cos\theta \left(a \sqrt{\cos2\theta}\cos\theta-\frac{a\sin2\theta\sin\theta}{\sqrt{\cos2\theta}}\right)\right] \\
         &-a \sqrt{\cos2\theta}\sin\theta\Big(-a \sqrt{\cos2\theta}\sin\theta-\frac{a\sin2\theta\cos\theta}{\sqrt{\cos2\theta}}\Big)\Big]d\theta \\
         &=a^{2}\int_{-\frac{\pi}{4}}^{\frac{\pi}{4}}\cos2\theta d\theta=a^{2}.
      \end{aligned}
   > $$
   > 注：实际上也可以直接用极坐标面积公式：$S=\frac{1}{2}\oint r^2d\theta$，简单证明如下：   
   > 因为$dx=\cos\theta dr-r\sin\theta d\theta,dy=\sin\theta dr+r\cos\theta d\theta$，所以
   > $$
      \begin{aligned}
         xdy-ydx & =(r\cos\theta)(\sin\theta dr+r\cos\theta d\theta)-(r\sin\theta)(\cos\theta dr-r\sin\theta d\theta) \\
         & =(r\sin\theta\cos\theta dr+r^2\cos^2\theta d\theta)-(r\sin\theta\cos\theta dr-r^2\sin^2\theta d\theta) \\
         & =r^2(\cos^2\theta+\sin^2\theta)d\theta \\
         & =r^2d\theta
      \end{aligned}
   > $$
   > 那么此题就可以直接得到$a^{2}\int_{-\frac{\pi}{4}}^{\frac{\pi}{4}}\cos2\theta d\theta=a^2$.
1. **曲面积分：** {.quiz}
2. 计算下列第一型曲面积分：{.quiz .fill}  
   $$
   \iint_S(x^2+y^2)dS
   $$
   其中$S$为立体$\sqrt{x^2+y^2}\leq z\leq 1$的边界曲面。
   > 将$S$分为$S_1:z=\sqrt{x^2+y^2}$和$S_2:z=1,x^2+y^2\leq 1$，二者在$xy$平面上投影均为$x^2+y^2\leq 1$，于是
   > $$
      \begin{aligned}
      \iint_{s}(x^{2}+y^{2}) dS & =\iint_{S_{1}}(x^{2}+y^{2}) dS+\iint_{S_{2}}(x^{2}+y^{2}) dS \\
      & =\sqrt{2}\iint_{x^{2}+y^{2}\leqslant1}(x^{2}+y^{2})dxdy+\iint_{x^{2}+y^{2}\leqslant1}(x^{2}+y^{2})dxdy \\
      & =\sqrt{2}\int_{0}^{2\pi}d\theta \int_{0}^{1}r^{3} dr+\int_{0}^{2\pi}d\theta \int_{0}^{1}r ^{3} dr \\
      & =\frac\pi2(\sqrt{2}+1).
      \end{aligned}
   > $$
   > 注意不要漏掉$S_2$ {.mistake}
3. 计算下列第二型曲面积分：{.quiz .fill}
   $$
   \iint_S x^2dydz+y^2dzdx+z^2dxdy
   $$
   其中$S$是球面$(x-a)^2+(y-b)^2+(z-c)^2=R^2$并取外侧为正向。
   > 由轮换对称性，只需计算$\iint_S z^2dxdy$.   
   > 由$z-c=\pm\sqrt{R^{2}-(x-a)^{2}-(y-b)^{2}}$，利用极坐标变换可得
   > $$
      \begin{aligned}
      & \iint_{S}x^{2}\mathrm{d}x\mathrm{d}y=\iint_{D_{xy},(x-a)^{2}+(y-b)^{2}\leqslant R^{2}}(c+\sqrt{R^{2}-(x-a)^{2}-(y-b)^{2}})\mathrm{d}x\mathrm{d}y \\
      & -\iint_{D_{xy}}(c-\sqrt{R^{2}-(x-a)^{2}-(y-b)^{2}})\mathrm{d}x\mathrm{d}y \\
      & =4c\int_{0}^{2\pi}\mathrm{d}\varphi\int_{0}^{R}\sqrt{R^{2}-r^{2}}\mathrm{d}r=\frac{8}{3}\pi R^{3}c,
      \end{aligned}
   > $$
   > 故原式=$\frac{8}{3}\pi R^{3}(a+b+c)$.