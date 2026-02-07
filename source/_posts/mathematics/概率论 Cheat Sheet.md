---
title: 概率论 Cheat Sheet
date: 2025-12-18 08:37:42
categories:
 - study
 - [数学,概率论] 
tags: 
 - 概率论
 - 随机变量
cover: folder.png
---
快要期末了，笔者发现好多题目不够熟练，故紧急把一些知识点整理成cheatsheet，也相当于复习了。

# 概率空间
+ 样本点、样本空间、随机事件、事件域、样本空间分割的概念（略，看书）
+ 概率的四种确定方法+公理化定义（见下）：
  1. 非负性：$P(A)\geq 0$；
  2. 正则性：$P(\Omega)=1$；
  3. 可列可加性：当$A_1,\cdots,A_n,\cdots$互不相容，有
   $$
   P(\sum_{n=1}^\infty A_n)=\sum_{n=1}^\infty P(A_n)
   $$
+ $P$本质为集函数（将集合映射到实数域）
+ 概率空间：$(\Omega,\mathcal{F},P)$（下面如无特别说明，均在此空间下讨论）
## 概率的性质
+ 对立事件概率、有限可加性、可减性、单调性、有界性、加法公式（容斥定理，庞加莱公式）、次可加性
+ 概率为$0$的事件不一定为不可能事件
+ 可列可加性$\Longrightarrow$有限可加性，反之不然
+ 概率的连续性：
  + 极限事件：
    + 对于单调不减事件列（$A_1\subset A_2\subset\cdots\subset A_n\subset\cdots$），为
    $$
    \lim_{n\to\infty}A_n=\bigcup_{n=1}^\infty A_n
    $$
    + 对于单调不增事件列（$A_1\supset A_2\supset\cdots\supset A_n\supset\cdots$），则为
    $$
    \lim_{n\to\infty}A_n=\bigcap_{n=1}^\infty A_n
    $$
  + 集函数$\mu$的上（下）连续性：对于单调不增（减）的事件列$\{A_n,n\geq 1\}$满足
    $$
    \mu(\lim_{n\to\infty}A_n)=\lim_{n\to\infty}\mu(A_n)
    $$
    （记忆：可以将事件包含关系理解为叠高楼，被包含的事件在上面）
  + 概率测度$P$满足上（下）连续性
+ 概率等价定义：可列可加性替换为有限可加性+连续性
## 常见概率模型
1. 不返回抽样（超几何模型）：
  $$
  P(A)=\frac{\binom{M}{m}\binom{N-M}{n-m}}{\binom{N}{n}}
  $$
2. 返回抽样
  $$
  P(A)=\binom{n}{m}\frac{M^m(N-M)^{n-m}}{N^n}=\binom{n}{m}(\frac{M}{N})^{m}(\frac{N-M}{N})^{n-m}
  $$
3. 盒子模型
  $$
  P(A)=\frac{P_N^n}{N^n}=\frac{N!}{N^n(N-n)!}
  $$
4. 配对模型（$n$人拿$n$个帽子，一人一个，至少一人拿到自己帽子的概率）
  $$
  \begin{aligned}
  P(\bigcup_{k=1}^n A_k)&=\sum_{k=1}^n P(A_k)-\sum_{i<j}P(A_iA_j)+\cdots+(-1)^{n-1}P(A_1A_2\cdots A_n)  \\
  &=\sum_{k=1}^n (-1)^{k-1}\frac{1}{k!}
  \end{aligned}
  $$
## 随机事件的独立性
+ 定义：$P(AB)=P(A)\cdot P(B)$
+ 独立事件不一定互斥（如掷骰子，$A=\{1,2,3\},B=\{1,4\}$），互斥事件不一定独立（比如两个事件概率均不为$0$）
+ 多个随机事件的独立：
  + 两两独立、三三独立、相互独立（简称独立）
  + 若$A,B,C$相互独立，则$A\cup B$与$C$独立，$A\cap B$与$C$独立，$A\backslash B$与$C$独立
+ 相互独立条件下的庞加莱公式：
  $$
  P(\bigcup_{k=1}^m A_k)=1-\prod_{k=1}^n(1-P(A_k))
  $$
+ 事件类的独立
  + 有限个随机事件类的独立——每个事件类中任取一个随机事件结合，相互独立
  + 任意多个随机事件类独立——基于有限子集事件类的独立性
+ 事件域的独立性

## 条件概率
+ 定义：在概率空间中，$B\in\mathcal{F}$且$P(B)>0$，则$\forall A\in\mathcal{F}$，定义$P(A\mid B)=\frac{P(AB)}{P(B)}$
+ $P(A\mid\Omega)=P(A)$，$P(A\mid B)=P(A)\Longleftrightarrow A$与$B$相互独立
+ 其他性质：
  + $P(B\mid B)=1$；
  + 若$P(B)=1$，则$P(A\mid B)=P(A)$
  + 若$A\cap B=\emptyset$，则$P(A|B)=\frac{P(A)}{P(B)}$
+ 乘法公式：
  + $A,B\in\mathcal{F}$，且$P(A)>0,P(B)>0$，则
  $$
  P(AB)=P(A)P(B\mid A)=P(B)P(A\mid B)
  $$
  + 推论1：$n>1,A_1,A_2,\cdots,A_n\in\mathcal{F},P(A_1,\cdots A_{n-1})>0$，则有
  $$
  P(A_1\cdots A_n)=P(A_1)P(A_2\mid A_1)\cdots P(A_n\mid A_1\cdots A_{n-1})
  $$
  + 推论2：$B,A_1,\cdots,A_n\in\mathcal{F},P(A_1\cdots A_{n-1}B)>0$，则
  $$
  P(A_1\cdots A_n\mid B)=P(A_1\mid B)P(A_2\mid A_1B)\cdots P(A_n\mid A_1\cdots A_{n-1}B)
  $$
+ 全概率公式：
  + $\forall A,B\in\mathcal{F}$，$0<P(B)<1$，则
  $$
  P(A)=P(A\mid B)P(B)+P(A\mid\overline{B})P(\overline{B})
  $$
  + 推论：设$B_1,\cdots B_n$为$\Omega$的一组分割，$P(B_k)>0,k=1,\cdots,n$，则对任意$A\in\mathcal{F}$有
  $$
  P(A)=\sum_{k=1}^nP(B_k)P(A\mid B_k)
  $$
+ 贝叶斯公式（后验概率公式）：
  + 设$B_1,\cdots,B_n$为样本空间$\Omega$的一组分割，$P(B_k)>0,k=1,\cdots,n$，且有$A\in\mathcal{F},P(A)>0$，则有
  $$
  P(B_j\mid A)=\frac{P(B_j)P(A\mid B_j)}{\sum_{k=1}^n P(B_k)P(A\mid B_k)},j=1,\cdots,n
  $$

## *可测映射
+ $\sigma$-代数：设$\Omega$是给定的非空集合，$\mathcal{F}$是由的部分子集组成的集合类，若$\mathcal{F}$满足下列条件，则称$\mathcal{F}$为$\Omega$上$\sigma$-代数：
  1. $\Omega\in\mathcal{F}$；
  2. 若$A\in\mathcal{F}$ 则 $\overline{A}\in\mathcal{F}$；
  3. 对任意$n\geq 1$，若$A_n\in\mathcal{F}$ 则 $\bigcup_{n=1}^\infty A_n\in\mathcal{F}$.
+ 定义$\mathcal{P}(\Omega)=\{A:A\subset\Omega\}$（所有子集集合，有时也记为$2^\Omega$）；$\mathcal{D}(\Omega)=\{\mathcal{F}\subset\mathcal{P}(\Omega):\mathcal{F}\text{为集合}\Omega\text{上的}\sigma\text{-代数}\}$.
+ 设$\mathcal{C}\subset\mathcal{P}(\Omega)$，则称集类
  $$
  \bigcap_{\{\mathcal{F}\in\mathcal{D}(\Omega):\hspace{0.3em}\mathcal{C}\subset\mathcal{F}\}}\mathcal{F}
  $$
  为由$\mathcal{C}$生成的$\sigma$-代数，记为$\sigma(\mathcal{C})$。
+ 注：设$\mathcal{C}\subset\mathcal{P}(\Omega),A\subset\Omega$，记 
  $$
  \mathcal{C}\cap A=\{B\cap A:B\in\mathcal{C}\}
  $$
  记$\sigma(\mathcal{C}\cap A)$为$\mathcal{C}\cap A$所生成集合在$A$上的$\sigma$-代数，则有：
  $$
  \sigma(\mathcal{C})\cap A=\sigma(\mathcal{C}\cap A)
  $$
  这也是条件概率的理论基础。
+ Borel $\sigma$-代数：设$\sigma=\mathbb{R}^d$，
  $$
  \mathcal{C}=\{\prod_{i=1}^d(a_i,b_i]\subset\Omega:-\infty\leq a_i<b_i\leq\infty\}
  $$
  则称$\sigma(\mathcal{C})$为Borel $\sigma$-代数或Borel域，记为$\mathcal{B}(\mathbb{R}^d)$，称$\mathcal{B}(\mathbb{R}^d)$中的集合为borel集。（注：之后的随机变量定义中，采用$a_i=-\infty$）
+ 可测映射：
  + 设$f$为集合$\Omega_1$到$\Omega_2$上的映射，对任意$B\subset\Omega_2$，定义
    $$
    f^{-1}(B)=\{\omega\in\Omega_1:f(\omega)\in B\}
    $$
    即$f^{-1}$建立了$\mathcal{P}(\Omega_2)$到$\mathcal{P}(\Omega_1)$上的映射.   
    若$\mathcal{C}\subset \mathcal{P}(\Omega_2)$，记
    $$
    f^{-1}(\mathcal{C})=\{\omega\in\Omega_1:f(\omega)\in\mathcal{C}\}
    $$
  + 定理1：设$f$为$\Omega_1$到$\Omega_2$上的映射，$\mathcal{A}\in\mathcal{D}(\Omega_2)$，则$f^{-1}(\mathcal{A})\in\mathcal{D}(\Omega_1)$.
  + 定理2：设$f$为$\Omega_1$到$\Omega_2$上的映射，$\mathcal{C}\in\mathcal{P}(\Omega_2)$，则$f^{-1}(\sigma(\mathcal{C}))\in\mathcal{D}(\Omega_1)$，且$f^{-1}(\sigma(\mathcal{C}))=\sigma(f^{-1}(\mathcal{C}))$.
+ 可测空间：若$\mathcal{F}\in\mathcal{D}(\Omega)$，则称$(\Omega,\mathcal{F})$为可测空间。
+ 可测映射：设$(\Omega_1,\mathcal{F}_1)$和$(\Omega_2,\mathcal{F}_2)$是两个可测空间，$f$为$\Omega_1$到$\Omega_2$上的映射，若对任意$B\in\mathcal{F}_2$，$f^{-1}(B)\in\mathcal{F}_1$成立，则称$f$为$(\Omega_1,\mathcal{F}_1)$到$(\Omega_2,\mathcal{F}_2)$上的可测映射，记为$f\in\mathcal{F}_1/\mathcal{F}_2$.
  + 定理：设$(\Omega_1,\mathcal{F}_1)$和$(\Omega_2,\mathcal{F}_2)$是两个可测空间，其中$\mathcal{F}_2=\sigma(\mathcal{C})$。若$f$为$\Omega_1$到$\Omega_2$上的映射，且对任意$B\in\mathcal{C}$，$f^{-1}(B)\in\mathcal{F}_1$成立，则$f\in\mathcal{F}_1/\mathcal{F}_2$.
  + 推论：设$f$为可测空间$(\Omega,\mathcal{F})$到Borel域$(\mathbb{R},\mathcal{B}(\mathbb{R}))$上的映射，则$f$为可测映射$\Longleftrightarrow\forall x\in\mathbb{R},\{\omega\in\Omega:f(\omega)\leq x\}\in\mathcal{F}$.
+ 可测映射的复合：若$f$是由$(\Omega_1,\mathcal{F}_1)$到$(\Omega_2,\mathcal{F}_2)$上的可测映射，$g$是由$(\Omega_2,\mathcal{F}_2)$到$(\Omega_3,\mathcal{F}_3)$上的可测映射，则$f$与$g$的复合映射$g\circ f$是由$(\Omega_1,\mathcal{F}_1)$到$(\Omega_3,\mathcal{F}_3)$上的可测映射.
+ 若$f$是可测空间$(\Omega,\mathcal{F})$到Borel域$(\mathbb{R},\mathcal{B}(\mathbb{R}))$上的可测映射，称
  $$
  \sigma(f)=f^{-1}(\mathcal{B}(\mathbb{R}))=\{f^{-1}(B):B\in\mathcal{B}(\mathbb{R})\}
  $$
  为由$f$诱导的$\sigma$-代数。$\sigma(f)$为使得$f$可测的最小$\sigma$-代数。
## 排列组合
+ 略，除了一个重复组合：
  + 本质：已知一个正整数$r$，求将其分为$n$个非负整数之和的不同方案数，即求$|\Omega|,\Omega=\{(x_1,\cdots,x_n)\mid\sum_{i=1}^nx_i=r,x_k\in\mathbb{N}\}$
  + 计算：可理解为将$r$个球和$n-1$个隔板进行组合，$n+r-1$个位置先选出$r$个位置放球，剩下位置放隔板，隔板间球的数量即为分配的非负整数。故总方案数为$\binom{n+r-1}{r}$。
# 随机变量
+ 定义：可测空间$(\Omega,\mathcal{F})$到$(\mathbb{R},\mathcal{B}(\mathbb{R}))$上的可测映射。
+ 等价定义：给定概率空间$(\Omega,\mathcal{F},P)$，$X$为定义在$\Omega$上的实值函数（将$\Omega$中元素映射到$\mathbb{R}$上），则$X$为随机变量$\Longleftrightarrow\forall x\in\mathbb{R},\{\omega\in\Omega:X(\omega)\leq x\}\in\mathcal{F}$
  + 当$X(\omega)$取值只有两个值时，称$X$为伯努利随机变量；取值只有一个值时称为常值随机变量。
+ 示性函数：设$A\in\mathcal{F}$，称
  $$
  1_A(\omega)=\left\{
              \begin{aligned}
              1,&\hspace{1em} \omega\in A \\
              0,&\hspace{1em} \omega\in \overline{A} 
              \end{aligned}
              \right.
  $$
  为$A$的示性函数。可知$1_A$为伯努利随机变量。
## 分布函数
+ 定义：设$X$为随机变量，对任意实数$x$，称函数$F(x)=P\circ X^{-1}((-\infty,x])=P(X\leq x)$为随机变量$X$的（累积）分布函数（记为d.f.）。
+ 常值随机变量的分布函数
  $$
  F(x)=\left\{
              \begin{aligned}
              0,&\hspace{1em} x<c \\
              1,&\hspace{1em} x\geq c 
              \end{aligned}
              \right.
      =1_{[c,\infty)}x
  $$
  也被称为退化分布函数。
+ 不同的随机变量可能有相同的分布函数。
+ 性质：
  1. 单调性：$x<y\Longleftrightarrow F(x)\leq F(y)$；
  2. 有界性：$0\leq F(x)\leq 1$，$F(+\infty)=1,F(-\infty)=0$；
  3. 右连续性：$\forall x\in\mathbb{R},F(x+0)=F(x)$（$F(x+0)\triangleq\lim_{y\to x^+}F(y)$）
+ 常用公式：
  + $P(X>x)=1-F(x)$
  + $P(X<x)=F(x-0)\triangleq\lim_{y\to x^-}F(y)$
  + $P(X=x)=F(x)-F(x-0)$，$F(X\geq x)=1-F(x-0)$
  + $P(a<X\leq b)=F(b)-F(a)$
## 概率分布
### 离散型分布
+ 定义：若随机变量$X$可能取值为$x_1,x_2,\cdots,x_n,\cdots$，则称$P_k=P(X=x_k),k=1,2,\cdots$为$X$的分布列（记为p.f.），并称$X$为离散型随机变量（具有离散型分布）。
+ 分布列形式：  
  | $X$ |$x_1$|$x_2$|$\cdots$|$x_n$|$\cdots$|
  |:---:|:---:|:---:| :---:  |:---:|:---:|
  | $P$ |$p_1$|$p_2$|$\cdots$|$p_n$|$\cdots$|
+ 性质：非负性、正则性（略）
+ 定理1：若离散型随机变量$X$有分布列$p_k=P(X=x_k),k=1,2,\cdots$，则$X$的分布函数为
  $$
  F(X)=\sum_{k:\hspace{0.2em}x_k\leq x}p_k
  $$
+ 定理2：若离散型随机变量$X$有分布列$p_k=P(X=x_k),k=1,2,\cdots$，则
  $$
  P(X\in D)=\sum_{k:\hspace{0.2em}x_k\in D}p_k=\sum_kp_k\cdot 1_D(x_k)
  $$
+ 分布函数特征：阶梯型、间断点为$X$可能取值点、跳跃高度为该点概率值
  + 已知分布函数求分布列：若$F(x)$为$X$分布函数，则$X$可能取值点为$F(x)$的所有间断点$x_1,x_2,\cdots$，分布列为$P(X=x_k)=F(x_k)-F(x_k-0)$
### 连续型分布
+ 定义：设随机变量$X$分布函数为$F(x)$，若存在函数$p(x)$使得对任意$x$，满足$F(x)=\int_{-\infty}^xp(t)dt$成立，则称$X$为连续型随机变量（具有连续型分布），称$p(x)$为概率密度函数（记为p.d.f.）
+ 性质：
  + $F(x)$在$\mathbb{R}$上连续，故对$\forall a\in\mathbb{R},P(X=a)=F(a)-F(a-0)=0$；
  + 若随机变量$X$的概率密度函数为$p(x)$，则
    $$
    P(x\in(x-\frac{\Delta x}{2},x+\frac{\Delta x}{2}))=\int_{x-\frac{\Delta x}{2}}^{x+\frac{\Delta x}{2}}p(t)dt\approx p(x)\Delta x
    $$
    故$p(x)$在$x$处取值反映了$X$在$x$附近取值可能性大小（但不是概率值）。
  + 非负性、正则性（略）
  + 注：概率密度函数可以有间断点（如分段函数）
+ 定理1：若随机变量$X$有概率密度函数$p(x)$，$D\subset\mathbb{R}$，则
  $$
  P(X\in D)=\int_Dp(x)dx
  $$
+ 定理2：若概率密度函数$p(x)$为偶函数，则对任意实数$a$，分布函数$F$满足
  $$
  \begin{aligned}
  &F(-a)=\frac{1}{2}-\int_0^ap(x)dx;&F(a)+F(-a)=1\hspace{0.5em}(F(0)=\frac{1}{2});\\
  &P(|x|\leq a)=2F(a)-1;&P(|x|\geq a)=2(1-F(a))
  \end{aligned}
  $$
### *混合型分布
+ 定义：若$F_1(x)$为离散随机变量分布函数，$F_2(x)$为连续随机变量分布函数，则$\forall\alpha,0<\alpha<1$，$F(x)=\alpha F_1(x)+(1-\alpha)F_2(x)$为混合型分布函数。
### 常用离散分布
1. 二项分布
   + 分布列：$P(X=k)=\binom{n}{k}p^k(1-p)^{n-k},k=0,1,\cdots,n$
   + 记为$X\sim b(n,p)$
   + 当$n=1$时称为两点分布（或0-1分布）
2. **几何分布**
   + 分布列：$P(X=k)=p(1-p)^{k-1}$
   + 记为$X\sim Ge(p)$
   + 具有无记忆性：$P(X>m+n\mid X>m)=P(X>n)$
3. 负二项分布（帕斯卡分布）
   + 分布列：$P(X=k)=\binom{k-1}{r-1}p^r(1-p)^{k-r}$
   + 记为$X\sim Nb(r,p)$
   + 与几何分布关系：$Nb(1,p)=Ge(p)$
4. 泊松分布
   + 分布列：$P(X=k)=\frac{\lambda^k}{k!}e^{-\lambda}$
   + 记为$X\sim P(\lambda)$
   + 泊松定理：设$\lim_{n\to\infty}np_n=\lambda$，则对给定正整数$k$，有
      $$
      \lim_{n\to\infty}\binom{n}{k}p_n^k(1-p_n)^{n-k}=\frac{\lambda^k}{k!}e^{-\lambda}
      $$
      + 应用：当$n$很大，$p$很小时（$np\in[0.1,10]$）可以用二项分布列近似泊松分布列
5. 超几何分布
   + 分布列：$P(X=k)=\frac{\binom{M}{k}\binom{N-M}{n-k}}{\binom{N}{n}}$（$n\leq N,k\leq M,n-k\leq N-M$）
   + 记为$X\sim h(n,N,M)$
   + 当$n,k$固定，$N\to\infty$，$\frac{M}{N}\to p$时，超几何分布$h(n,N,M)$近似于二项分布$b(n,p)$
### 常用连续型分布
1. 正态分布
   + 概率密度函数：
    $$
    p(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\{-\frac{(x-\mu)^2}{2\sigma^2}\}
    $$
   + 记为$X\sim N(\mu,\sigma^2)$
   + 性质：
     + $p(x)$关于$x=\mu$对称，在$\mu$处取得最大值；
     + $\mu$控制$p(x)$对称轴位置，$\sigma$控制$p(x)$陡峭程度。
   + 标准正态分布：取$\mu=0,\sigma=1$，概率密度函数$p(x)=\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}$被记为$\varphi(x)$，其对应的分布函数$\int_{-\infty}^x\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}$被记为$\Phi(x)$。
     + 性质：$\Phi(0)=\frac{1}{2}$，$\Phi(x)+\Phi(-x)=1$
   + 正态分布标准化：当$X\sim N(\mu,\sigma^2),Y=\frac{X-\mu}{\sigma}$，则$Y\sim N(0,1)$
     + 推论：设$X\sim N(\mu,\sigma^2)$，则$X$分布函数$F(x)=\Phi(\frac{x-\mu}{\sigma})$.
   + 正态分布$3\sigma$准则（略）
2. 均匀分布
   + 概率密度函数：
      $$
      p(x)=\left\{
              \begin{aligned}
              \frac{1}{b-a},&\hspace{1em} a<x<b \\
              0,&\hspace{1em} \operatorname*{otherwise}.
              \end{aligned}
              \right.
      $$
   + 记为$X\sim U(a,b)$
   + 分布函数：
      $$
      F(x)=\left\{
              \begin{aligned}
              0,&\hspace{1em} x<a \\
              \frac{x-a}{b-a},&\hspace{1em} a\leq x<b\\
              1,&\hspace{1em} x\geq b
              \end{aligned}
              \right.
      $$
      $$ 
3. Gamma分布
   + 概率密度函数：
    $$
    p(x)=\left\{
              \begin{aligned}
              \frac{\lambda^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\lambda x},&\hspace{1em} x>0 \\
              0,&\hspace{1em} x\leq 0.
              \end{aligned}
              \right.
    $$
    其中
    $$
    \Gamma(\alpha)=\int_0^\infty x^{\alpha-1}e^{-x}dx
    $$
    （$\Gamma(1)=1,\Gamma(\frac{1}{2})=\sqrt{\pi},\Gamma(s+1)=s\Gamma(s)$）
   + 记为$X\sim Ga(\alpha,\lambda)$
   + 当$\alpha=1$时，为**指数分布**$X\sim Exp(\lambda)$
     + 概率密度函数：
      $$
      p(x)=\left\{
              \begin{aligned}
              \lambda e^{-\lambda x},&\hspace{1em} x>0 \\
              0,&\hspace{1em} x\leq 0.
              \end{aligned}
              \right.
      $$
     + 分布函数：
      $$
      F(x)=\left\{
              \begin{aligned}
              1-e^{-\lambda x},&\hspace{1em} x>0 \\
              0,&\hspace{1em} x\leq 0.
              \end{aligned}
              \right.
      $$
     + 具有无记忆性 
   + 当$\alpha=\frac{n}{2},\lambda=\frac{1}{2}$时，为卡方分布$X\sim \chi^2(n)$
## 数学期望和方差
### 数学期望
+ 定义：
  + 设离散型随机变量$X$有分布列$P(X=x_k)=p_k,k=1,2,\cdots$，若
    $$
    \sum_{k=1}^\infty x_kp_k
    $$
    绝对收敛，则称其为$X$的数学期望，记为$EX$；
  + 设连续型随机变量$X$有概率密度函数$p(x)$，若
    $$
    \int_{-\infty}^\infty xp(x)dx
    $$
    绝对收敛，则称其为$X$的数学期望，也记为$EX$；

+ 随机变量函数的期望：当$Y=f(X)$为随机变量$X$的函数，若$E(f(X))$存在，则
  $$
  EY=E(f(X))=\left\{
              \begin{aligned}
              &\sum_{k} f(x_k)p_k,&\hspace{1em} \text{离散情形} \\
              &\int_{-\infty}^\infty f(x)p(x)dx,&\hspace{1em} \text{连续情形}
              \end{aligned}
              \right.
  $$
+ 性质：
  + $E(c)=c$（$c$为常数）
  + $E(aX)=aEX$（$a$为常数）
  + $E(f(X)+g(X))=E(f(X))+E(g(X))$
### 方差
+ 定义：若$E(X-EX)^2$存在，则称其为随机变量$X$的方差，记为$\mathrm{Var}X$
  + 另称$\sigma_x=\sigma(X)=\sqrt{\mathrm{Var}X}$为$X$的标准差
+ 具体计算：
  $$
  \mathrm{Var}X=\left\{
              \begin{aligned}
              &\sum_{k} (x_k-EX)^2p_k,&\hspace{1em} \text{离散情形} \\
              &\int_{-\infty}^\infty (x-EX)^2p(x)dx,&\hspace{1em} \text{连续情形}
              \end{aligned}
              \right.
  $$
+ 性质：
  + $\mathrm{Var}(c)=0$
  + $\mathrm{Var}(aX+b)=a^2\mathrm{Var}X$
  + $\mathrm{Var}X=EX^2-(EX)^2$
  + $\mathrm{Var}X\geq 0$
+ 常见分布期望&方差汇总：
  |分布|形式|期望$EX$|方差$\mathrm{Var}X$|
  |:---:|:---:|:---:|:---:|
  |二项分布|$b(n,p)$|$np$|$np(1-p)$|
  |泊松分布|$P(\lambda)$|$\lambda$|$\lambda$|
  |几何分布|$Ge(p)$|$\frac{1}{p}$|$\frac{1-p}{p^2}$|
  |负二项分布|$Nb(r,p)$|$\frac{r}{p}$|$\frac{r(1-p)}{p^2}$|
  |超几何分布|$h(n,N,M)$|$n\cdot\frac{M}{N}$|$\frac{N-n}{N-1}\cdot n\cdot\frac{M}{N}\cdot\frac{N-M}{N}$|
  |均匀分布|$U(a,b)$|$\frac{a+b}{2}$|$\frac{(b-a)^2}{12}$|
  |正态分布|$N(\mu,\sigma^2)$|$\mu$|$\sigma^2$|
  |指数分布|$Exp(\lambda)$|$\frac{1}{\lambda}$|$\frac{1}{\lambda^2}$|
  |Gamma分布|$Ga(\alpha,\lambda)$|$\frac{\alpha}{\lambda}$|$\frac{\alpha}{\lambda^2}$|
### 马尔科夫不等式
+ 设随机变量$X\geq 0$且$EX$存在，则$\forall\epsilon>0$，有
  $$
  P(X\geq\epsilon)\leq\frac{EX}{\epsilon}
  $$
+ 推论：若$F(X)$在$X\geq 0$时单调递增，则
  $$
  P(X\geq\epsilon)\leq\frac{E(F(x))}{F(\epsilon)}
  $$
  （当然这里$F(X)$条件可以放宽，只要保证$X\geq\epsilon$时$F(X)\geq F(\epsilon)$即可）
### 切比雪夫不等式
+ 若随机变量$X$的方差存在，则$\forall\epsilon>0$，有
  $$
  \begin{aligned}
  &P(|X-EX|\geq\epsilon)\leq\frac{\mathrm{Var}X}{\epsilon^2}\\
  &P(|X-EX|<\epsilon)\geq 1-\frac{\mathrm{Var}X}{\epsilon^2}
  \end{aligned}
  $$
  + 推论：设$X$的方差存在，则$\mathrm{Var}X=0\Longleftrightarrow$存在常数$a$，满足$P(X=a)=1$
## 矩与其他数字特征
### 中心矩与原点矩
+ 设$k$为正整数，若$X^k$数学期望存在，则称$E(X-EX)^k$为$X$的$k$阶**中心矩**（记为$\nu_k$），称$EX^k$为$X$的$k$阶**原点矩**（记为$\mu_k$）
+ 性质：
  + 若高阶矩存在，则低阶矩必存在
  + $\mu_1=EX,\nu_2=\mathrm{Var}X$
  + $\nu_k=\sum_{i=0}^k\binom{k}{i}\mu_i(-1)^{k-i}\mu_1^{k-i}$（二项展开式）
+ 常用分布矩公式：
  + 正态分布$N(\mu,\sigma^2)$：
    $$
    \begin{aligned}
    &\nu_k=\left\{
      \begin{aligned}
      &(k-1)!!\sigma^{k}, &k=2m \\
      &0, &k=2m-1
      \end{aligned}
      \right.\\
    &\mu_k=E[\mu+(X-\mu)]^k=\sum_{j=0}^{\lfloor k/2\rfloor}\binom{k}{2j}(2j-1)!!\sigma^{2j}\mu^{k-2j}
    \end{aligned}
    $$
  + Gamma分布$Ga(\alpha,\lambda)$：
    $$
    \mu_k=\frac{\Gamma(\alpha+k)}{\lambda^k\Gamma(\alpha)}=\frac{(\alpha+k-1)(\alpha+k-2)\cdots(\alpha+1)\alpha}{\lambda^k}
    $$
    （$\nu_k$根据二项展开式计算）
### 分位数
+ 设$F(x)$为随机变量$X$的分布函数，$0<\alpha<1$，称
    $$
    X_\alpha=\operatorname*{inf}\{x:F(x)\geq a\}
    $$
  为$X$（或分布$F$）的$\alpha$-分位数。特别地，当$F$严格单调时，$X_\alpha$为$F(x)=\alpha$的解。
  + 标准正态分布的$\alpha$-分位数一般用$u_\alpha$表示。
+ 中位数：根据分位数的定义，称$X_{\frac{1}{2}}$为$X$（或分布$F$）的中位数。
  + 中位数满足$P(X\geq X_{\frac{1}{2}})=P(X\leq X_{\frac{1}{2}})$.
### 变异系数
+ 公式：
  $$
  C_v=\frac{\sqrt{\nu_2}}{\mu_1}=\frac{\sqrt{\mathrm{Var}X}}{EX}\hspace{1em} (EX\neq 0)
  $$
+ 消除量纲的影响，用于比较不同量纲的两个随机变量的波动大小
### 偏度系数与峰度系数
+ 偏度系数公式：
  $$
  \beta_s=\frac{\nu_3}{\sigma^3}=\frac{E(X-EX)^3}{(\sqrt{\mathrm{Var}X})^3}
  $$
  + 用于衡量随机变量的分布对称性：若$\beta_s>0$则右偏（中位数小于均值），$\beta_s<0$则左偏（中位数大于均值）
  ![skew](https://cdn.analyticsvidhya.com/wp-content/uploads/2024/09/sk1.webp)
+ 峰度系数公式：
  $$
  \beta_k=\frac{\nu_4}{\nu_2^2}-3=\frac{E(X-EX)^4}{(\mathrm{Var}X)^2}-3
  $$
  + 其中$3$为标准正态分布$N(0,1)$的$4$阶原点（中心）距$\mu_4(\nu_4)$
  + 用于刻画分布尾部的肥瘦程度（与标准正态分布相比）
+ 计算Gamma分布$Ga(\alpha,\lambda)$的偏度系数与峰度系数：   
  因为
  $$
  \mathrm{Var}X=\nu_2=\frac{\alpha}{\lambda^2},\hspace{1em}\nu_3=\frac{2\alpha}{\lambda^3},\hspace{1em}\nu_4=\frac{3\alpha(\alpha+2)}{\lambda^4}
  $$
  故
  $$
  \beta_s=\frac{2}{\sqrt{\alpha}},\hspace{1em}\beta_k=\frac{6}{\alpha}
  $$
# 随机向量（多元随机变量）
+ 定义：
  + 二维随机变量：设$X,Y$为定义在$(\Omega,\mathcal{F},P)$上随机变量，则称$(X,Y)$为二维随机变量（向量）；
  + 类似可定义$n$维随机变量。
## 二维随机变量
+ 联合分布函数：对任意$x,y$，称$F(x,y)=P(X\leq x,Y\leq y)$为$(X,Y)$的联合分布函数
  + 性质：单调性、有界性（$F(-\infty,y)=F(x,-\infty)=F(\infty,-\infty)=0,F(\infty,\infty)=1$）、右连续性、非负性（$\forall a_1<b_1,a_2<b_2,F(b_1,b_2)-F(b_1,a_2)-F(a_1,b_2)+F(a_1,a_2)\geq 0$）
+ 二维离散随机变量
  + $(X,Y)$取值有限对或可列对
  + 称$p_{ij}=P(X=x_i,Y=y_j),i,j=1,2,\cdots$为$(X,Y)$的联合分布列
  + 形式：
    |$X\backslash Y$|$y_1$|$y_2$|$\cdots$|$y_j$|$\cdots$|
    |:-------------:|:---:|:---:|:------:|:---:|:------:|
    |$x_1$|$p_{11}$|$p_{12}$|$\cdots$|$p_{1j}$|$\cdots$|
    |$x_2$|$p_{21}$|$p_{22}$|$\cdots$|$p_{2j}$|$\cdots$|
    |$\cdots$|$\cdots$|$\cdots$|$\cdots$|$\cdots$|$\cdots$|
    |$x_i$|$p_{i1}$|$p_{i2}$|$\cdots$|$p_{ij}$|$\cdots$|
    |$\cdots$|$\cdots$|$\cdots$|$\cdots$|$\cdots$|$\cdots$|
  + 定理：
    $$
    P((X,Y)\in D)=\sum_{i,j:\hspace{0.3em} x_i,y_j\in D}p_{ij}
    $$
+ 二维离散随机变量
  + 设二维随机变量$(X,Y)$联合分布函数为$F(x,y)$，则存在函数$p(x,y)$使得
    $$
    F(x,y)=\int_{-\infty}^x\int_{-\infty}^yp(u,v)dudv
    $$
    则称$p(x,y)$为其联合概率密度函数
  + 定理：设$(X,Y)$概率密度函数为$p(x,y)$，$D\in\mathcal{B}(\mathbb{R}^2)$，则
    $$
    P((X,Y)\in D)=\iint_D p(x,y)dxdy
    $$
    + 若$p(x,y)$可分解为$f(x)\cdot g(y)$，则积分可转化为
      $$
      P((X,Y)\in D)=\int_{D_x}f(x)dx\int_{D_y}g(y)dy
      $$
      其中$D_x$，$D_y$为$D$在$x,y$轴上的投影。
## 常用多维分布
1. 多项分布
   + 每次实验$r$种结果$A_1,\cdots,A_r$，设$P(A_i)=p_i,i=1,2,\cdots,r$， 记$X_i$为$n$次独立重复试验中$A_i$出现次数
   + $(X_1,\cdots,X_r)$联合分布列：
   $$
   P(X_1=n_1,\cdots,X_r=n_r)=\left\{
   \begin{aligned}
   &\frac{n!p_1^{n_1}\cdots p_r^{n_r}}{n_1!\cdots n_r!},& \sum_{i=1}^rn_i=n\\
   &0,& \mathrm{otherwise}.
   \end{aligned}
   \right.
   $$
2. 多维超几何分布
   + $N$个球，分为$r$个类，第$i$种有$N_i$个，$\sum_{i=1}^rN_i=N$
   + $(X_1,\cdots,X_r)$联合分布列：
   $$
   P(X_1=n_1,\cdots,X_r=n_r)=\left\{
   \begin{aligned}
   &\binom{N_1}{n_1}\cdots\binom{N_r}{n_r},& \sum_{i=1}^rn_i=n\\
   &0,& \mathrm{otherwise}.
   \end{aligned}
   \right.
   $$
3. 二维均匀分布
   + 记为：$(X,Y)\sim U(D)$
   + 联合概率密度函数：
      $$
      p(x,y)=\left\{
      \begin{aligned}
      &\frac{1}{S_D},& (x,y)\in D\\
      &0,& \mathrm{otherwise}.
      \end{aligned}
      \right.
      $$
      其中$S_D$为$D$的面积。
4. 二维正态分布
   + 记为：$(X,Y)\sim N(\mu_1,\sigma_1^2;\mu_2,\sigma_2^2;\rho)$
   + 联合概率密度函数：
      $$
      p(x,y)=\frac{1}{2\pi\sigma_1\sigma_2 c}\mathrm{exp}\{-\frac{1}{2c^2}(a^2+b^2-2\rho ab)\}
      $$
      其中
      $$
      a=\frac{x-\mu_1}{\sigma_1},b=\frac{y-\mu_2}{\sigma_2},c=\sqrt{1-\rho^2}
      $$
5. 高维正态分布
   + 记为：设$d$为随机向量$\mathbf{X}=(X_1,\cdots,X_d)^T$，向量$\mathbf{x}=(x_1,\cdots,x_d)^T$
   + 联合概率密度函数：
      $$
      p(\mathbf{x})=\frac{1}{(2\pi)^{\frac{d}{2}}|\Sigma|^\frac{1}{2}}\mathrm{exp}\{-\frac{1}{2}(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)\}
      $$
      其中
      $$
      \Sigma=(\operatorname*{Cov}(x_i,x_j))_{d\times d}
      $$
      为协方差矩阵，$\mu=(\mu_1,\cdots,\mu_d)^T$为$\mathbf{X}$的期望向量
    + 记为$\mathbf{X}\sim N(\mu,\Sigma)$
## 边际分布
+ 边际分布函数
  + 设随机变量$(X,Y)$联合分布函数为$F(X,Y)$，则随机变量$X$和$Y$的分布函数分别为：
  $$
  \begin{aligned}
  F_X(x)=F(x,\infty)\triangleq\lim_{y\to\infty}F(x,y)\\
  F_Y(y)=F(\infty,y)\triangleq\lim_{x\to\infty}F(x,y)
  \end{aligned}
  $$
+ 边际分布列
  + 设随机变量$(X,Y)$有分布列$p_{ij}=P(X=x_i,Y=y_j),i,j=1,2,\cdots$，则随机变量$X$和$Y$的边际分布列为
  $$
  \begin{aligned}
  P(X=x_i)=\sum_{j=1}^\infty p_{ij}\triangleq p_i,i=1,2,\cdots\\
  P(Y=y_j)=\sum_{i=1}^\infty p_{ij}\triangleq p_j,j=1,2,\cdots
  \end{aligned}
  $$
+ 边际概率密度函数
  + 已知$(X,Y)$联合概率密度函数为$p(x,y)$，则随机变量$X$和$Y$的边际概率密度函数分别为
  $$
  \begin{aligned}
  p_X(x)=\int_{-\infty}^\infty p(x,y)dy\\
  p_Y(y)=\int_{-\infty}^\infty p(x,y)dx
  \end{aligned}
  $$
+ 一般无法由边际分布推导出联合分布；二维均匀分布的边际分布不一定是均匀分布。
+ 二维正态分布的边际分布：
  + 设$(X,Y)$服从二维正态分布$N(\mu_1,\sigma_1^2;\mu_2,\sigma_2^2;\rho)$，则$X$服从正态分布$N(\mu_1,\sigma_1^2)$，$Y$服从正态分布$N(\mu_2,\sigma_2^2)$
## 随机向量的独立性
+ 二维定义：若对任意$(x,y)\in\mathbb{R}^2$，随机事件$\{X\leq x\}$与$\{Y\leq y\}$相互独立，则称随机变量$X$与$Y$相互独立。即：
  $$
  F(x,y)=F_X(x)F_Y(y)
  $$
+ 等价描述：
  + 若$(X,Y)$为离散随机向量，则$X$与$Y$独立$\Longleftrightarrow\forall (i,j),p_{ij}=p_ip_j$；
  + + 若$(X,Y)$为连续随机向量，则$X$与$Y$独立$\Longleftrightarrow\forall (x,y)\in\mathbb{R}^2,p(x,y)=p_X(x)p_Y(y)$；
+ 二维正态分布的独立性：
  + 设随机变量$(X,Y)$服从二维正态分布$N(\mu_1,\sigma_1^2;\mu_2,\sigma_2^2;\rho)$，则$X$与$Y$相互独立$\Longleftrightarrow\rho=0$
+ 一般定义：$X_1,\cdots,X_d$相互独立$\Longleftrightarrow F(x_1,\cdots,x_d)=F_1(x_1)\cdots F_d(x_d),(x_1,\cdots,x_d)\in\mathbb{R}^d$
### 随机变量函数的独立性
+ 定理：设$X,Y$为定义在概率空间$(\Omega,\mathcal{F},P)$上的随机变量，$f,g$均为$(\mathbb{R},\mathcal{B}(\mathbb{R}))$上的可测函数，若$X$与$Y$相互独立，则$f(X)$与$g(Y)$相互独立。
+ 推论：若$(X_1,\cdots,X_n)$相互独立，则$(X_{i1},\cdots,X_{ir})$与$(X_{j1},\cdots,X_{jr})$相互独立（$i1,\cdots,ir$与$j1,\cdots,jr$为互不相同的下标，均属于$\{1,2,\cdots,n\}$）
## 随机向量函数的分布
### 一元情形
+ 定理：已知随机变量$X$有分布列$\{p_k,k=1,2,\cdots\}$或概率密度函数$p_X(x)$，则$Y=f(X)$（$f$为Borel可测函数）的分布函数为
  $$
  F_Y(y)=\left\{
   \begin{aligned}
   &\sum_{k:\hspace{0.3em} x_k\in D_y}p_k,& \text{离散情形}\\
   &\int_{D_y}p_X(x)dx,& \text{连续情形}.
   \end{aligned}
   \right.
  $$
  其中$D_y=\{x:f(x)\leq y\}$
+ 例子：
  + 若$X\sim N(0,1)$，则$Y=X^2\sim Ga(\frac{1}{2},\frac{1}{2})=\chi^2(1)$；
  + 若$X\sim N(0,1)$，$Y=e^X$，则
    $$
    p_Y(y)=\left\{
    \begin{aligned}
    &\frac{1}{\sqrt{2\pi}y\sigma}e^{-\frac{(\ln y-\mu)^2}{2\sigma^2}},& y>0\\
    &0,& y\leq 0.
    \end{aligned}
    \right.
    $$
+ 定理1：若随机向量$X$的分布函数$F(x)$严格单调增，则$Y=F(X)\sim U(0,1)$
  + 证明：
  $$
  \begin{aligned}
  F_Y(y)=P(Y\leq y)&=P(F(X)\leq y)\\
                          &=\left\{
                          \begin{aligned}
                          &0, &y<0\\
                          &P(X\leq F^{-1}(y))=F(F^{-1}(y))=y, &0\leq y<1\\
                          &1, &y\geq 1
                          \end{aligned}
                          \right.
  \end{aligned}
  $$
+ 定理2：设随机变量$X$的概率密度函数为$p_X(x)$，函数$y=f(x)$严格单调，且其反函数$x=h(y)$连续可导，则随机变量$Y=f(X)$的概率密度函数为：
  $$
  p_Y(y)=\left\{
    \begin{aligned}
    &p_X(h(y))\cdot|h'(y)|,& y\in E\\
    &0,& \mathrm{otherwise}.
    \end{aligned}
    \right.
  $$
  其中$E$为$f$的值域。
+ 定理3（正态分布的线性不变性）：设$X\sim N(\mu,\sigma^2)$，则$a\neq 0$时，$Y=aX+b\sim N(a\mu+b,a^2\sigma^2)$.
  + 推论：当$X\sim N(\mu,\sigma^2)$时，$\frac{X-\mu}{\sigma}\sim N(0,1)$
+ 定理4（Gamma分布线性性）：设$X\sim Ga(\alpha,\lambda),c>0$，则$cX\sim Ga(\alpha,\frac{\lambda}{c})$.
  + 推论：设$X\sim N(0,\sigma^2)$，则$X^2\sim Ga(\frac{1}{2},\frac{1}{2\sigma^2})$.
### 多个随机变量函数的分布（二元情形）
+ 定理：设$(X,Y)$有分布列$\{p_{ij},i,j,=1,2,\cdots\}$或联合概率密度函数$p(x,y)$，则$Z=g(X,Y)$（$g$为$\mathbb{R}^2$到$\mathbb{R}$上可测函数）的分布函数为
  $$
  F_Z(z)=P((X,Y)\in D_z)=\left\{
   \begin{aligned}
   &\sum_{i,j:\hspace{0.3em} (x_i,y_j)\in D_z}p_{ij},& \text{离散情形}\\
   &\iint_{D_z}p(x,y)dxdy,& \text{连续情形}.
   \end{aligned}
   \right.
  $$
  + 特别地，当$g(x,y)=x$或$y$时，即为求边际分布函数。
  + 推论：当$Z=g(X,Y)$所有取值为$\{z_k,k=1,2,\cdots\}$,则$Z$的分布函数为
  $$
  \sum_{i,j:\hspace{0.3em}g(x_i,y_j)=z_k}p_{ij}
  $$
### 卷积公式
+ 离散情形：设$(X,Y)$分布列为$p_{ij}=P(X=x_i,Y=y_j)$，则$Z=X+Y$的分布列为
  $$
  \begin{aligned}
  P(Z=z_k)=\sum_i P(X=x_i,Y=z_k-x_i)\\
  =\sum_j P(X=z_k-y_j,Y=y_j)
  \end{aligned}
  $$
  特别地，当$X$与$Y$相互独立，则
  $$
  \begin{aligned}
  P(Z=z_k)=\sum_i P(X=x_i)P(Y=z_k-x_i)\\
  =\sum_j P(X=z_k-y_j)P(Y=y_j)
  \end{aligned}
  $$
+ 连续情形：设$(X,Y)$联合概率密度函数为$p(x,y)$，则$\forall t\in\mathbb{R}$，$Z=tX+Y$的概率密度函数为
  $$
  p_Z(z)=\int_{-\infty}^{\infty}p(x,z-tx)dx
  $$
  + 特别地，当$t=1$时，$Z=X+Y$的概率密度函数为
  $$
  p_Z(z)=\int_{-\infty}^{\infty}p(x,z-x)dx=\int_{-\infty}^{\infty}p(z-y,y)dy
  $$
  + 当然，当$X$与$Y$相互独立，则$p(x,z-tx)=p_X(x)p_Y(z-tx)$.
+ 另外，当$p_X(x)$与$p_Y(y)$为分段函数时，计算$D_z$就需要根据分段函数划定范围，再积分
### 概率分布的可加性
1. 二项分布的可加性
   + 若$X\sim b(n,p),Y\sim n(n,p)$，且相互独立，则$Z=X+Y\sim b(n+m,p)$
   + 即二项分布可视为多个相互独立的两点分布的和
2. 泊松分布的可加性
   + 若$X\sim P(\lambda_1),Y\sim P(\lambda_2)$，且相互独立，则$Z=X+Y\sim P(\lambda_1+\lambda_2)$
3. Gamma分布的可加性
   + 若$X\sim Ga(\alpha_1,\lambda),Y\sim Ga(\lambda_2,\lambda)$，且相互独立，则$Z=X+Y\sim Ga(\alpha_1+\alpha_2,\lambda)$
   + 推论：卡方分布的可加性
     + 若$X\sim\chi^2(n),Y\sim\chi^2(m)$，且相互独立，则$Z=X+Y\sim \chi^2(n+m)$
   + 注：指数分布不具有可加性
4. 正态分布的可加性
   + 设$(X,Y)$服从二维正态分布$N(\mu_1,\sigma_1;\mu_2,\sigma_2^2;\rho)$，则$Z=X+Y\sim N(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2+2\rho\sigma_1\sigma_2)$.
   + 推论：若$X\sim N(\mu_1,\sigma_1^2),Y\sim N(\mu_2,\sigma_2^2)$，且相互独立，则$X\pm Y\sim N(\mu_1\pm\mu_2,\sigma_1^2+\sigma_2^2)$
   + 正态分布的线性性：若$X_i\sim N(\mu_i,\sigma_i^2),i=1,2,\cdots,n$且相互独立，$\alpha_i$为任意常数，则
   $$
   \alpha_1X_1+\alpha_2X_2+\cdots+\alpha_nX_n\sim N(\alpha_1\mu_1+\cdots+\alpha_nX_N,\alpha_1^2\sigma_1^2+\cdots+\alpha_n^2\sigma_n^2)
   $$
### 极值分布
+ 设$(X_1,\cdots,X_n)$相互独立，分布函数为$F_i(x),i=1,\cdots,n$，记$Y=\max(X_1,\cdots,X_n),Z=\min(X_1,\cdots,X_n)$，则$Y$与$Z$的分布函数分别为
  $$
  \begin{aligned}
  &F_Y(y)=\prod_{i=1}^n F_i(y),\\
  &F_Z(z)=1-\prod_{i=1}^n(1-F_i(z))
  \end{aligned}
  $$
  + 特别地，当$X_1,\cdots,X_n$相互独立且同分布，分布函数均为$F_X(x)$，则$F_Y(y)=[F_X(y)]^n,F_Z(z)=1=[1-F_Z(z)]^n$.
  + 更进一步，当上述$X_1,\cdots,X_n$均为连续型随机变量，概率密度函数均为$p_X(x)$，则$p_Y(y)=n[F_X(y)]^{n-1}p_X(y),p_Z(z)=n[1-F_X(z)]^{n-1}p_X(z)$
### 混合型随机变量函数的分布
+ 计算分布函数：使用定义（$F(x,y)=P(X\leq x,Y\leq y),F_T(X+Y)=P(X+Y\leq t),\cdots$）
## 随机向量变换的分布
+ 注：本节所述随机变量均为连续型随机变量
+ 问题：设随机变量$\mathbf{X}=(X_1,\cdots,X_n)$的联合概率密度函数为$p(x_1,\cdots,x_n)$，且有$Y_i=h_i(\mathbf{X}),i=1,\cdots,m$（$h_i$为$\mathbb{R}^n$到$\mathbb{R}$上可测函数），试求$\mathbf{Y}=(Y_1,\cdots,Y_m)$的分布。
+ 使用定义——多重积分（麻烦，略去）
+ 使用坐标变换：若对任意$i=1,2,\cdots,n$，$y_i=h_i(x_1,\cdots,x_n)$的逆变换$x_i=k_i(y_1,\cdots,y_n)$存在且有连续偏导数，则$\mathbf{Y}=(Y_1,\cdots,Y_m)$的概率密度函数为
  $$
  p^*(y_1,\cdots,y_n)=\left\{
    \begin{aligned}
    &p(k_1(y_1,\cdots,y_n),\cdots,k_n(y_1,\cdots,y_n))|J|,&(y_1,\cdots,y_n)\in E,\\
    & 0, &\mathrm{otherwise}.
    \end{aligned}
    \right.
  $$
  其中$E$为$(h_1\mathbf{X},\cdots,h_n\mathbf{X})$的值域，$J$为坐标变换的雅格比行列式：
  $$
  J=\frac{\partial(x_1,\cdots,x_n)}{\partial(y_1,\cdots,y_n)}=\left|
    \begin{array}{cccc}
    \frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} & \cdots & \frac{\partial x_1}{\partial y_n}\\
    \frac{\partial x_2}{\partial y_1} & \frac{\partial x_2}{\partial y_2} & \cdots & \frac{\partial x_2}{\partial y_n}\\
    \cdots & \cdots & \cdots & \cdots\\
    \frac{\partial x_n}{\partial y_1} & \frac{\partial x_n}{\partial y_2} & \cdots & \frac{\partial x_n}{\partial y_n}
    \end{array}
    \right|
  $$
### 增补变量法求联合分布函数的分布
+ 问题：已知$(X,Y)$的联合分布，求$U=G(X,Y)$的分布。
+ 方法：增加一个随机变量$V=h(X,Y)$（通常取$V=X$或$V=Y$），再求$(U,V)$的联合分布，最后对$U$求边际分布。
+ 例：
  + 两个随机变量的积（取$V=Y$）：
    $$
    U=XY\Longrightarrow p_U(u)=\int_{-\infty}^\infty p_X(\frac{u}{v})p_Y(v)\frac{1}{|v|}dv
    $$
  + 两个随机变量的商（取$V=Y$）：
    $$
    U=\frac{X}{Y}\Longrightarrow p_U(u)=\int_{-\infty}^\infty p_X(uv)p_Y(v)|v|dv
    $$
## 随机变量的数字特征
### 数学期望与方差
+ 数学期望定义：设$(X,Y)$为二维随机向量，若$EX$与$EY$均存在（通过边际分布求得），则称$(EX,EY)$为$(X,Y)$的数学期望（向量）【也可推广为$d$维随机向量，下略】
+ 方差定义：类似一维随机变量，故省略
+ 定理：设随机变量$Z=g(X,Y)$为$(X,Y)$的函数，若$EZ$存在，则
  $$
  EZ=E[g(X,Y)]=\left\{
    \begin{aligned}
    &\sum_{i,j}g(x_i,y_j)p_{ij},&\text{离散情形}\\
    &\iint g(x,y)p(x,y)dxdy,&\text{连续情形}
    \end{aligned}
    \right.
  $$
+ 性质（注：下面将$X,Y$替换为$f(X),g(Y)$也成立）：
  + $E(X+Y)=EX+EY$
  + 若$X,Y$相互独立，则$E(XY)=EX\cdot EY$
  + $\mathrm{Var}(X\pm Y)=\mathrm{Var}X+\mathrm{Var}Y\pm 2E[(X-EX)(Y-EY)]$
  + $E[(X-EX)(Y-EY)]=E(XY)-EX\cdot EY$
  + 若$X$与$Y$相互独立，则：
    + $E[(X-EX)(Y-EY)]=0$
    + $\mathrm{Var}(X\pm Y)=\mathrm{Var}X+\mathrm{Var}Y$
+ 推论：若$n$维随机变量$\mathrm{X}=\sum_{i=1}^nX_i$，则
    $$
    \mathrm{Var}\mathbf{X}=\sum_{i=1}^n\mathrm{Var}(X_i)+2\sum_{i=1}^n\sum_{j=1}^{i-1}E[(X_i-EX_i)(Y_j-EY_j)]
    $$
### 协方差
+ 定义：设$(X,Y)$为二维随机变量，若$E[(X-EX)(Y-EY)]$存在，则称其为$X$与$Y$的协方差，记为$\operatorname*{Cov}(X,Y)$.
+ 性质：
  + $\operatorname*{Cov}(X,Y)=\operatorname*{Cov}(Y,X)$
  + $\operatorname*{Cov}(aX,bY)=ab\operatorname*{Cov}(X,Y)$
  + $\operatorname*{Cov}(X,Y)=E(XY)-EX\cdot EY$（计算时更常用）
  + 若$X$与$Y$独立，则$\operatorname*{Cov}(X,Y)=0$
  + $\operatorname*{Cov}(X,a)=0,\operatorname*{Cov}(X,X)=\operatorname*{Var}X$
  + $\operatorname*{Cov}(X+Y,Z)=\operatorname*{Cov}(X,Z)+\operatorname*{Cov}(Y,Z)$
  + $\operatorname*{Var}(aX\pm bY)=a^2\mathrm{Var}(X)+b^2\mathrm{Var}(Y)\pm 2ab\mathrm{Cov}(X,Y)$
+ 协方差阵
  + 设$(X_1,\cdots,X_n)$为$n$维随机向量，若$\forall i,j(1\leq i,j\leq n)$，$\mathrm{Cov}(X_i,X_j)$存在，则称$\Sigma=(\mathrm{Cov}(X_i,X_j))_{n\times n}$为$(X_1,\cdots,X_n)$的协方差矩阵
  + 性质：
    + 主对角线上元素为$X_i$的方差
    + 所有元素之和为$\mathbf{X}=X_1+\cdots+X_n$的方差（$\mathrm{Var}(\sum_{i=1}^n X_i)=\sum_{i,j=1}^n\mathrm{Cov}(X_i,X_j)$）
    + 若$(X,Y)\sim N(\mu_1,\sigma_1^2;\mu_2,\sigma_2^2;\rho)$，则 $$
      $$
      \Sigma=\left(
        \begin{array}{cc}
        \sigma_1^2 & \rho\sigma_1\sigma_2\\
        \rho\sigma_1\sigma_2 & \sigma_2^2
        \end{array}
        \right)
      $$
    + $\Sigma$有对称性与半正定性
### 相关系数
+ 定义：若$(X,Y)$为二维随机向量，则称
  $$
  \mathrm{Corr}(X,Y)=\frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathrm{Var}X}\sqrt{\mathrm{Var}Y}}
  $$
  为$X$与$Y$的相关系数（也记为$\rho_{xy}$）
  + 在实际计算中，一般采用下述公式：
    $$
    \mathrm{Corr}(X,Y)=\frac{E(XY)-EX\cdot EY}{\sqrt{EX^2-(EX)^2}\sqrt{EY^2-(EY)^2}}
    $$
    即计算$EX,EY,EX^2,EY^2,E(XY)$五个量。
+ 性质：
  + 若$X^*=\frac{X-EX}{\sqrt{\mathrm{Var}X}},Y^*=\frac{Y-EY}{\sqrt{\mathrm{Var}Y}}$，则$\mathrm{Corr}(X,Y)=\mathrm{Cov}(X^*,Y^*)=\mathrm{Corr}(X^*,Y^*)=E(X^*Y^*)$
  + $|\mathrm{Corr}(X,Y)|\leq 1$，且取等号当且仅当$X$与$Y$有线性关系（即$\exists a,b$满足$P(Y=aX+b)=1$）
+ 推论：
  + 由$|\mathrm{Corr}(X,Y)|\leq 1$可立即推得
    $$
    |E[(X-EX)(Y-EY)]|^2\leq \mathrm{Var}X\cdot\mathrm{Var}Y
    $$
    而令$EX=EY=0$即得柯西-施瓦茨不等式：
    $$
    |E(XY)|^2=EX^2\cdot EY^2
    $$
    当然这个不等式在$EX,EY\neq 0$时也成立，不过就需要其他证明方法。
+ 当$\mathrm{Corr}(X,Y)=\pm 1$时，称$X$与$Y$正/负相关（隐去了线性性）；当$\mathrm{Corr}(X,Y)=0$时，称$X$与$Y$不相关（但不代表没有关系！）
  + 若$X$与$Y$相互独立，则$\mathrm{Corr}(X,Y)=0$，即$X$与$Y$不相关，但反之不成立
+ 二维正态分布的相关系数：
  + 若$(X,Y)\sim N(\mu_1,\sigma_1^2;\mu_2,\sigma_2^2;\rho)$，则$\mathrm{Corr}(X,Y)=\rho$.
  + 如果$(X,Y)$满足二维正态分布，则$X$与$Y$不相关$\Longleftrightarrow X$与$Y$相互独立（但如果只有$X$与$Y$满足正态分布命题不成立）
## 条件分布
+ 定义：设$(X,Y)$为二维随机变量，且对任意$\Delta y>0$，$P(y-\Delta y< Y\leq y)>0$（若$Y$离散，则等价于$Y$可取到$y$；若$Y$连续，则等价于$p_Y(y)>0$）。若对任意实数$x$，
  $$
  \begin{aligned}
  &\lim_{\Delta y\to 0^+}P(X\leq x\mid y-\Delta y<Y\leq y)\\
  =&\lim_{\Delta y\to 0^+}\frac{P(X\leq x, y-\Delta y<Y\leq y)}{P(y-\Delta y<Y\leq y)}
  \end{aligned}
  $$
  存在，则称其为$Y=y$条件下$X$的条件分布函数，记为$P(X\leq x\mid Y=y)$或$F_{X\mid Y}(x\mid y)$。
### 条件分布列
+ 设$(X,Y)$的联合分布列为$P(X=x_i,Y=y_j)=P_{ij},i,j=1,2,\cdots$，若$P(Y=y_j)>0$，则$Y=y_j$条件下$X$的条件分布列为
  $$
  P(X=x_i\mid Y=y_j)=\frac{P(X=x_i,Y=y_j)}{P(Y=y_j)}=\frac{p_{ij}}{\sum_k p_{kj}},i=1,2,\cdots
  $$
### 条件概率密度函数
+ 设$(X,Y)$的联合概率密度函数为$p(x,y)$，$X$与$Y$的边际概率密度函数分别为$p_X(x)$与$p_Y(y)$，则$Y=y$（$p_Y(y)>0$）条件下$X$的条件分布函数为
  $$
  F_{X\mid Y}(x\mid y)=\int_{-\infty}^x\frac{p(u,y)}{p_Y(y)}du
  $$
  其中$\frac{p(x,y)}{p_Y(y)}$被称为$Y=y$条件下$X$的条件概率密度函数，记为$p_{X\mid Y}(x\mid y)$.
  + 公式推导：
    $$
    \begin{aligned}
    P(X\leq x\mid y-\Delta y<Y\leq y)&=\int_{-\infty}^x\int_{y-\Delta y}^y p(u,v)dudv\\
    &=\int_{y-\Delta y}^y(\int_{-\infty}^xp(u,v)du)dv
    \end{aligned}
    $$
    故
    $$
    \begin{aligned}
    &\lim_{\Delta y\to 0^+}\frac{1}{\Delta y}P(X\leq x\mid y-\Delta y<Y\leq y)\\
    =&\lim_{\Delta y\to 0^+}\frac{1}{\Delta y}\int_{y-\Delta y}^y(\int_{-\infty}^xp(u,v)du)dv\\
    =&\int_{-\infty}^x p(u,y)du
    \end{aligned}
    $$
    最后一个等号由微分中值定理得到。   
    同理，
    $$
    \begin{aligned}
    &\lim_{\Delta y\to 0^+}\frac{1}{\Delta y}P(y-\Delta y<Y\leq y)\\
    =&\lim_{\Delta y\to 0^+}\frac{1}{\Delta y}\int_{y-\Delta y}^y p_Y(v)dv\\
    =&p_Y(y)
    \end{aligned}
    $$
+ 注：在计算连续型随机变量条件概率（如$P(X>a\mid Y=y)$）时不能直接使用条件概率公式（因为$P(Y=y)=0$），而应该先求条件概率密度函数，再用积分求解。
### 条件数学期望
+ 定义：设$(X,Y)$为二维随机变量，称
  $$
  E(X\mid Y=y)=\left\{
    \begin{aligned}
    &\sum_i x_iP(X=x_i\mid Y=y),& \text{离散情形}\\
    &\int_{-\infty}^\infty xp_{X\mid Y}(x\mid y)dx,& \text{连续情形}
    \end{aligned}
    \right.
  $$
  为$Y=y$条件下$X$的条件数学期望。（在连续情形下，需要$p_{X\mid Y}(x\mid y)$有意义）
  + 另外，对于随机变量$X$与随机事件$B$，若$P(B)\neq 0$，可定义$X$关于$B$的条件数学期望：
  $$
  E(X\mid B)=\frac{E(X\cdot 1_B)}{P(B)}
  $$
  + 在实际运算中，一般会先求出条件分布，再使用期望公式得出。
+ 注：条件数学期望是关于$y$的函数，且可能不存在。
+ 设$E(X\mid Y=y)=g(y)$，则可以考虑随机变量$Y$的函数$g(Y)$，记为$E(X\mid Y)$（即$Y=y$时取值$E(X\mid Y=y)$）。
+ 重期望公式：
  + 设$(X,Y)$为二维随机变量，且$EX$存在，则$EX=E(E(X\mid Y))=E(g(Y))$.
  + 本质为全概率公式：
  $$
  EX=\left\{
    \begin{aligned}
    &\sum_j E(X\mid Y=y_j)P(Y=y_j),&\text{离散情形}\\
    &\int_{-\infty}^\infty E(X\mid Y=y)p_Y(y)dy,&\text{连续情形}
    \end{aligned}
    \right.
  $$
+ 条件数学期望的性质：
  + $E(aX+bY\mid Z)=aE(X\mid Z)+bE(Y\mid Z)$
  + $E(X\mid X)=X$
### 条件方差
+ 定义：称
  $$
  \begin{aligned}
  &\mathrm{Var}(X\mid Y=y)=E[(X-E(X\mid Y=y))^2\mid Y=y]\\
  =&E(X^2\mid Y=y)-(E(X\mid Y=y))^2
  \end{aligned}
  $$
  为$Y=y$时$X$的条件方差（也为$y$的函数）
+ 类似有$\mathrm{Var}(X\mid Y)$为$Y$的函数
+ 条件方差公式：$\mathrm{Var}X=E(\mathrm{Var}(X\mid Y))+\mathrm{Var}(E(X\mid Y))$
  + 简单证明（基于重期望公式）：
    $$
    \begin{aligned}
      &E(\mathrm{Var}(X\mid Y))+\mathrm{Var}(E(X\mid Y))\\
      &=E[E(X^2\mid Y)-(E(X\mid Y))^2]+E[(E(X\mid Y))^2]-[E[E(X\mid Y)]]^2\\
      &=E[E(X^2\mid Y)]-[E[E(X\mid Y)]]^2\\
      &=EX^2-(EX)^2=\mathrm{Var}X
    \end{aligned}
    $$
# 极限理论
## 随机变量序列的收敛性
1. 几乎处处收敛（依概率$1$收敛）
   + 定义：若随机变量序列$\{X_n,n\geq 1\}$满足
      $$
      P(\lim_{n\to\infty}X_n=X)=1
      $$
     则称$\{X_n,n\geq 1\}$几乎处处收敛到$X$（记为$X_n\stackrel{a.s.}{\longrightarrow}X$）
   + 样本点角度定义：
      $$
      P(\{\omega:\lim_{n\to\infty} X_n(\omega)=X(\omega)\})=1
      $$
      即满足$X_n(\omega)\nrightarrow X(\omega)\hspace{0.5em}(n\to\infty)$的$\omega$发生的概率为$0$
2. 依概率收敛
   + 定义：若$\forall\epsilon>0,$
      $$
      \lim_{n\to\infty}P(|X_n-X|\geq\epsilon)=0
      $$
      则称$\{X_n,n\geq 1\}$依概率收敛到$X$（记为$X_n\stackrel{P}{\longrightarrow}X$）
   + 与几乎处处收敛的区别：“几乎处处收敛”针对随机变量取值取极限，而“依概率收敛”针对随机变量偏离的概率求极限
3. 依分布收敛
   + 定义：设随机变量$X_n$的分布函数为$F_n(x)$，随机变量$X$的分布函数为$F(x)$，如果对于$F(x)$的任意连续点$x$，有
      $$
      \lim_{n\to\infty}F_n(x)=F(x)
      $$
      则称$\{X_n,n\geq 1\}$依分布收敛到$X$（记为$X_n\stackrel{d}{\longrightarrow}X$或$X_n\stackrel{L}{\longrightarrow}X$），也称$\{F_n,n\geq 1\}$弱收敛于$F$（记为$F_n\stackrel{w}{\longrightarrow}F$）
### 性质
1. 几乎处处收敛性质
   + 下面四个条件等价：
      $$
      \begin{gather}
      X_n\stackrel{a.s.}{\longrightarrow}X\\
      P(\bigcup_{k=1}^\infty\bigcap_{n=1}^\infty\bigcup_{m=n}^\infty\{|X_m-X|\geq\frac{1}{k}\})=0\\
      \forall\epsilon>0,P(\bigcap_{n=1}^\infty\bigcup_{m=n}^\infty\{|X_m-X|\geq\epsilon\})=0\\
      \forall\epsilon>0,\lim_{n\to\infty}P(\bigcup_{m=n}^\infty\{|X_m-X|\geq\epsilon\})=0
      \end{gather}
      $$
   + Borel-Cantelli引理：设$\{A_n,n\geq 1\}$为$(\Omega,\mathcal{F},P)$中的随机事件列，记
      $$
      \{A_n,i.o.\}=\bigcap_{n=1}^\infty\bigcup_{m=n}^\infty A_m
      $$
      表示为“事件$A_n$发生无穷多次”（infinity often），那么：
      + 若$\sum_{n=1}^\infty P(A_n)<\infty$，则$P(A_n,i.o.)=0$；
      + 若$\{A_n,n\geq 1\}$为相互独立的事件列，且$\sum_{n=1}^\infty p(A_n)=\infty$，则$P(A_n,i.o.)=1$
   + 那么由引理可得：
     + 若$\forall\epsilon>0,\sum_{n=1}^\infty P(|X_n-X|\geq\epsilon)<\infty$，则$X_n\stackrel{a.s.}{\longrightarrow}X$；
     + 若$\{X_n,n\geq 1\}$为相互独立的随机变量序列，$c$为常数，则
      $$
      X_n\stackrel{a.s.}{\longrightarrow}c\Longleftrightarrow \forall\epsilon>0,\sum_{n=1}^\infty P(|X_n-c|\geq\epsilon)<\infty
      $$
2. 依概率收敛性质
   + 若随机变量序列$\{X_n,n\geq 1\}$满足$\lim_{n\to\infty}EX_n^2=0$，则$X_n\stackrel{P}{\longrightarrow}0$（可使用马尔科夫不等式证明）
   + 计算：设$X_n\stackrel{P}{\longrightarrow}X,Y_n\stackrel{P}{\longrightarrow}Y$，则
      $$
      \begin{aligned}
      X_n+Y_n\stackrel{P}{\longrightarrow}X+Y\\
      X_nY_n\stackrel{P}{\longrightarrow}XY
      \end{aligned}
      $$
   + 判别：若$X_n\stackrel{P}{\longrightarrow}X$，则
     + 设$f$为定义在$\mathbb{R}$上的连续函数，则$f(X_n)\stackrel{P}{\longrightarrow}f(X)$；
     + $$
        lim_{n\to\infty}E(\frac{|X_n-X|}{1+|X_n-X|})=0
       $$
3. 依分布收敛性质
   + Slutsky定理：若$X_n\stackrel{d}{\longrightarrow}X,Y_n\stackrel{P}{\longrightarrow}c$（$c$为常数），则
     + $X_n+Y_n\stackrel{d}{\longrightarrow}X+c$；
     + $X_nY_n\stackrel{d}{\longrightarrow}cX$
     + $c\neq 0,Y_n\neq 0$时，$\frac{X_n}{Y_n}\stackrel{d}{\longrightarrow}\frac{X}{c}$
### 三种收敛之间的关系
1. $X_n\stackrel{a.s.}{\longrightarrow}X$蕴含$X_n\stackrel{P}{\longrightarrow}X$，反之不成立
2. $X_n\stackrel{P}{\longrightarrow}X\Longleftrightarrow$对每个子列$\{X_{n_k},k\geq 1\}$，存在子子列$\{X_{n_{k_j}},j\geq 1\}$满足$X_{n_{k_j}}\stackrel{a.s.}{\longrightarrow}X$
3. $X_n\stackrel{P}{\longrightarrow}X$蕴含$X_n\stackrel{d}{\longrightarrow}X$，反之不成立
4. 设$c$为一个常数，则$X_n\stackrel{d}{\longrightarrow}c$蕴含$X_n\stackrel{P}{\longrightarrow}c$
## 特征函数
+ 前提概念：
  + 复值随机变量：设$X,Y$为$(\Omega,\mathcal{F},P)$上的实值随机变量，则称$Z=X+iY$为复值随机变量（若$EX,EY$存在，则称$EZ=EX+iEY$为$Z$的数学期望）；
  + $Z$的共轭随机变量：$\overline{Z}=X-iY$；
  + $Z$的模：$|Z|=\sqrt{|X|^2+|Y|^2}$；
  + 独立性：设$Z_1=X_1+iY_1,Z_2=X_2+iY_2$，若$X_1,Y_1$分别与$X_2,Y_2$相互独立，则称$Z_1$与$Z_2$相互独立。
    + 另一种形式：若$X$与$Y$相互独立，则$e^{iX}$与$e^{iY}$相互独立
+ 定义：设$X$为随机变量，其分布函数为$F$，则称复值函数
  $$
  f(t)=Ee^{itX}=E[\cos(tX)+i\sin(tX)]=E[\cos(tX)]+iE[\sin(tX)]
  $$
  为$X$（或$F$）的特征函数。
  + 任何随机变量（或概率分布）的特征函数均存在（因为$|e^{itX}|=1<\infty$）
+ 常见分布特征函数：
  |分布|特征函数|
  |:--:|:-----:|
  |$b(1,p)$|$pe^{it}+1-p$|
  |$Ge(p)$|$\frac{pe^{it}}{1-(1-p)e^{it}}$|
  |$P(\lambda)$|$e^{\lambda(e^{it}-1)}$|
  |$U(0,1)$|$\left\{\begin{aligned}&\frac{e^{it-1}}{it},&t\neq 0\\&1,&t=0\end{aligned}\right.$|
  |$Exp(\lambda)$|$\frac{\lambda}{\lambda-it}$|
  |$N(0,1)$|$e^{-\frac{t^2}{2}}$|
### 特征函数的性质
1. $f(0)=1,|f(t)|\leq 1$
2. $f(-t)=\overline{f(t)}$（共轭）
3. 设$a,b$为常数，$Y=aX+b$，则$f_Y(t)=e^{ibt}f_X(at)$
4. 若$X$与$Y$相互独立，则$f_{X+Y}(t)=f_X(t)f_Y(t)$ !!比卷积公式好记!!
5. 特征函数与矩的关系：若随机变量$X$的$k$阶矩$\mu_k=EX^k$存在，则$\forall j,1\leq j\leq k,f(t)$的$j$阶导数存在，且
   $$
   f^{(j)}(0)=i^j\mu_j
   $$
+ 定理1：设$f(t)$为随机变量$X$的特征函数，则$f(t)$非负定，即对于任意正整数$n$，取任意$n$个复数$z_1,\cdots,z_n$与$n$个实数$t_1,\cdots,t_n$，有
  $$
  \sum_{i,j=1}^n f(t_i-t_j)z_i\overline{z_j}>0
  $$
+ 定理2（唯一性定理）：概率分布由其特征函数唯一确定。设$f$为分布函数$F$的特征函数，则
  $$
  F(x)=\lim_{y\to-\infty}\lim_{T\to\infty}\int_{-T}^T\frac{e^{-ity}-e^{-itx}}{i\cdot 2\pi t}f(t)dt
  $$
  （这一公式也被称为逆转公式）
+ 定理3（连续性定理）：
  + 设$X_n\stackrel{d}{\longrightarrow}X, f_n(n\geq 1)$为$X_n$的特征函数，$f$为$X$的特征函数，则$\lim_{n\to\infty}f_n(t)=f(t),t\in\mathbb{R}$；
  + 设$f$在$0$点连续，$\lim_{n\to\infty}f_n(t)=f(t)$对任意$t\in\mathbb{R}$成立，则$X_n\stackrel{d}{\longrightarrow}X$（$X_n,X$同上）
    + 注：上面条件中$f$在$0$点连续是为了保证$\lim_{n\to\infty}f_n(t)$为分布函数。
### 多维随机变量的特征函数
+ 定义：设$\mathbf{X}=(X_1,\cdots,X_d)^T$为$d$维随机变量，则称
  $$
  f(t_1,\cdots,t_d)=Ee^{i\mathbf{t}^T\mathbf{X}}=E\operatorname*{exp}\{i\sum_{i=1}^dt_iX_i\}
  $$
  为$\mathbf{X}$的特征函数，这里$\mathbf{t}=(t_1,\cdots,t_d)^T\in\mathbb{R}^d$
+ 例：设$(X,Y)$服从二维正态分布$N(0,1;0,1;\rho)$，则$(X,Y)$的特征函数为
  $$
  f(t_1,t_2)=E\operatorname*{exp}[i(t_1X+t_2Y)]=\operatorname*{exp}\{-\frac{t_1^2+t_2^2+2\rho t_1t_2}{2}\}
  $$
### 矩母函数（特征函数的实数版本）
+ 定义：设$X$为一随机变量，对于$t\in\mathbb{R}$使得$E[e^{tX}]<\infty$，则称
  $$
  M_X(t)=E[e^{tX}]
  $$
  为$X$的矩母函数。
+ 注：矩母函数不一定对所有$t\in\mathbb{R}$均存在（比特征函数条件更严格）
+ 若$\mathbf{X}=(X_1,\cdots,X_d)^T$为$d$维随机变量，则称
  $$
  M_X(\mathbf{t})=Ee^{\mathbf{t}^T\mathbf{X}}=E\operatorname*{exp}\{\sum_{i=1}^dt_iX_i\}
  $$
  为$\mathbf{X}$的特征函数，这里$\mathbf{t}=(t_1,\cdots,t_d)^T\in\mathbb{R}^d$
+ 性质：
  1. 若随机变量$X$与$Y$相互独立，则$M_{X+Y}(t)=M_X(t)M_Y(t)$（与特征函数相同）
  2. 与原点矩的关系（基于泰勒展开）：$M_X^{(n)}(t)=E[X^ne^{tX}]\Longrightarrow M_X^{(n)}(0)=E[X^n]=\mu_n$
  3. 与中心矩的关系：设$\mu=EX,M_{X-\mu}(t)=E[e^{t(X-\mu)}]=e^{-\mu}M_X(t)$，则
      $M_{X-\mu}^{(n)}(t)=E[(X-\mu)^ne^{t(X-\mu)}]\Longrightarrow M_{X-\mu}^{(n)}(0)=E[(X-\mu)^n]=\nu_n$
+ 常见分布矩母函数（基本就是特征函数把$it$换成了$t$）：
  |分布|矩母函数|
  |:--:|:-----:|
  |$b(n,p)$|$(pe^{t}+1-p)^n$|
  |$Ge(p)$|$\frac{pe^{it}}{1-(1-p)e^{it}}$|
  |$P(\lambda)$|$e^{\lambda(e^{t}-1)}$|
  |$Exp(\lambda)$|$\frac{\lambda}{\lambda-t},\forall t<\lambda$|
  |$N(0,1)$|$e^{\frac{t^2}{2}}$|
## 大数定律
1. **伯努利大数定律**
    + 设一次试验中事件$A$发生的概率为$p$，记$S_n$为$n$次独立实验中事件$A$的发生次数，则$n\to\infty$时，
      $$
      \frac{S_n}{n}\stackrel{P}{\longrightarrow}p
      $$
2. **（弱）大数定律**
    + 定义：对于随机变量序列$\{X_n,n\geq 1\}$，若存在数列$\{a_n,n\geq 1\}$与$\{b_n,n\geq 1\}$（$\{b_n\}$单调增且趋于无穷），且有$n\to\infty$时，
      $$
      \frac{S_n-a_n}{b_n}\stackrel{P}{\longrightarrow}0
      $$
      其中$S_n=\sum_{k=1}^nX_k$.则称$X_n$满足（弱）大数定律。
    + 注1：如无特别说明，一般取$a_n=ES_n,b_n=n$，即
      $$
      \frac{S_n-ES_n}{n}\stackrel{P}{\longrightarrow}0
      $$
    + 注2：若$\{X_n\}$独立同分布，上式也等价于
      $$
      \lim_{n\to\infty}P(|\frac{\sum_{k=1}^nX_k}{n}-EX_1|\geq\epsilon)=0
      $$
      （伯努利大数定律就是$X_k\sim b(1,p)$的情形）
2. 其他大数定律（均对于随机变量序列$\{X_n,n\geq 1\}$，并设$S_n=\sum_{k=1}^nX_k$，满足下述条件之一即满足（弱）大数定律）
    + **切比雪夫大数定律**：$\{X_n,n\geq 1\}$两两不相关，且$\{\mathrm{Var}X_n\}$一致有界；
    + **马尔科夫大数定律**：马尔科夫条件成立，即$\lim_{n\to\infty}\frac{\mathrm{Var}S_n}{n^2}=0$；
    + **辛钦大数定律**：$\{X_n,n\geq 1\}$独立同分布且$EX_1$存在。
+ 相互关系：
  + 伯努利大数定律是切比雪夫大数定律的特例；切比雪夫大数定律是马尔科夫大数定律的推论（前者成立，后者一定成立）；马尔科夫大数定律可以由切比雪夫不等式推得：
  $$
  P(|\frac{S_n-ES_n}{n}|\geq \epsilon)=P(|S_n-ES_n|\geq n\epsilon)\leq\frac{\mathrm{Var}S_n}{n^2\epsilon^2}\to 0(n\to\infty)
  $$
  + 辛钦大数定律可以用特征函数的连续性定理证明
### （弱）大数定律的应用
+ 蒙特卡洛法计算定积分：
  $$
  \begin{aligned}
  I=\int_0^1f(x)dx&=\int_0^1f(x)\cdot 1dx\\
  &=E(f(X))\hspace{1em}(X\sim U(0,1))\\
  &=EY\hspace{1em}(Y=F(X))
  \end{aligned}
  $$
  故可以在$(0,1)$中随机生成$n$个均匀分布的随机数$x_1,\cdots,x_n$，再由大数定律，
  $$
  I\approx\frac{1}{n}\sum_{k=1}^nf(x_k)
  $$
### 更一般的大数定律（不要求数学期望存在）
+ 设$\{X_n,n\geq 1\}$为相互独立的随机变量序列，对任意$n\geq 1$，记
  $$
  Y_{n,k}=\left\{
    \begin{aligned}
    &X_k,& |X_k|\leq n\\
    &0,& |X_k|>n
    \end{aligned}
  \right.
  $$
  另记$a_n=\sum_{k=1}^nEY_{n,k},S_n=\sum_{k=1}^nX_k$，若$n\to\infty$时，
  $$
  \begin{aligned}
  &\sum_{k=1}^nP(|X_k|>n)\to 0\\
  &\frac{1}{n^2}\sum_{k=1}^nEY_{n,k}^2\to 0
  \end{aligned}
  $$
  则有
  $$
  \frac{S_n-a_n}{n}\stackrel{P}{\longrightarrow}0
  $$
  即$\{X_n,n\geq 1\}$满足大数定律。
+ 利用这一构造，也可以证明辛钦大数定律（略）。
### 强大数定律
+ 定义：对于随机变量序列$\{X_n,n\geq 1\}$，若存在数列$\{a_n,n\geq 1\}$与$\{b_n,n\geq 1\}$（$\{b_n\}$单调增且趋于无穷），且有$n\to\infty$时，
  $$
  \frac{S_n-a_n}{b_n}\stackrel{a.s.}{\longrightarrow}0
  $$
  其中$S_n=\sum_{k=1}^nX_k$.则称$X_n$满足强大数定律。
  + 特别地，当$\{X_n,n\geq 1\}$独立同分布，且数学期望$EX_1=\mu$存在，则有：$n\to\infty$时，
  $$
  \frac{S_n}{n}\stackrel{a.s.}{\longrightarrow}\mu
  $$
## 中心极限定理
+ 与大数定律的区别：大数定律研究的是随机变量和的均值与和的期望的关系；而中心极限定理研究的是随机变量和的分布与正态分布的关系
+ 定义：设$\{X_n,n\geq 1\}$为随机变量序列，$S_n=\sum_{k=1}^n,n\geq 1$，若数学期望$ES_n$与方差$\mathrm{Var}S_n$均存在，且
  $$
  S^*_n=\frac{S_n-ES_n}{\sqrt{\mathrm{Var}S_n}}
  $$
  依分布收敛于标准正态分布$N(0,1)$，则称$\{X_n,n\geq 1\}$满足中心极限定理。
+ 性质：若$\{X_n,n\geq 1\}$满足中心极限定理，则对任意$x\in\mathbb{R}$，
  $$
  \lim_{n\to\infty}P(\frac{S_n-ES_n}{\sqrt{\mathrm{Var}S_n}}\leq x)=\Phi(x)
  $$
  其中$\Phi$为标准正态分布$N(0,1)$的分布函数。
+ **Lindeberg-Lévy中心极限定理**：设$\{X_n,n\geq 1\}$为独立同分布的随机变量序列，数学期望为$\mu$，方差为$\sigma^2$，则$\{X_n,n\geq 1\}$满足中心极限定理。
  + 可用于近似计算概率：
    $$
    P(a\leq\sum_{k=1}^nX_n\leq b)\approx\Phi(\frac{b-n\mu}{\sigma\sqrt{n}})-\Phi(\frac{a-n\mu}{\sigma\sqrt{n}})
    $$
    （提示：$S_n$均值为$n\mu$，方差为$n\sigma^2$）
+ **DeMoivre-Laplace中心极限定理**：设$Y_n$服从二项分布$b(n,p)$，则
  $$
  \lim_{n\to\infty}P(\frac{Y_n-np}{\sqrt{np(1-p)}}\leq y)=\Phi(y)
  $$
  其中$\Phi$为标准正态分布$N(0,1)$的分布函数。
  + 注：由于二项分布为离散分布，正态分布为连续分布，故用正态分布近似二项分布时，需要做以下修正：
    $$
    \begin{aligned}
    P(k_1\leq Y_n\leq k_2)&=P(k_1-0.5<Y_n<k_2+0.5)\\
    &\approx\Phi(\frac{k_2+0.5-np}{\sqrt{np(1-p)}})-\Phi(\frac{k_1-0.5-np}{\sqrt{np(1-p)}})
    \end{aligned}
    $$
+ 三类应用：已知$n$和$y$，求概率；已知$n$和概率，求$y$；已知$y$和概率，求$n$。（具体见题目分类汇编）
+ 中心极限定理比直接用切比雪夫不等式作近似计算精度更好。
### *独立不同分布情形（仅作了解）
+ **Lindeberg-Feller条件**：对于独立的随机变量序列$\{X_n,n\geq 1\}$，若$EX_n=\mu_n,\mathrm{Var}X_n=\sigma_n^2<\infty$，记$s_n=\sqrt{\sum_{i=1}^n\sigma_i^2}$，若$\forall\epsilon>0$，
  $$
  \frac{1}{s_n^2}\sum_{i=1}^n E(X_i-\mu_i)^2\cdot 1_{\{|X_i-\mu_i|>\epsilon s_n\}}\to 0\hspace{1em}(n\to\infty)  
  $$
  则称$\{X_n,n\geq 1\}$满足Lindeberg-Feller条件。
+ **Lindeberg-Feller中心极限定理**：若独立随机变量序列$\{X_n,n\geq 1\}$满足Lindeberg-Feller条件，则对任意$x\in\mathbb{R}$，
  $$
  \lim_{n\to\infty}P\left(\frac{\sum_{i=1}^n(X_i-\mu_i)}{S_n}\leq x\right)=\Phi(x)
  $$
+ **Lyapunov定理**：对于独立的随机变量序列$\{X_n,n\geq 1\}$，$EX_n=\mu_n,\mathrm{Var}X_n=\sigma_n^2<\infty$，记$s_n=\sqrt{\sum_{i=1}^n\sigma_i^2}$，若存在$\delta>0$，满足
  $$
  \lim_{n\to\infty}\frac{1}{s_n^{2+\delta}}\sum_{i=1}^n E|X_i-\mu_i|^{2+\delta}=0,
  $$
  则对任意$x\in\mathbb{R}$，
  $$
  \lim_{n\to\infty}P\left(\frac{\sum_{i=1}^n(X_i-\mu_i)}{S_n}\leq x\right)=\Phi(x)
  $$
  + 注：这里$\delta$常取$1$或$2$。
  + 推论（**伯努利分布的中心极限定理**）：若独立随机变量序列$\{X_n,n\geq 1\}$满足$\forall n\geq 1,X_n\sim b(1,p_n)$，记$Y_n=\frac{\sum_{i=1}^n(X_i-p_i)}{\sqrt{\sum_{i=1}^np_i(1-p_i)}}$，若$\sum_{i=1}^\infty p_i(1-p_i)=\infty$，则对任意$x\in\mathbb{R}$，
    $$
    \lim_{n\to\infty}P(Y_n\leq x)=\Phi(x)
    $$
### *Delta方法（也仅作了解）
+ **Delta定理**：设$\{Y_n,n\geq 1\}$为随机变量序列，$F^*$为连续的分布函数，$\theta$为实数，数列$\{a_n,n\geq 1\}$满足$0<a_n\uparrow\infty$，且使得
  $$
  a_n(Y_n-\theta)\stackrel{d}{\longrightarrow}F^*,
  $$
  若$\alpha(\theta)$为$\theta$的函数，且有连续导函数$\alpha'(\theta)\neq 0$，则
  $$
  \frac{a_n(\alpha(Y_n)-\alpha(\theta))}{\alpha'(\theta)}\stackrel{d}{\longrightarrow}F^*
  $$
  即$\frac{a_n(\alpha(Y_n)-\alpha(\theta))}{\alpha'(\theta)}$与$a_n(Y_n-\theta)$具有相同渐近分布$F^*$。
  + 推论：设随机变量序列$\{X_n,n\geq 1\}$独立同分布，具有数学期望$\mu$和方差$\sigma^2$，记$\overline{X_n}=\frac{1}{n}\sum_{k=1}^nX_k$.设$\alpha(\mu)$为$\mu$的函数且有连续的导函数$\alpha'(\mu)\neq 0$，则
    $$
    \frac{\sqrt{n}}{\sigma\cdot\alpha'(\mu)}(\alpha(\overline{X_n})-\alpha(\mu))\stackrel{d}{\longrightarrow}N(0,1).
    $$