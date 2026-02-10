---
title: CS70 Chapter 6
date: 2025-11-27 21:58:14
categories: 
 - study
 - [计算机科学,CS70]
 - [离散数学]
tags:
 - 离散数学
 - 初等数论
cover: penguin_and_pigeon.png
---
注：本节可参考国内《信息安全数学基础》《初等数论》《抽象代数》等相关内容（即如果学习过以上课程，下面内容可以跳过）。
# 模运算(Modular Arithmetic)
在计算机的一些领域（如密码学），我们通常希望处理一串范围较小的数字。这时，模运算就起到了巨大的作用，它能将数压缩到一个较小的范围$\{0,1,\cdots,N-1\}$，从而简化大量运算。   
在生活中，最常见的模运算例子就是时钟。时钟以12小时为一循环（如果以24小时制计算则为24小时一循环），因而我们能够轻松计算当前时刻若干小时后的时刻。   
用数学语言描述的话，我们可以定义$x\hspace{0.3em}(\operatorname*{mod}m)\coloneqq r$，其中$r$为$x$除以$m$得到的余数。即：
$$
x\hspace{0.3em}(\operatorname*{mod}m)\equiv r \Longleftrightarrow x=mq+r\hspace{0.5em}(0\leq r\leq m-1,q\in\mathbb{Z})
$$
## 计算法则（加法，减法与乘法）
+ 在计算$x+y\hspace{0.3em}(\operatorname*{mod}m)$时，我们可以先计算$x\hspace{0.3em}(\operatorname*{mod}m)$和$y\hspace{0.3em}(\operatorname*{mod}m)$，再将二者相加取模（这很容易证明）。同理，$x-y\hspace{0.3em}(\operatorname*{mod}m)$也等于$(x\hspace{0.3em}(\operatorname*{mod}m)-y\hspace{0.3em}(\operatorname*{mod}m))(\operatorname*{mod}m)$。
+ 在乘法运算上，这一法则的效果更明显：$xy(\operatorname*{mod}m)=(x\hspace{0.3em}(\operatorname*{mod}m))\times(y\hspace{0.3em}(\operatorname*{mod}m))(\operatorname*{mod}m)$，借助这一公式，计算的规模可以大大减小。
# 集合表示（Set representation）
+ 下面我们换个视角分析模运算。对于任意整数$m$与$x,y$，如果$m$整除$x-y$，则称$x$与$y$对$m$同余(congruent modulo m)，即：
$$
x\equiv y\hspace{0.3em}(\operatorname*{mod}m)\Longleftrightarrow m\mid (x-y)
$$
+ 显然，在这个式子中，$x$与$y$除以$m$得到的余数相同，故上述式子也可记为：
$$
x\equiv y\hspace{0.3em}(\operatorname*{mod}m)\Longleftrightarrow x\hspace{0.3em}(\operatorname*{mod}m)=y\hspace{0.3em}(\operatorname*{mod}m)
$$
+ 于是我们可以将所有对$m$同余的数组成一个集合（如$\{\cdots,-m,0,m,2m,\cdots\}$，$\{\cdots,-m+1,1,m+1,2m+1,\cdots\}$等），一共可以组成$m$个集合。这些集合可以覆盖所有整数，也被称为模$m$的剩余类(residue classes mod m)。  
+ 通过建立这些集合，我们就可以理解上述模运算法则的原理。由于每个集合里的所有元素均相互同余，所以我们可以将运算的数转化到$\{0,1,\cdots,m-1\}$这个范围内。    
+ 下面再用数学语言将这一命题进行阐述（可以直接用定义证明，故证明过程省略）：    
  如果$a\equiv c\hspace{0.3em}(\operatorname*{mod}m)$且$b\equiv d\hspace{0.3em}(\operatorname*{mod}m)$，那么$a\pm b\equiv c\pm d\hspace{0.3em}(\operatorname*{mod}m)$，同时$a\cdot b\equiv c\cdot d\hspace{0.3em}(\operatorname*{mod}m)$。
+ 注意到我们这里没有涉及到除法运算，这将在后面被讨论。
# 指数运算
+ 在密码学中，指数运算也是一种重要的运算。那么，对于$x^y$（$x,y$均为自然数），如何求它对$m(m>1)$的模呢？一种朴素的方法是依次计算$x\hspace{0.3em}(\operatorname*{mod}m),x^2\hspace{0.3em}(\operatorname*{mod}m),\cdots$一直到$x^y\hspace{0.3em}(\operatorname*{mod}m)$，但这样效率太低（如果用程序实现时间复杂度可达$O(2^y)$）。
+ 为了优化这一运算，我们可以使用下述“重复平方(repeated squaring)”技巧：
  + 设计递归函数`mod_exp(x,y,m)`，返回值为$x^y\hspace{0.3em}(\operatorname*{mod}m)$；
  + 如果$y=0$，则返回`1`（$x^0\equiv 1\hspace{0.3em}(\operatorname*{mod}m)$）;
  + 如果$y\neq 0$，设`z=mod_exp(x,y//2,m)`（$z=x^{\lfloor\frac{y}{2}\rfloor}\hspace{0.3em}(\operatorname*{mod}m)$）;
    + 如果$y$为偶数，则返回$z^2\hspace{0.3em}(\operatorname*{mod}m)$；
    + 如果$y$为奇数，则返回$xz^2\hspace{0.3em}(\operatorname*{mod}m)$。
+ 这样算法的时间复杂度就降到了$O(\log y)$（具体关于算法时间复杂度的知识可参见CS61B）。
# 双射(Bijecions)
+ 在介绍除法运算前，先做一个铺垫。这里我们回顾一下高等数学中映射的概念：
  + **映射(mapping)**：设$X$,$Y$是两个给定的集合，若按照某种规则$f$，使得$X$中的每一个元素，都可以在$Y$中找到唯一的元素与之对应，则称$f$是集合$X$到集合$Y$的一个映射。
  + **单射(one-to-one)**：设$f$是集合$X$到集合$Y$的一个映射，若$f$的逆像也具有唯一性，即对$X$中的任意两个不同元素$x_1\neq x_2$,也满足$f(x_1)\neq f(x_2)$（也可表述为$f(x_1)=f(x_2)\Longrightarrow x_1=x_2$），则$f$为单射。
  + **满射(onto)**：如果$\forall y\in Y$，$\exists x\in X$满足$f(x)=y$，则称$f$为满射。
  + **双射**：如果$f$既是单射又是满射，那么$f$就是双射。
+ 利用这些概念，我们考虑以下函数（均将集合$\{0,1,\cdots,m-1\}$映射到自身）：
$$
f(x)\equiv x+1\hspace{0.3em}(\operatorname*{mod}m)
$$
$$
g(x)\equiv 2m\hspace{0.3em}(\operatorname*{mod}m)
$$
+ 可以发现，映射$f$是双射（集合中的每一个$y$，都存在唯一原像$y-1$）；而映射$g$只有当$m$是奇数时才是双射（$\{0,1,\cdots,m-1\}$映射到$\{0,2,\cdots,m,1,3,\cdots,m-1\}$），当$m$为偶数时既不是单射也不是满射。
+ 另外关于双射，有以下定理：
  + 对于一个有限集合$A$，如果映射$f:A\rightarrow A$存在对应的逆映射$g:A\rightarrow A$满足$g(f(x))=x$，则$f$为双射。
# 逆(Inverses)
+ 之前我们进行减法模运算时，注意到$-b\equiv m-b\hspace{0.3em}(\operatorname*{mod}m)$，所以加法模运算很容易拓展到减法模运算。
+ 然而想通过乘法模运算推导出除法模运算就困难一些。在实数域上，我们知道除以$x$等价于乘以$\frac{1}{x}$。相应的，计算$\frac{a}{x}\hspace{0.3em}(\operatorname*{mod}m)$，就需要找到$y$，使得$x\cdot y\equiv 1\hspace{0.3em}(\operatorname*{mod}m)$，那么就可以将问题转化为$a\cdot y\hspace{0.3em}(\operatorname*{mod}m)$。这里$y$也被称为$x$模$m$的乘法逆(multiplicative inverse)。
+ 那么，能否保证$y$一定存在？$y$又是否唯一呢？我们先举一些例子：令$x=9,m=17$，则当$y=2$时，$xy=18\equiv 1\hspace{0.3em}(\operatorname*{mod}17)$，即$2$是$9$模$17$的乘法逆；而当$x=8,m=12$时，取$y=\{1,2,3,\cdots\}$，$xy\hspace{0.3em}(\operatorname*{mod}m)=\{8,4,0,8,4,\cdots\}$循环，无法取到$1$，即$8$模$12$的乘法逆不存在。
+ 关于$x$模$m$的乘法逆的存在性与唯一性，有下述定理：
  + 当$m$和$x$满足$\operatorname*{gcd}(m,x)=1$（即$m$与$x$互质），则$x$模$m$的乘法逆存在且唯一。
+ 证明如下：
  + 考虑以下数列：$\{0\hspace{0.3em}(\operatorname*{mod}m),x\hspace{0.3em}(\operatorname*{mod}m),2x\hspace{0.3em}(\operatorname*{mod}m),\cdots,(m-1)x\hspace{0.3em}(\operatorname*{mod}m)\}$。可以证明这一数列没有重复值（使用反证法：如果存在$a,b\in [0,m-1]\cap\mathbb{Z}$，使得$ax\equiv bx\hspace{0.3em}(\operatorname*{mod}m)$，则$m\mid (a-b)x$，而$m$与$x$互质，故只能$m\mid a-b$，但$|a-b|< m$，产生矛盾）。因此数列包含$\{0,1,\cdots,m-1\}$中的每一个值，故一定存在$y\in [0,m-1]\cap\mathbb{Z}$，使得$y$为$x$模$m$的乘法逆。
  + 下面证明$y$唯一性：如果存在$y'$满足$y'x\equiv 1\hspace{0.3em}(\operatorname*{mod}m)$，那么$y\equiv yy'x\equiv y'\hspace{0.3em}(\operatorname*{mod}m)$。又因为$y,y'\in\{0,1,\cdots,m-1\}$，所以$y=y'$。
  + 当然，这一定理的逆命题同样成立（当$x$模$m$的乘法逆存在$\Longrightarrow m$与$x$互质）简单证明如下：设$a$为$x$模$m$的乘法逆，则$ax\equiv 1\hspace{0.3em}(\operatorname*{mod}m)$，即存在$k\in\mathbb{Z}^+$，使得$ax=km+1$。而如果$\operatorname*{gcd}(m,x)=d>1$，则$\operatorname*{gcd}(ax,km)>1$，产生矛盾。
+ 我们一般将$x$模$m$的乘法逆记为$x^{-1}$，就像一般的算术一样。下面我们将介绍如何利用最大公约数计算$x^{-1}$。
## 乘法逆的计算
+ 事实上，对于最大公约数，我们可以得到以下结论（也被称为**裴蜀定理**）：
  + 如果$\operatorname*{gcd}(a,b)=d$，那么一定存在整数$x,y$，使得$d=ax+by$
+ 基于这一结论，我们可以得出：$1=\operatorname*{gcd}(m,x)=am+bx,a,b\in\mathbb{Z}$。从而$bx\equiv 1\hspace{0.3em}(\operatorname*{mod}m)$，那么$b$对$m$取模即为我们所求的乘法逆。
### 欧几里得算法（辗转相除法）
+ 首先，我们定义$\operatorname*{gcd}(a,0)=a,\operatorname*{gcd}(0,b)=b$。对于非零的两数$x,y$，存在以下定理：
  + 设$x\geq y>0$，则$\operatorname*{gcd}(x,y)=\operatorname*{gcd}(y,x(\operatorname*{mod}y))$。
  + 证明很容易（直接使用模运算定义即可），故省略。
+ 通过这一定理，可以对$x$和$y$反复互相取模，直到其中一项变为$0$，则另一项即为$x$和$y$的最大公因数。
+ 我们可以将这一算法用python代码实现：
```python
def gcd(x,y):
  if y==0:
    return x
  else:
    return gcd(y,x%y)
```
+ 这里我们再分析一下这个程序运行的时间复杂度（递归调用的次数）。可以证明，`gcd(x,y)`在经过两次递归后最大项不会超过$\frac{x}{2}$：
  + 若$y\leq\frac{x}{2}$，则在一次递归后，最大项已经不超过$\frac{x}{2}$，故第二次递归也一定不超过；
  + 若$\frac{x}{2}< y\leq x$，则在两次递归后，最大项会变为$x(\operatorname*{mod}y)<\frac{x}{2}$。
+ 所以最多经过$2\lceil\log_2n\rceil$次递归后，$x$变为$0$。故时间复杂度为$O(\log x)$。
+ 下面我们就使用欧几里得算法求解乘法逆。
### 欧几里得算法的扩展
我们将上面求解最大公约数的代码进行扩展：
```python
def extended_gcd(x,y):
  if y==0:
    return x,1,0
  else:
    d,a,b=extended_gcd(y,x%y)
    return d,b,a-(x//y)*b
```
+ 这个函数输入$x,y$后返回$d,a,b$三个值，满足$d=\operatorname{gcd}(a,b)=ax+by$。可以验证，当$d=1$时，$b$即为$x$模$y$的乘法逆。
+ 那么这个算法是如何设计出来的呢？我们可以尝试逆向递推：
  + 当$y=0$时，返回值$d=x,a=1,b=0$，显然满足$d=ax+by$；
  + 当$y>0$时，首先通过递归调用`extended_gcd(y,x%y)`得到$d=\operatorname*{gcd}(y,x(\operatorname*{mod}y))$，且$d=a\cdot y+b\cdot x(\operatorname*{mod}y)$，然后再返回$d,a'=b,b'=a-\lfloor\frac{x}{y}\rfloor b$。
  + 很明显$d$即为最大公约数（和先前的方法相同），而对于$a'$和$b'$，有以下推导：
  $$
  \begin{gathered}
  d=a\cdot y+b\cdot x(\operatorname*{mod}y)\\
  =a\cdot y+b(x-\lfloor\frac{x}{y}\rfloor y)\\
  =bx+(a-\lfloor\frac{x}{y}\rfloor b)y
  \end{gathered}
  $$
  这便是$a'$和$b'$表达式的由来。
+ 与先前的算法相比，仅仅是增加了常数倍的计算量，时间复杂度仍为$O(\log x)$。
+ 当然，除了递归，我们还可以直接采用递推法求解。注意到$d\mid x$且$d\mid y$，所以$d$一定整除由$x$和$y$组成的线性组合。那么基于下面两个等式：
  $$
  \begin{gathered}
  (1)x+(0)y=x\\
  (0)x+(1)y=y
  \end{gathered}
  $$
对两个式子的右边做辗转相除（类似$x-\lfloor\frac{x}{y}\rfloor y$的操作）,左边也作出相应的运算，最后一定能得到$ax+by=d$。
## 除法模运算
+ 在得到乘法逆后，我们就可以做除法模运算了。一种典型的使用场景如下：
  + 已知$8x\equiv 11(\operatorname*{mod} 15)$，求$x$。
  + 为解这一方程，注意到$8$模$15$的乘法逆为$2$，故$x=22(\operatorname*{mod} 15)=7$。而且这是取模后的唯一解。
# 算术基本定理（Fundamental Theorem of Arithmetic）
+ 在初等数论中，有一个非常重要的定理：任何一个大于$1$的整数可分解为若干个质数的乘积。事实上，我们可以用拓展后的欧几里得算法进行证明。
+ 首先证明一个引理：
  + 设$x,y,z\in\mathbb{Z^+}$且$\operatorname*{gcd}(x,y)=1$，那么如果$x\mid yz$，则$x\mid z$。
  + 简单证明：因为$\operatorname*{gcd}(x,y)=1$，所以存在整数$a,b$，满足$ax+by=1$。两边乘以$z$得$axz+byz=z$，因为$x\mid axz$，$x\mid byz$（因为$x\mid yz$），所以$x\mid axz+byz=z$。
+ 有了这个引理，既可以对算术基本定理进行证明：
  + 在[归纳法例4](/2025/10/06/computer-science/CS70/CS70-Chapter-3/#%E4%BE%8B4)中，我们已经证明了任意大于$1$的整数$n$都可以表示为一个或多个质数的乘积$p_1p_2\cdots p_k$。那么，就只需要证明这一序列在对质因子进行排序后唯一即可。（用数学语言描述即为：如果$n=p_1p_2\cdots p_k=q_1q_2\cdots q_l$，$p_i,q_j(1\leq i\leq k,1\leq j\leq l)$均为质数，那么$k=l$，且$\{p_i\}$与$\{q_j\}$仅顺序不同）
  + 下面给出具体证明：
  + 对于$p_1$，因为$p_1\mid n$，所以$p_1\mid q_1q_2\cdots q_l$，又因为$p_1,q_1,\cdots,q_l$均为质数，所以$p_1$一定等于$q_1,q_2,\cdots,q_l$的其中之一（设为$q_{j1}$）。于是将等式$p_1p_2\cdots p_k=q_1q_2\cdots q_l$两边各除去$p_1$和对应的$q_{j1}$。
  + 以此类推，可以得到$p_2,p_3,\cdots,p_k$对应的$q_{ji}$项。最终左式除到$1$，此时右式还剩下$l-k$项。又因为质数均大于$1$，所以只有当$l-k=0$，即$l=k$时等式成立。再由先前已经建立的$p_i$与$q_j$一一对应，命题得证。
+ 这一证明体现了欧几里得算法的核心思想：除法与余数的唯一性（对于任意$x$与$m>0$，存在唯一的$q$与$r(0\leq r\leq m-1)$，使得$x=qy+r$）
# 中国剩余定理（Chinese Remainder Theorem）
最后我们介绍另一个与模运算有关的定理。我们首先给出这个定理的简化形式：
+ 对于任意$m,n$满足$\operatorname*{gcd}(m,n)=1$，存在唯一的$x\hspace{0.3em}(\operatorname*{mod}mn)$，满足：
  $$
  x\equiv a\hspace{0.3em}(\operatorname*{mod}n)\text{且}x\equiv b\hspace{0.3em}(\operatorname*{mod}m)
  $$
+ 证明如下：
  + 先证明存在性：因为$\operatorname*{gcd}(m,n)=1$，由先前乘法逆的存在性定理，分别存在$m$关于$n$的乘法逆（记为$m^{-1}$）与$n$关于$m$的乘法逆（记为$n^{-1}$）。令
    $$
    u\equiv m\cdot(m^{-1}(\operatorname*{mod}n))\hspace{0.3em}(\operatorname*{mod}n),v\equiv n\cdot(n^{-1}(\operatorname*{mod}m))\hspace{0.3em}(\operatorname*{mod}m)
    $$
    注意到$u\equiv 1\hspace{0.3em}(\operatorname*{mod}n)$且$u\equiv 0\hspace{0.3em}(\operatorname*{mod}m)$（类似有$v\equiv 1\hspace{0.3em}(\operatorname*{mod}m)$且$v\equiv 0\hspace{0.3em}(\operatorname*{mod}n)$），因此令$x=ua+vb$，则有$x\equiv 1\cdot a+0\cdot b\equiv a\hspace{0.3em}(\operatorname*{mod}n)$（类似有$x\equiv b\hspace{0.3em}(\operatorname*{mod}m)$），故$x$满足命题要求。
  + 再证明唯一性：设$x$与$y$均满足命题要求，则有$n\mid(x-y)$且$m\mid(x-y)$，又因为$m,n$互质，故$mn\mid(x-y)$，即存在$k\in\mathbb{Z}$，使得$x-y=kmn$。而$x,y\in\{0,1,\cdots,mn-1\}$，所以$x-y$只能等于$0$，即$x=y$。
+ 将这一定理进行推广，就得到了下述**中国剩余定理**：
  + 设$n_1,n_2,\cdots,n_k$为正整数且两两互质，那么对于任意数列$\{a_i\}(i=1,2,\cdots,k)$，存在唯一的$x\in[0,\prod_{i=1}^kn_i]$满足以下方程组：
    $$
    \begin{cases}
    x\equiv a_1\hspace{0.3em}(\operatorname*{mod}n_1)\\
    \cdot\cdot\equiv\cdots\\
    x\equiv a_i\hspace{0.3em}(\operatorname*{mod}n_i)\\
    \cdot\cdot\equiv\cdots\\
    x\equiv a_k\hspace{0.3em}(\operatorname*{mod}n_k)
    \end{cases}
    $$
    且
    $$
    x\equiv\left(\sum_{i=1}^ka_ib_i\right)(\operatorname*{mod}N),\hspace{1em}N=\prod_{i=1}^kn_i,\hspace{0.3em}b_i=\frac{N}{n_i}\left(\frac{N}{n_i}\right)_{n_i}^{-1}
    $$
    其中$\left(\dfrac{N}{n_i}\right)_{n_i}^{-1}$表示$\dfrac{N}{n_i}$模$n_i$的乘法逆。
  + 这里$b_i(\operatorname*{mod}n_j)$的取值只可能为$1$（当$i=j$）与$0$（当$i\neq j$）因此，如果将$x$看成一个$k$维向量（第$i$个元素为$x\hspace{0.3em}(\operatorname*{mod}n_i)$），那么$b_i$就可以看成第$i$个元素为$1$，其余元素均为$0$的基向量，而$x$为$b_1,\cdots,b_k$的线性组合（所以$b_1,\cdots,b_k$乘以任意倍数均满足方程）
+ 证明$x$的唯一性与上面的二元情况类似，就不再赘述了。
+ 最后，我们注意到如果用$(x_a,x_b)$表示$x$（$x_a=x(\operatorname*{mod}n),x_b=x(\operatorname*{mod}m)$），用$(y_a,y_b)$表示$y$，那么就可以用$(x_a+y_a,y_a+y_b)$表示$x+y$（即$x_a+y_a,y_a+y_b$可以唯一确定$x+y$）。对于乘法也是类似，$(x_ay_a,x_by_b)$可以唯一表示$xy$。因此$\{0,1,\cdots,mn-1\}$与$\{0,1,\cdots,m-1\}\times\{0,1,\cdots,n-1\}$形成了一个同构（环同构）。