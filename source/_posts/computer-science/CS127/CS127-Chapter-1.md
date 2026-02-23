---
title: CS127 Chapter 1
date: 2026-02-18 14:57:31
categories:
    - study
    - [计算机科学,CS127]
    - [最优化理论]
tags:
    - 线性代数
    - 优化模型
cover: EECS127.png
---
本系列笔记参考：
+ Lecture：[EECS 127 sp24](https://space.bilibili.com/326165077/lists/7194391?type=season)
+ 官网：[EECS 127](https://eecs127.github.io/)
+ 课程讲义：[Course reader](https://eecs127.github.io/assets/notes/eecs127_reader.pdf)
+ 教材：[Optimization Models](https://www.cambridge.org/us/universitypress/subjects/engineering/control-systems-and-optimization/optimization-models) 与 [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
# 课程介绍
+ 本课程需要的先修条件：
  + 线性代数或类似课程（第一节课会对相关知识进行回顾）
  + CS70（离散数学）相关课程（作为逻辑推理的基础）
  + 微积分相关课程（会涉及到梯度等计算）
+ 本课程使用python语言进行编程，所以也需要一定的python基础（CS61A水平即可）。
+ 本课程核心围绕最优化理论展开，在回顾完线性代数内容后，首先会介绍**梯度下降**（gradient descent）等优化模型，接着会讲解函数的**凸性**（Convexity），然后会讲解**对偶性**（duality），最后会介绍最优化理论的应用（控制论、机器学习算法等）
## 关于优化
+ 优化渗透于生活中方方面面。无论是生产还是生活，一旦选择了目标（使其最大化/最小化等），那么就不可避免地会涉及到目标的优化。
+ 优化的核心在于目标及损失函数的选择，它们会决定最终优化的结果。比如，在一些回归问题中，会使用L1/L2损失函数，而在一些分类问题中，则会使用分类错误率作为损失函数……有时，我们还会涉及到多个目标的优化（比如航班调度），或者队列优化等等!!有点运筹学的感觉!!
+ 一个标准的优化问题包括一个需要最小化的目标函数（如果是最大化那就取负值）以及可行解的集合（一般由约束条件决定）。
## 优化问题示例
+ 假设有一家天然气公司需要精炼$10^5$桶原油。现在有两家炼油厂：一家加工为航空燃油，另一家则加工为汽油。航空燃油售价$0.10$美元/桶，汽油售价$0.20$美元/桶。优化目标为最大化利润。
+ 其他限制如下：
  + 政府要求公司至少生产$1000$桶航空燃油与$500$桶汽油。
  + 原油储存点距离航空燃油厂$10$英里，距离汽油厂$30$英里，而总运输不超过$2\times 10^6$桶英里（即运输里程与运输桶数乘积之和不超过$2\times 10^6$）
+ 假设生产航空燃油的桶数为$x_1$，生产汽油的桶数为$x_2$，那么我们就能构建以下优化模型：
  $$
  \begin{aligned}
  &\max_{x_1,x_2}\hspace{0.5em}{\frac{1}{10}x_1+\frac{1}{5}x_2}\\
  &\mathrm{s.t.}\left\{
    \begin{aligned}
        &x_1\geq 1000\\
        &x_2\geq 500\\
        &10x_1+30x_2\leq 2\cdot 10^6\\
        &x_1+x_2=10^5
    \end{aligned}
    \right.
  \end{aligned}
  $$
  这个模型是一个典型的**线性规划**模型，其求解方法会在之后讨论。
## 标准的优化问题
+ 下面给出更一般的优化问题形式：
  $$
  \begin{aligned}
    &\min_{\vec{x} \in \mathbb{R}^n}\hspace{0.5em}f_0(\vec{x}) \\[1em]
    &\text{s.t.} \left\{
    \begin{aligned}
    &f_i(\vec{x}) \leq 0, \quad \forall i \in \{1,\ldots,m\} \\
    &h_j(\vec{x}) = 0, \quad \forall j \in \{1, \ldots, p\}.
    \end{aligned}
    \right.
    \end{aligned}
  $$
  其中：
  + $\vec{x}$为优化向量（包括所有优化参数）
  + $f_i$与$h_j$均为定义在$\R^n\rightarrow\R$上的函数
  + $f_0$为目标函数
  + $f_i(\vec{x}) \leq 0$为不等式约束，而$h_j(\vec{x}) = 0$为等式约束
  + 我们可以将满足约束条件的可行解记为集合$\Omega$，即：
    $$
    \Omega\doteq\left\{\vec{x}\in\R^n\left|\hspace{0.2em}
        \begin{aligned}
            &f_i(\vec{x}) \leq 0, \quad \forall i \in \{1,\ldots,m\} \\
            &h_j(\vec{x}) = 0, \quad \forall j \in \{1, \ldots, p\}
        \end{aligned}\right.
        \right\}.
    $$
    而优化问题就可以记为
    $$
    \min_{\vec{x} \in \Omega}\hspace{0.5em}f_0(\vec{x})
    $$
  + 这一优化问题的解记为$\vec{x}^*\in\Omega$，其在$\Omega$中的所有优化向量$\vec{x}$中能使$f_0(\vec{x})$取到最小值（即$f_0(\vec{x})$在$\Omega$上的极小值点），也可以用$\argmin$表示：
    $$
    \argmin_{\vec{x}\in\Omega}f_{0}(\vec{x}) = \left\{ \vec{x} \in \Omega | f_{0}(\vec{x}) = \min_{\vec{u}\in\Omega} f_{0}(\vec{u}) \right\}
    $$
    解的数量可以不止一个（甚至可以有无穷多个），但一般可以用$\vec{x}^*$代表。
+ 上面的约束条件中可以没有不等式约束/等式约束（如果没有任何约束那么$\Omega=\R^n$，也称为无约束问题）
### 讨论：当最小值/最大值不存在
+ 在上面的优化问题中，我们默认$f_0(\vec{x})$在$\Omega$上的最小值能取到，但在一些特殊情况下，可能无法取到最小值（如开区间端点）。
+ 此时我们就要用到分析学中的上/下确界（supremum and infimum）的概念。实际上，我们只需要将上面问题中的$\min$改为$\inf$即可。（由确界定理，只要下界存在，下确界一定存在）
+ 在后续的讨论中，我们不会特别考虑最小值能否取到的问题（即默认为$\min$）
## 最小二乘法（Least Squares）
+ 我们可以从最简单（也是最通用）的最优化问题开始：
+ 给定一个数据矩阵$A\in\R^{m\times n}$与一个参考向量$\vec{y}\in\R^m$，目标是找到参数向量$\vec{x}\in\R^n$，使得$\left\|A\vec{x}-\vec{y}\right\|_2^2$最小。其中，$\left\|\cdot\right\|_2$表示向量的L2范数（即欧几里得范数）：
  $$
  \begin{aligned}
    \| \vec{z} \|_2 & \doteq \sqrt{\vec{z}^{\top} \vec{z}} = \sqrt{\sum_{i=1}^{n} z_i^2}.
    \end{aligned}
  $$
+ 对于这一问题的求解，有以下定理：设$A\in\R^{m\times n}$列满秩，则
  $$
  \min_{\vec{x}\in\R^n}\left\|A\vec{x}-\vec{y}\right\|_2^2
  $$
  的解唯一且表示为
  $$
  \vec{x}^*=(A^\top A)^{-1}A^\top\vec{y}
  $$
  > 注：这里默认$m>n$。$m\leq n$时最小值一定为$0$（$m<n$时$\vec{x}$有无穷解（$A$行满秩），$m=n$时$\vec{x}$有唯一解）
+ 证明：
  + 设$A=(\vec{a_1},\vec{a_2},\cdots,\vec{a_n})$，则$A\vec{x}=x_1\vec{a_1}+x_2\vec{a_2}+\cdots+x_n\vec{a_n}$，可以理解为由$n$个列向量组成的$n$维超平面（一定过原点，因为$\vec{x}=\vec{0}$一定在超平面上）。
  + 对于$m$维向量$\vec{y}$，因为$m>n$，所以它不一定在超平面上（如果在超平面上则最小值为$0$）。设$\vec{y}$不在超平面上（并假设它的起点为原点），那么可以将$\vec{y}$对超平面投影，定义投影向量为$\vec{z}$，$\vec{e}=\vec{y}-\vec{z}$。示意图如下（$n=2,m=3$情形）：![ls](least_square.png)
  + 于是根据定义可得$\vec{e}$与整个超平面垂直（与超平面上的任意向量垂直），下面证明$\vec{z}$是超平面上与$\vec{y}$距离最短的向量：
    + 首先，在超平面上任意另取一个向量$\vec{u}$，并定义$\vec{v}=\vec{y}-\vec{u}$，$\vec{w}=\vec{z}-\vec{u}$（易得$\vec{w}\perp\vec{e}$）。示意图如下：![ls2](least_square_2.png)
    + 那么有
        $$
        \begin{aligned}
        ||\vec{y} - \vec{u}||_2^2 &= ||\vec{v}||_2^2 \\
        &= ||\vec{w}||_2^2 + ||\vec{e}||_2^2 \\
        &= \underbrace{||\vec{z} - \vec{u}||_2^2}_{>0} + ||\vec{e}||_2^2 \\
        &> ||\vec{e}||_2^2 \\
        &= ||\vec{y} - \vec{z}||_2^2.
        \end{aligned}
        $$
        所以$\vec{z}$就是距离最短的向量。
  + 下面来求$\vec{z}$对应的向量$\vec{x}^*$。由于向量$\vec{y}-A\vec{x}^*$与超平面垂直，所以向量与$A$的所有列向量垂直，故有
    $$
    \begin{aligned}
    A^\top(\vec{y}-A\vec{x}^*)&=\vec{0}\\
    A^\top A\vec{x}^*&=A^\top\vec{y}\\
    \vec{x}^*&=(A^\top A)^{-1}A^\top\vec{y}
    \end{aligned}
    $$
    其中因为$A$列满秩，所以$A^\top A$可逆。
+ 最后我们介绍最小二乘法用在最基本的应用——线性回归（拟合）。假设给定数据点$(x_1,y_1),\cdots,(x_n,y_n)$，我们尝试用一条直线$y=mx+b$拟合这些数据点。理想情况下，
  $$
    \begin{aligned}
    mx_{1} + b &= y_{1} \\
    mx_{2} + b &= y_{2} \\
    & \vdots \\
    mx_{n} + b &= y_{n}.
    \end{aligned}
  $$
  用矩阵的形式表示：
  $$
  \begin{aligned}
    \begin{bmatrix}
    x_1 & 1 \\
    x_2 & 1 \\
    \vdots & \vdots \\
    x_n & 1
    \end{bmatrix}
    \begin{bmatrix}
    m \\
    b
    \end{bmatrix}
    =
    \begin{bmatrix}
    y_1 \\
    y_2 \\
    \vdots \\
    y_n
    \end{bmatrix}
    .
    \end{aligned}
  $$
  当方程组有解时，拟合的直线就能精确覆盖所有数据点；而当方程组无解时，就可以用最小二乘法得到最佳拟合直线。
+ 最后补充：最小二乘问题的求解之所以简单，是因为它属于“凸”问题（convex problem），其性质为任意局部最优解都是全局最优解。当然现实中也有不少优化问题是非凸的，其求解（包括凸性理论）会在后续讨论。