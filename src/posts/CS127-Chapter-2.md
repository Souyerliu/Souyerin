---
title: CS127 Chapter 2
date: 2026-02-25 23:08:53
categories: [计算机科学,CS127]
tags:
    - 线性代数
    - 优化模型
    - 最优化理论
cover: ./CS127-Chapter-2/EECS127.png
---
# 线性代数回顾
## 范数（Norms）
+ 向量的范数定义如下：设$\mathcal{V}$为$\R$上的向量空间，$f$为$\mathcal{V}$到$\R$上的映射，若$f$满足：
  + 非负性：$\forall \vec{x}\in\mathcal{V},f(\vec{x})\geq 0$，且等号成立当且仅当$\vec{x}=\vec{0}$；
  + 正齐性：$\forall\alpha\in\R,\forall\vec{x}\in\mathcal{V},f(\alpha\vec{x})=|\alpha|f(\vec{x})$；
  + 三角不等式：$\forall \vec{x},\vec{y}\in\mathcal{V},f(\vec{x}+\vec{y})\leq f(\vec{x})+f(\vec{y})$.  

  则称$f$为范数。
+ 在空间中满足条件的范数不止一个，最常见的如$L1$范数，$L2$范数（也称欧几里得范数）等。这里我们主要讨论一类重要的范数：$l^p$范数。其定义如下：
  + 设$1\leq p<\infty$，则$\R^n$上的$l^p$范数表示为
    $$
    \|\vec{x}\|_p\doteq\left(\sum_{i=1}^n|x_i|^p\right)^{\frac{1}{p}}
    $$
  + 当$p=\infty$，定义无穷范数
    $$
    \|\vec{x}\|_\infty\doteq\max_{i=\{1,2,\cdots,n\}}|x_i|=\lim_{p\to\infty}\|\vec{x}\|_p
    $$
    + 对第二个等号的简单证明：设$M=\max_i|x_i|$，则若$M=0$，结论显然成立；若$M>0$，则
      $$
      \|\vec{x}\|_{p}=\left(\sum_{i=1}^{n}\left|x_{i}\right|^{p}\right)^{1/p}=\left(M^{p}\sum_{i=1}^{n}\left(\frac{\left|x_{i}\right|}{M}\right)^{p}\right)^{1/p}=M\left(\sum_{i=1}^{n}\left(\frac{\left|x_{i}\right|}{M}\right)^{p}\right)^{1/p}
      $$
      可以利用夹逼得到
      $$
      \lim_{p\to\infty}\left(\sum_{i=1}^{n}\left(\frac{\left|x_{i}\right|}{M}\right)^{p}\right)^{1/p}=1
      $$
      故等号成立。
    