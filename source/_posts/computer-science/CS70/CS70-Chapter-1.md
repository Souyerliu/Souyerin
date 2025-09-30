---
title: CS70 Chapter 1
date: 2025-09-29 17:06:40
categories:
    - study
    - [计算机科学,CS70]
    - [离散数学]
tags:
    - 命题逻辑
    - 离散数学
cover: penguin_and_pigeon.png
---
课前引入：
程序与处理器的基本原理？——逻辑(Logic)和证明(Proofs)，归纳(Induction)和递归(Recursion)。   
1. 计算机面对的是**离散的对象(Discrete objects)**。   
   + **离散数学(Discrete math)** 有着巨大的应用。       
2. 计算机可以通过学习并与这个世界交互（机器学习，数据分析，机器人...）   
   + 这些都与**概率论(Probability)** 有关。

注：本系列主要基于CS70官网Notes和2015版Lecture编写。
CS70官网：[CS70](https://www.eecs70.org/)
2015版Lecture：[CS70_2015](https://www.bilibili.com/video/BV1e1421r74V/)
2019版Lecture(供参考)：[CS70_2019](https://www.bilibili.com/video/BV1Kq4y1k7gz)

# 课程前置知识：集合与数学符号(Review of Sets and Mathematical Notation)
（概念略，因为高中数学与高等数学完全覆盖，在此只列出一些符号表示）~~权当latex练习~~
1. 集合表示1：$A=\{2,3,5,7,11\}$
2. 元素(elements)属于集合：$x \in A$
3. 元素不属于集合：$x\notin A$
4. 集合表示2：$B=\{\frac{a}{b}\mid a,b \in \mathbb{Z},b\neq 0\}$
5. 集合的基数(cardinality)： $|A|$
6. 子集(subsets)：$A\subseteq B,B\supseteq A,A\subsetneq B,B\supsetneq A$
7. 真子集(proper subsets)：$A\subset B,B\supset A,A\not\subset B,B\not\supset A$
8. 空集(empty set): $\{\},\emptyset$
9. 集合的交与并(intersections and unions)：$A\cap B,A\cup B$ (当$A\cap B=\emptyset$,$A$与$B$互不相容disjoint)
10. 相对补集（relative complement,亦称集合的差）：$A\setminus B,A-B$
11. 一些重要集合：自然数集$\mathbb{N}$，整数集$\mathbb{Z}$，有理数集$\mathbb{Q}$，实数集$\mathbb{R}$，复数集$\mathbb{C}$
12. 集合的笛卡尔乘积（Cartesian product，亦称直积）：$A\times B=\{(a,b)\mid a\in A,B\in B\}$
13. 幂集(power set)：$\mathcal{P}(S)=\{T\mid T\subseteq S\}$，亦可写作$2^{S}$
14. 求和与求积：$\sum_{i=m}^{n}f(i),\prod_{j=p}^{q}g(j)$
15. 任意和存在：$(\forall x \in \mathbb{Z})(\exists y \in \mathbb{Z})(y>x)$

# Chapter 1 Propositional Logic (命题逻辑)
## 命题(Proposition)的定义
  + a statement which is either true or false (非真即假的陈述)
  + 以下例子属于命题:
    + $1+1=2$
    + 在欧氏几何中，三角形的内角和为$180^{circ}$。
    + 如果$n$为非负整数，那么$n^2+n+41$是质数。
    + 任意大于$2$的整数，都可以写成两个质数之和。（哥德巴赫猜想，尽管无法明确知道真假，但也是命题）
  + 以下例子不是命题：
    + $114514+1919810$
    + $0=7x^2+1$(没有明确$x$是多少)
    + ECNU是一所好大学。（什么是“好”？）
  + 注：以上命题常用单个字母表示(如$P,Q$)
## 命题的连接词(Connectives)
### 最基本的三个：与(AND,$\land$)，或(OR,$\lor$)，非(NOT,$\lnot$)
  + 与(conjunction):$P\land Q$为真当且仅当$P$为真且$Q$为真。
  + 或(disjunction):$P\lor Q$为真当$P$和$Q$任意一个为真。
  + 非(negation):$\lnot P$为真当$P$为假。
  + 上述这些语句可称为**命题式(propositional forms)**
  + **排中律(law of the excluded middle)**:对于命题$P$，$P$与$\lnot P$中有且仅有一个是真命题。
    + 也就是说，$P \lor \lnot P$总是真命题，称为重言式(tautology);
    + 而$P \land \lnot P$总是假命题，称为矛盾(contradiction)
### 蕴含(Implication)
  + 最重要也是最微妙的命题式
  + 形式：$P\Longrightarrow Q$(P蕴含Q,相当于“如果P，那么Q”)
  + 可以用真值表(Truth table)表示命题的可能值：
  + | $P$ | $Q$     | $P\Longrightarrow Q$ | $\lnot P\lor Q$|
    | - | - | ----------- | -----------|
    | F  | F  | T                      |T|
    | F  | T  | T                      |T|
    | T  | F  | F                      |F|
    | T  | T  | T                      |T|
  + 由真值表可知，$P\Longrightarrow Q$为假当且仅当$P$为真且$Q$为假。
  + 当$P$为假时，$Q$无论如何取都不会影响结果的真，这种真称为“空真”(vacuously true)。
  + 注意到，$P\Longrightarrow Q$和 $\lnot P\lor Q$在逻辑上等价(它们的真值表相同)，可记作$P\Longrightarrow Q\equiv \lnot P\lor Q$。
  + (当然，对于蕴含的真值表还有一种理解方式：将真视为1，将假视为0，“P蕴含Q”的意思就是P的值$\geq$Q的值)
  + 基于蕴含，我们还可以定义以下命题式：
    + 否命题：$Q\Longrightarrow P$;
    + 逆否命题：$\lnot Q\Longrightarrow \lnot P$;
    + 双向蕴含：$P\Longleftrightarrow Q$.
  + 注意到$P\Longrightarrow Q$和它的逆否命题$\lnot Q\Longrightarrow \lnot P$的真值表相同，所以二者逻辑等价：
    $$
    P\Longrightarrow Q\equiv \lnot Q\Longrightarrow \lnot P
    $$
    + 这一点在证明某些命题时非常有用！
## 量词(Quantifiers)
+ 用于更加明确地表示命题(将大量重复的命题浓缩为一个)，同时也量化命题里**谓词(predicate,or a propositional formula)** 的变量的范围。
  + 比如：“对任意非负整数$n$，$n^2+n+41$是质数”这一命题中，“$n^2+n+41$是质数”就是命题的谓词，其中变量为$n$。
+ 数学上使用“任意”($\forall$)和“存在”($\exists$)量化命题。
+ 在求一个命题的否命题时，可以将它条件的所有量词取反（$\forall$变成$\exists$，$\exists$变成$\forall$），而结论不变。
## 关于“非”连接词的补充：德摩根定理(De Morgan's Laws)
+ 通过真值表可以证明，
$$
  \lnot(P \land Q)\equiv \lnot P\lor\lnot Q,  
  \lnot(P \lor Q)\equiv \lnot P\land\lnot Q
$$
+ 对于命题也是类似的：在求一个包含“非”连接词命题的等价命题时，可以将它条件的所有量词取反（$\forall$变成$\exists$，$\exists$变成$\forall$），并将结论取非。
+ 例：$\lnot(\forall x\exists y,P(x,y))\equiv\exists x\lnot(\forall y,P(x,y))\equiv\exists x\forall y\lnot P(x,y)$