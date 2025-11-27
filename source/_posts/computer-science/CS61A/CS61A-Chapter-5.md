---
title: CS61A Chapter 5
date: 2025-11-21 19:20:14
categories:
 - study
 - [计算机科学,CS61A]
 - [python] 
tags: 
 - python
cover: 61A-header.png
---
# 递归函数(Recursion function)
+ 如果一个函数直接或间接调用了自身，那么它就被称作递归函数。一种常见的形式是，在函数体中调用函数自身的功能。
+ 下面以求自然数所有位数的和作为例子：   
递归的核心理念在于简化问题（即“分而治之”的思想）。在将自然数分解的过程中，会频繁用到取模（`n%10`）和整除（`n//10`）操作，而求解`n//10`的所有位数和的方法和`n`是完全一致的。于是就可以用递归函数实现上述操作：
```python
>>> def sum_digits(n):
        """返回正整数n的所有位数之和。"""
        if n < 10:
            return n
        else:
            all_but_last, last = n // 10, n % 10
            return sum_digits(all_but_last) + last
```
+ 如果使用环境帧解释的话，那么在运行时函数体中调用自身相当于创建了一个局部帧，接收来自上一个帧的参数，并返回相应的值（或继续创建局部帧直到参数满足一定条件，直接返回值）
## 结构剖析
+ 在递归函数体中，常常使用条件语句区分基础条件（base case）与需要继续递归的条件。
+ 下面我们对递归与迭代方法进行比较（以计算$n$的阶乘为例）：
```python
>>> def fact_iter(n):
        """迭代法"""
        total, k = 1, 1
        while k <= n:
            total, k = total * k, k + 1
        return total

>>> fact_iter(4)
24
>>> def fact(n):
        """递归法"""
        if n == 1:
            return 1
        else:
            return n * fact(n-1)

>>> fact(4)
24
```
+ 可以看出，迭代方法直接从基础条件`fact(1)=1`开始往目标值不断连乘得到`fact_iter(n)`，而递归方法则是从`fact(n)`开始，将其拆解为待乘项`n`和简化后的问题`fact(n-1)`（即$n!=n\cdot (n-1)!$），向基础条件不断递归，最终到达基础条件并层层返回结果。
+ 另一方面，在参数与变量的选择上，迭代法在函数体中使用了`k`和`total`两个参数记录当前总阶乘与待乘项，而在递归法则直接将`k`嵌入函数的参数中，将`total`作为函数的返回值进行运算。
+ 此外，在设计递归函数的结构时，也使用到了函数抽象——不考虑`fact(n-1)`的具体实现方式，先假定`fact(n-1)`的返回值就是$(n-1)!$。这就将递归函数正确性的验证转化为归纳法（Induction）的形式（可参见[CS70 Chapter 3](/2025/10/06/computer-science/CS70/CS70-Chapter-3/)）
## 互递归（Mutual Recursion）
+ 顾名思义，互递归表示的是两个函数互相调用（也属于递归中的间接调用）。
+ 以下面这个luhn校验算法为例：
```python
def split(n):
    return n // 10, n % 10
def sum_digits(n):
    if n<10:
        return n
    else:
        all_but_last,last = split(n)
        return sum_digits(all_but_last) + last
def luhn_sum(n):
    if n<10:
        return n
    else:
        all_but_last,last = split(n)
        return luhn_sum_double(all_but_last) + last
def luhn_sum_double(n):
    all_but_last,last = split(n)
    luhn_digit = sum_digits(2 * last)
    if n<10:
        return luhn_digit
    else:
        return luhn_sum(all_but_last) + luhn_dig
```
+ 在这段代码中，函数`luhn_sum(n)`和`luhn_sum_double(n)`相互调用，形成互递归（在此例子中即为交替对`n`各位数求和，其中偶数位乘$2$再各位求和）
+ 一般情况下，可以将互递归转化为单一递归函数。这也说明，互递归和单一递归函数本质上没有区别，只是提供了另一种函数抽象的手段。
+ 互递归常常在回合制游戏模型中使用（这样两名玩家各自的策略就可以单独编写）
## 在递归中输出
+ 通过在递归函数中使用`print`语句，可以将递归计算的过程可视化。比如下面这个函数：
```python
>>> def cascade(n):
        """将n的前缀数按山谷型（从大到小再到大）输出."""
        print(n)
        if n >= 10:
            cascade(n//10)
            print(n)
```
注意到这里没有特意编写基础条件，因为无论`n`是否为个位数，它都要先将输出自身。
# 树递归(Tree recursion)
+ 在树递归中，递归函数体内会同时多次调用自身。最典型的例子就是计算斐波那契数列：
```python
def fib(n):
    if n == 1:
        return 0
    if n == 2:
        return 1
    else:
        return fib(n-2) + fib(n-1)

result = fib(6)
```
+这非常符合我们的直觉：直接根据数列的递推式定义编写递归函数。在调用这个函数时，其会产生多个子调用，如此下去，直到回到基础条件。这样的函数调用结构类似于一棵多叉树，所以被称为树递归。   
下面举一个用树递归解决问题的例子：
## 示例：整数拆分
+ 设正整数$n$被拆分为若干个正整数的和（即$n=a_1+a_2+\cdots+a_i,1\leq i\leq n$），且拆分后的正整数不超过正整数$m$（即$a_k\leq m,1\leq k\leq i$）。现求不同的拆分方案数。    
以$n=6,m=4$为例，有下述$9$种不同的拆分方案：
```
1.  6 = 2 + 4
2.  6 = 1 + 1 + 4
3.  6 = 3 + 3
4.  6 = 1 + 2 + 3
5.  6 = 1 + 1 + 1 + 3
6.  6 = 2 + 2 + 2
7.  6 = 1 + 1 + 2 + 2
8.  6 = 1 + 1 + 1 + 1 + 2
9.  6 = 1 + 1 + 1 + 1 + 1 + 1
```
我们定义函数`count_partitions(n, m)`对此问题进行求解。利用树递归的理念，我们可以发现如下规律：
+ 使用最大数为$m$的整数分割$n$的方案的数量等于下面两个数量之和：
  1. 使用最大数为$m$的整数分割$n-m$的方案的数量；
  2. 使用最大数为$m-1$的整数分割$n$的方案的数量。   
+ 使用表达式表述即为：`count_partitions(n, m)=count_partitions(n-m, m)+count_partitions(n, m-1)`
+ 简单证明如下：
  将分割方案分为两组：包含最大数$m$的和不包含$m$的。如果包含$m$，那么去掉一个$m$，则对应上述情况1；如果不包含$m$，那么对应上述情况2。
+ 最后确定基础条件：
  1. $n=0$时方案数唯一
  2. $n<0$时方案数为$0$
  3. $n>0,m=0$时方案数也为$0$
+ 呈上代码：
```python
>>> def count_partitions(n, m):
        """计算使用最大数m的整数分割n的方案数"""
        if n == 0:
            return 1
        elif n < 0:
            return 0
        elif m == 0:
            return 0
        else:
            return count_partitions(n-m, m) + count_partitions(n, m-1)

>>> count_partitions(6, 4)
9
>>> count_partitions(5, 5)
7
>>> count_partitions(10, 10)
42
>>> count_partitions(15, 15)
176
>>> count_partitions(20, 20)
627
```
