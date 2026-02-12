---
title: CS61A Chapter 7
date: 2026-02-08 16:54:23
categories:
 - study
 - [计算机科学,CS61A]
 - [python] 
tags: 
 - python
cover: 61A-header.png
---
# 数据抽象（Data Abstraction）
+ 在前面的章节中，我们主要讨论的是函数的一些抽象（即将函数的具体实现与函数的模块化处理分离）。而本节开始，我们将聚焦于数据本身的抽象。
+ 在现实世界中，数据往往无法直接用程序语言的内置数据类型直接表示，通常需要不同数据类型进行复合表示。而数据抽象就是让数据的具体组成细节和其使用的方式进行分离，使得设计的函数不需要考虑复合数据的构成，将其作为整体进行处理。当然，也需要函数作为链接数据整体与具体部分的桥梁。
+ 下面以有理数的构建与计算为例具体阐释这一概念：
## 示例：有理数
+ 这里我们将有理数统一表示为`分子(numerator)/分母(denominator)`的形式。同时，我们可以假设以下函数已经有了定义（相当于先射箭后画靶）：
  + `rational(n,d)`：返回$\dfrac{n}{d}$形式的有理数；
  + `numer(x)`：返回有理数$x$的分子；
  + `denom(x)`：返回有理数$x$的分母。
+ 那么，我们就可以定义以下关于有理数的函数了：
```python
>>> def add_rationals(x, y): # 加法
        nx, dx = numer(x), denom(x)
        ny, dy = numer(y), denom(y)
        return rational(nx * dy + ny * dx, dx * dy)

>>> def mul_rationals(x, y): # 乘法
        return rational(numer(x) * numer(y), denom(x) * denom(y))

>>> def print_rational(x): # 输出
        print(numer(x), '/', denom(x))

>>> def rationals_are_equal(x, y): # 相等判断
        return numer(x) * denom(y) == numer(y) * denom(x)
```
+ 现在我们回过头来看看有理数的表示函数是如何构建的。因为有理数可以表示为分子与分母的一对（pair），所以我们可以利用上节提到的列表来表示：
```python
>>> def rational(n, d):
    return [n, d]

>>> def numer(x):
        return x[0]

>>> def denom(x):
        return x[1]
```
  + 当然我们也可以不使用列表，而使用高阶函数表示（比较难理解一些）：
    ```python
    def rational(n,d):
        def select(name):
            if name == 'n':
                return n
            elif name =='d':
                return d
        return select
    def numer(x): # 这里的x是调用rational返回的函数，下同
        return x('n')
    def denom(x):
        return x('d')
    ```
+ 这样我们就完整定义了有理数的构建与运算。具体使用例如下：
```python
>>> half = rational(1, 2)
>>> print_rational(half)
1 / 2
>>> third = rational(1, 3)
>>> print_rational(mul_rationals(half, third))
1 / 6
>>> print_rational(add_rationals(third, third))
6 / 9
```
+ 当然，这里的结果还没有化简到最简分数。我们可以通过对分子与分母同时除以最大公约数`gcd`得到化简结果：
```python
>>> from math import gcd
>>> def rational(n, d):
        g = gcd(n, d)
        return (n//g, d//g) # 注：这里使用了元组（tuple）这一结构，它和列表（list）非常相似，除了tuple无法对数据进行修改
```
## 抽象屏障（Abstraction Barriers）
+ 我们可以对上面的有理数示例进行总结：数据抽象的本质在于将某类数值的所有操作行为进行分组，每组独立设计各自的操作函数，互不影响。这种限制可以更好地将程序进行模块化处理，使得调试维护代码时的影响最小化。
+ 当然，这些分组会形成一种层级关系。比如，上面的例子中：`add_rational, mul_rational, rationals_are_equal, print_rational`这四个函数位于最高层，`rational, numer, denom`处在中间层，而双元素列表/元组则处于最底层。低层的函数/数据抽象通过高层函数的调用得以实现功能。
+ 在理想情况下，高层的函数只会调用其相邻低层的函数。如果其调用了更低层的函数，就会破坏抽象屏障，这会减弱代码的鲁棒性。
+ 总而言之，在数据抽象中，只要数据的**构建方式**（也称为构造函数Constructors）与**选取方式**（也称为选择函数Selectors）之间满足指定的规则，它们的具体实现方式就可以灵活多变，这就为代码的编写提供了更多的灵活性。
# 可变数据（Mutability）
