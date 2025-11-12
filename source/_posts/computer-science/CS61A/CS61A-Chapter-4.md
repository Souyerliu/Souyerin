---
title: CS61A Chapter 4
date: 2025-10-19 17:48:35
categories:
 - study
 - [计算机科学,CS61A]
 - [python] 
tags: 
 - python
cover: 61A-header.png
---
# Higher-Order Functions（高阶函数）
+ 之前我们已经了解到，函数通过接收以表达式为形式的参数，输出表达式（值）。而高阶函数则是将表达式改为函数，即接收函数形式的参数，或者输出形式为函数。
+ 这种高阶函数具有以下特点：模块化(Modularity)，抽象化(Abstraction)以及关注点分离(Separation of Concerns)。
## 函数作为参数
+ 有时我们在设计多个函数时，这些函数之间可能会高度相似，只有微小的部分存在区别，比如：
```python
from math import pi, sqrt
def area_square(r):
    """正方形面积"""
    return r*r
def area_circle(r):
    """圆形面积"""
    return r*r*pi
def area_hexagon(r):
    """正六边形面积"""
    return r*r*3*sqrt(3)/2
```
这三个函数都使用到了r*r，只有后面的系数不同。那么我们就可以使用一个“ **通用模版（general patterns）** ”来计算面积，而落实到具体的计算时使用嵌套函数：
```python
def area(r,shape_constant):
    assert r > 0, 'A length must be positive' # 断言语句，当输入r为负数时报错并中止
    return r * r * shape_constant
def area_square(r):
    return area(r, 1)
def area_circle(r):
    return area(r, pi)
def area_hexagon(r):
    return area(r,3*sqrt(3)/2)
```
+ 通用模版不仅能节省数的计算，还能概括一系列函数执行的过程，是一种强大的抽象方法。比如下例：
```python
def summation(n, term): #注：这里term传递的是函数名称（类似C语言中的函数指针）
    total, k = 0, 1
    while k <= n:
        total, k = total + term(k), k + 1
    return total

def cube(x):
    return x*x*x
def sum_cubes(n):
    return summation(n, cube)

def pi_term(x):
    return 8 / ((4*x-3) * (4*x-1))
def pi_sum(n): #这个级数可以计算π的近似值
    return summation(n, pi_term)

result1 = sum_cubes(3)
result2 = pi_sum(1e6)
```
输出：
```python
36
3.141592153589902
```
## 函数作为通用方法（General methods）
+ 有些函数可以作为一种通用的方法，即不考虑函数体内各参数的具体算法。我们以下面这个用迭代改进（iterative improvement）法计算黄金分割比$\phi$的程序为例：
```python
def improve(update, close, guess=1):
    while not close(guess):
        guess = update(guess)
    return guess

def golden_update(guess):
    return 1/guess + 1

def square_close_to_successor(guess):
    return approx_eq(guess * guess,guess + 1)

def approx_eq(x, y, tolerance=1e-3):
    return abs(x - y) < tolerance

phi = improve(golden_update,square_close_to_successor)
```
在上面的这段代码中，`improve`函数就是一种通用的方法，它不考虑`update`和`close`函数的具体形式，而构建了一个整体框架。因此，将`update`和`close`函数改为不同的形式，就可以迭代计算不同的值。
+ 这说明：
  1. 通过命名和函数可以让我们对大量复杂的事物进行抽象。通过简单的函数定义，我们可以解决一些复杂的计算。
  2. 由于Python语言有通用的求解过程，我们可以将小的求解组件合为一个复杂的程序。（有点分形的感觉）
+ 当然，每次我们设计完这样的高阶函数后，需要进行调试以验证设计的正确性，这里不详细讨论。
## 定义函数：嵌套定义
+ 注意到，上面这些方法存在一些不足：
  1. 大量的函数命名使得全局帧变得混乱（占用大量命名空间）
  2. 像上面`improve`函数的`update`参数只能接收包含一个参数的函数。
+ 于是我们使用嵌套定义解决这一问题。
+ 以计算一个数的平方根为例，我们使用以下两个函数：
```python
>>> def average(x, y):
        return (x + y)/2

>>> def sqrt_update(x, a):
        return average(x, a/x)
```
经过反复迭代更新求得`a`的平方根。
+ 但是，我们无法将`sqrt_update`函数放进上面`improve`函数的`update`参数中（因为它有两个参数），另外它自身也不能重复迭代更新。于是我们使用嵌套定义：
```python
>>> def sqrt(a):
        def sqrt_update(x):
            return average(x, a/x)
        def sqrt_close(x):
            return approx_eq(x * x, a)
        return improve(sqrt_update, sqrt_close) #improve函数定义同上
```
在函数体内定义的函数（也称为局部定义函数）类似局部赋值，它们只在函数体内发挥作用，也就是只影响局部帧。
+ 局部定义函数参数的调用满足 **词法作用域（Lexical scope）**： 即局部定义函数也可以访问整个**定义**作用域内的名称变量（注意不是调用作用域）。比如上面的`sqrt_update`函数就可以调用`sqrt()`里的参数`a`。
  + 当然局部定义函数也可以调用全局环境中定义的名称变量，因为它的环境继承了定义它的函数的环境。
**继承环境（Extended Environments）** ：通过调用局部定义函数，实际上我们创建了一个更长的帧链——原来我们的环境只有全局帧和（一级）局部帧，现在我们可以在局部帧下进一步构建新的局部帧（有种嵌套递归的感觉，不是吗？）
另外，局部定义的函数由于会使用外部函数的参数值，所以也被成为 **“闭包”(closures)** 。
## 函数作为返回值
+ 我们还可以设计一些接收参数函数，同时返回值也是函数的函数。比如下面这个函数，它的功能就是返回一个定义为它的两个参数函数组合的函数：
```python
>>> def compose1(f, g):
        def h(x):
            return f(g(x))
        return h
```
其实上面这个例子理解起来也比较直观——`h(x)`函数相当于是`f`和`g`两个函数的复合。

## 示例：牛顿法
+ 下面我们将通过一个示例展示如何将函数局部定义与返回函数值用于实际应用中。
+ 牛顿法是一种经典的迭代方法，常用于寻找数学函数的零点，这对于求解一些算式和方程非常有用（实际上许多计算器甚至计算机的解方程算法都是基于牛顿法）
+ 基本算法：在函数（前提：在零点附近可微）的零点附近取一点$(x,f(x))$，过此点作函数切线，与x轴有一交点$(x',0)$。判断$f(x')$，如果$|f(x')|$足够小（在容忍误差范围内），则返回零点近似值$x'$。否则从$(x',f(x'))$继续迭代。
+ 接下来是具体的代码实现：
```python
>>> def newton_update(f, df):
        """通过下面update函数更新x坐标"""
        def update(x):
            return x - f(x) / df(x) # df(x)为f(x)的导函数
        return update
>>> def find_zero(f, df):
        """最终调用的函数"""
        def near_zero(x):
            """判断f(x)是否足够接近0"""
            return approx_eq(f(x), 0)
        return improve(newton_update(f, df), near_zero) # improve函数同上
```
其实和上面的求平方根函数结构比较类似。基于牛顿法求零点，我们可以写出下面的求n次方根的程序：
```python
>>> def power(x, n):
        """返回 x * x * x * ... * x，n 个 x 相乘"""
        product, k = 1, 0
        while k < n:
            product, k = product * x, k + 1
        return product

>>> def nth_root_of_a(n, a):
        def f(x):
            return power(x, n) - a   # 即方程为x^n=a
        def df(x):
            return n * power(x, n-1) # 导函数
        return find_zero(f, df) # 定义见上

>>> nth_root_of_a(2, 64)
8.0
>>> nth_root_of_a(3, 64)
4.0
>>> nth_root_of_a(6, 64)
2.0
```
+ 我们可以通过调整`approx_eq()`函数中的`tolerance`改变解的精度。另外，迭代初始值的选择也很重要，这决定了函数的收敛速度（有时初值选择不佳甚至可能无法收敛）
## 柯理化（Currying）
+ 我们在处理多参数高阶函数时，可以将这些其拆分为多个单参数函数链，这一过程就被称为柯理化。
+ 以下面这个计算幂次的函数为例：
```python
>>> def curried_pow(x):
        def h(y):
            return pow(x, y)
        return h
>>> curried_pow(2)(3)
8
```
我们将函数`pow(x, y)`转化为了`curried_pow(x)(y)`。这里高阶函数`curried_pow()`接收参数`x`并返回另一个接收参数`y`的函数`h()`。
+ 将多元函数柯理化的好处在于，我们可以将函数的功能进行拆分细化，这可以让函数功能的组合更加灵活。比如我们就可以直接使用`curried_pow(x)`计算`x`的前10次幂：
```python
>>> def map_to_range(start, end, f):
        while start < end:
            print(f(start))
            start = start + 1
>>> map_to_range(0, 10, curried_pow(2))
1
2
4
8
16
32
64
128
256
512
```
我们不需要再另外编写一个复杂的函数实现这一功能。
+ 另外，我们也可以自主编写将函数柯理化（以及逆柯理化uncurrying）的通用代码：
```python
>>> def curry2(f):
        """返回给定的双参数函数的柯里化版本"""
        def g(x):
            def h(y):
                return f(x, y)
            return h
        return g
>>> def uncurry2(g):
        """返回给定的柯里化函数的双参数版本"""
        def f(x, y):
            return g(x)(y)
        return f
>>> pow_curried = curry2(pow)
>>> pow_curried(2)(5)
32
>>> map_to_range(0, 10, pow_curried(2))
1
2
4
8
16
32
64
128
256
512
```
+ 这里，`curry2(f)(x)(y)`等价于`f(x,y)`，而`uncurry2(curry2(f))`则等价于`f`。
+ 一个有趣的事实：Currying这个名称来源于逻辑学家Haskell Curry，但实际上这一技术是由Moses Schnfinkel首先发明的。（btw，Haskell语言也是来源于Haskell Curry）
## Lambda表达式（匿名函数）
+ 对于一些局部定义的函数，我们有时候只需要它返回的结果，而对它的命名不那么重要。这是，我们就可以使用Lambda表达式临时创建函数。比如下例：
```python
>>> def compose1(f, g):
        return lambda x: f(g(x))
>>> f = compose1(lambda x: x * x,lambda y: y + 1)
>>> result = f(12)
169
```
+ 这种函数除了没有名称外，这样的函数与其他函数没有任何区别。
+ 当然lambda函数也可以返回一个lambda函数，甚至一个lambda函数可以没有输入参数（如`c = lambda : x + y`，`x`和`y`为给定常值，则调用`c()`返回`x+y`的值）
+ 虽然lambda表达式看起来更加简洁，但也在一定程度上减弱了程序的可读性。因此，lambda匿名函数一般只适用于计算一些简单的表达式。
+ 关于使用lambda作为匿名函数的符号的原因，有下面这段小故事：
> 注意：用 lambda 来引入一个过程或函数看起来也许有些古怪。这个记号可以追溯到 1930 年代的阿隆佐·丘奇（Alonzo Church）。他最初使用一个“帽子”符号来表示函数，例如把平方函数写作 “ŷ . y × y”。不过，由于印刷排版人员对此符号感到困扰，他们把这个“帽子”移到了参数的左边，并改成了一个大写的 Λ，于是变成 “Λy . y × y”；后来，大写的 Λ 又被改成了小写的 λ，于是在数学书中我们现在看到的是 “λy . y × y”，而在 Lisp 语言中则写作 (lambda (y) (* y y))。如果由我来决定的话，我会用 fun，或者也许是 ^。 —— Peter Norvig (norvig.com/lispy2.html， 由GPT-5翻译)
+ lambda 表达式和对应的函数应用的形式语言——$\lambda$演算（lambda calculus）是计算机科学中的基本概念。之后在解释器的设计中，我们将重新讨论这个问题。
## 抽象与一等函数（Abstractions and First-Class Functions）
+ 用户定义函数是编写程序中非常重要的一个手段，因为它能让我们将计算时的通用方法转化为代码中的显式元素（就和一般的变量名称一样）。而高阶函数则可以对这些通用方法进行操作（如作为参数接收，作为返回值输出等），以构建更进一步的抽象。
+ 当然，并不是说抽象化程度越高的程序就越好。作为程序员，需要针对特定的任务决定编写代码的抽象化程度。
+ 一般而言，程序中的一些元素会被赋予一等地位（first-class status），这种地位所具有的权利是：
  + 可以与名称绑定
  + 可以作为参数传递给函数
  + 可以作为函数的结果返回
  + 可以包含在数据结构中
+ python中的函数就具有这种地位，这很大程度上提升了程序的表达能力。
## 函数装饰器（Function Decorators）
+ 最后我们介绍一个python的特殊语法——装饰器。装饰器使用高阶函数作为执行 def 语句的一部分，最常见的例子是下面的`trace`：
```python
>>> def trace(fn):
        def wrapped(x):
            print('-> ', fn, '(', x, ')')
            return fn(x)
        return wrapped

>>> @trace
    def triple(x):
        return 3 * x

>>> triple(12)
->  <function triple at 0x102a39848> ( 12 )
36
```
高阶函数`trace`的作用是返回一个函数，这个函数在调用参数前先输出它使用的参数（在`@trace`之后定义的函数相当于在`trace`函数内进行了定义，但运行环境在全局帧）。上面这段代码等价于：
```python
>>> def triple(x):
        return 3 * x
>>> triple = trace(triple)
```
