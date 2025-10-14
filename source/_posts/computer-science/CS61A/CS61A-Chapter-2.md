---
title: CS61A Chapter 2
date: 2025-10-07 18:53:25
categories:
 - study
 - [计算机科学,CS61A]
 - [python] 
tags: 
 - python
 - Functions
cover: 61A-header.png
---
# Functions, Values, and Calling(函数，值与调用)
## 非纯print函数(The Non-Pure Print Function)
+ 函数根据输出模式分为以下两种：
  + **纯函数(Pure function)** ：输入参数后通过函数计算返回函数值（函数结果）
    + 比如`abs()`函数，在接收参数后输出参数的绝对值：
    ```python
    >>> abs(-2)
    2
    ```
  + **非纯函数(Non-pure function)** ：在返回函数值之外还会有其他的副作用
    + 最典型的就是产生额外输出的`print()`函数：
    ```python
    >>> print(print(1), print(2))
    1
    2
    None None
    ```
    + 因为`print()`函数本身的返回值为`None`，而python不会自动输出`None`，所以`print()`里面的内容输出就属于副作用。
  + 纯函数相比非纯函数有以下好处：
    + 纯函数更好组成复合的调用表达式（非纯函数无法返回有用的结果）
    + 纯函数的结果更加稳定（相同的参数得到相同的结果）
    + 纯函数对编写计算多个调用表达式的并发程序是必不可少的（之后会详细讨论）
  + 当然，非纯函数也会在后续进行分析讨论
+ 因此，下述函数除了`print()`之外均为纯函数
## 定义新函数
+ 函数是一种更加强大的抽象方式（相比赋值而言），可以将名称与一系列操作绑定在一起。
+ 自定义函数的方式：
```python
def <name>(<formal parameters>):
    return <return expression>
```
+ 自定义函数包含`def`，`<name>`函数名和以逗号分隔的`<formal parameters>`形式参数（这两个也称为函数签名function signature）；在函数体内，包含`return`以及返回的表达式
+ 例子（定义平方函数）：
```python
>>> def square(x):
        return mul(x, x)
```
+ 在自定义函数之后，就可以随时调用函数（就像调用内置函数一样）
### 一些定义函数与命名形参时的注意事项
+ 实际上，在定义函数时，python会将函数名称导入到全局帧(Global frame)中（类似c++里的类列表）\[全局帧中也包含变量名称\]
+ 每一次函数定义或赋予其他名称时，全局帧里对应的名（称为**内在名称(intrinsic name)**）就会指向赋予的名称（称为 **绑定名称(bound name)** ）
+ 例子：
```python
f = max
max = 3
result = f(2, 3, 4)
max(1, 2)  # Causes an error
```
+ 上面第四行函数调用报错是因为第二行将`max`赋值为`3`，所以现在`max`成了变量，无法作为函数调用运算符。
+ 另外，所有的内置函数的形式都如下：
  ```python
  <name>(...)
  ```
### 调用自定义函数
+ 调用用户定义的函数会引入 **局部帧（local frame）** ，它只能被该函数访问。通过一些实参调用用户定义的函数：
1. 在新的局部帧中，将实参绑定到函数的形参上。
2. 在以此帧开始的环境中执行函数体。
+ 环境中帧的顺序会影响通过表达式查找名称而返回的值。因而程序通过名称查找对应表达式的计算结果时，会遵循以下过程：
  + 在环境中寻找该名称，找到最早的含有该名称的帧，其里边绑定的值就是这个名称的计算结果。
### 局部名称(Local names)
+ 一个函数的含义应该与编写者选择的参数名称无关，只要函数体内调用的参数名称与形参名称相同
+ 这就对编程有很大的好处：不同函数体的形参名称可以相同，因为它们各自在不同的局部帧中，不会互相影响。
### 选择名称
+ 虽然形式参数名称不重要，但合适的形参名称可以增强程序的可读性。
+ 可参考[python代码风格指南](https://peps.python.org/pep-0008/)
## 函数抽象
+ 函数定义能够隐藏细节。用户可能不会自己去编写函数，而是从另一个程序员那里获得它，然后将它作为一个“黑盒”，用户只需要调用，而不需要知道实现该功能的细节。
+ Python库就具有此属性，许多开发人员使用这里定义的函数，但很少有人去探究它们的实现。
### 抽象函数的三个核心属性
1. 域(domain)：参数的取值范围（相当于定义域）
2. 范围(range)：返回值的集合（相当于值域）
3. 意图(intent)：计算输入与输出间的关系（以及可能的副作用）（相当于函数的目标）
+ 这些属性不会反映函数具体如何执行，因为这里讨论的对象已经抽象化了。
### 运算符(operators)
+ 补充python中运算符的求值过程：
  + 我们可以将运算符视为调用表达式的简写过程，比如：
```python
>>> 2 + 3 * 4 + 5
19
```
和
```python
>>> add(add(2, mul(3, 4)), 5)
19
```
完全等价。当然两种表示方法各自有优势。
+ Python 还允许使用括号对子表达式进行分组，用以覆盖正常的优先级规则，或使表达式的嵌套结构更加明显。
+ 对于除法，python中提供`/`和`//`两种运算符：
  + `/`为浮点除法，返回小数；
  + `//`为整数除法，返回整数。
## 设计函数
### 什么是一个好的函数？
+ 核心：函数属于抽象。
+ 准则：
  + 一个函数应该只执行一个任务，这个任务可以用一行文本清晰表述。如果需要完成多个任务，那应该给每个任务分配一个函数。
  + 不要过多地写重复的代码块——如果程序有多个部分需要执行相似的功能，那么就可以设计函数并调用解决。
  + 定义函数时考虑函数的通用性，让它的功能覆盖尽可能多的情况。
+ 这些准则可以提高代码的可读性，同时减少代码量，提高效率。
### 函数内文档（python特性）
+ 在python中，可以在函数体的开头撰写描述函数的文档（称为 **文档字符串docstring** ）格式通常如下：
```python
>>> def pressure(v, t, n):
        """Compute the pressure in pascals of an ideal gas.

        Applies the ideal gas law: http://en.wikipedia.org/wiki/Ideal_gas_law

        v -- volume of gas, in cubic meters
        t -- absolute temperature in degrees kelvin
        n -- particles of gas
        """
        k = 1.38e-23  # Boltzmann's constant
        return n * k * t / v
```
+ 之后在调用`help(函数名称)`时，就会输出文档字符串(按`q`键退出)
+ 当然也可以在后面的行末尾加上`#`对特定行进行注释
+ 具体的文档字符串准则详见：<https://peps.python.org/pep-0257/>
### 函数内置测试用例(doctest)
+ python支持将测试用例直接嵌入docstring中，像下面这样：
```python
from operator import floordiv,mod
def divide_exact(n, d):
    """Return the quotient and remainder of dividing N by D.
    >>>q, r = divide_exact(2025,10)
    >>> q
    202
    >>> r
    5
    """
    return floordiv(n, d), mod(n, d)
```
+ 然后在命令行执行代码（`ex.py`为保存的文件名）：
```shell
python -m doctest -v ex.py
```
就会输出以下内容：
```shell
Trying:
    q, r = divide_exact(2025,10)
Expecting nothing
ok
Trying:
    q
Expecting:
    202
ok
Trying:
    r
Expecting:
    5
ok
1 item had no tests:
    ex
1 item passed all tests:
   3 tests in ex.divide_exact
3 tests in 2 items.
3 passed.
Test passed.
```
+ 当然，当程序测试时的输出与文档不符时，也会提示测试失败。这样就便于程序员进行调试。
+ 更多有关doctest的介绍可见：<https://docs.python.org/3/library/doctest.html>
### 函数的默认参数
+ 在python中，可以为函数的某些参数设置默认值，当调用函数时未设置这些参数值时，函数就可以用默认值代替。
+ 比如：
```python
>>> def pressure(v, t, n=6.022e23):
        """Compute the pressure in pascals of an ideal gas.

        v -- volume of gas, in cubic meters
        t -- absolute temperature in degrees kelvin
        n -- particles of gas (default: one mole)
        """
        k = 1.38e-23  # Boltzmann's constant
        return n * k * t / v
>>> pressure(1, 273.15)
2269.974834
>>> pressure(1, 273.15, 3 * 6.022e23)
6809.924502
```
+ 注意上面例子中的`n`与`k`的两个`=`的区别：前者是设置默认值，可以通过输入参数改变；后者是名称绑定数值，无法通过输入改变。