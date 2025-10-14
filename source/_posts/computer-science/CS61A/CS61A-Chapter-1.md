---
title: CS61A Chapter 1
date: 2025-10-03 21:22:39
categories:
 - study
 - [计算机科学,CS61A]
 - [python] 
tags: 
 - python
 - Expressions
cover: 61A-header.png
---
+ 本系列笔记基于CS61A在线Lecture，课件和教材内容进行编写。
+ CS61A官网：[CS61A](https://cs61a.org/)
+ 教材官网：[Composing Programs](https://www.composingprograms.com/) (此教材基于经典教材[SICP](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/index.html)进行编写)
+ 教材中文翻译版：[CP zh-CN ver.](https://composingprograms.netlify.app/)
+ 注：笔者使用的是windows系统，故以下内容均基于windows操作。

# 在开始之前
##  安装python 3:
   + [python3 download](https://www.python.org/downloads/)
   +  安装时需要注意配置环境变量（以在命令行更方便地操作）
##  交互页面(Interactive Sessions)
   +  在python交互页面（可以在自带的IDLE或其他终端中显示），在`>>>`后输入python代码，python会自动读取并执行输入的命令
   +  比如：
      ```python
      >>> 33+16
      49
      ```
## 表达式与语句(Expressions and statements)
+ 广义上（对任意的编程语言），计算机程序包括以下两部分：
  + 计算值(Compute some value)
  + 执行操作(Carry out some action)
+ 这两个部分分别对应 **表达式(Expressions)** 与 **语句(Statements)** （当然两者常常混用）
+ 比如以下的赋值语句：
  ```python
  >>> lpwang=chr(71)+chr(79)+chr(65)+chr(84)
  ```
  将`lpwang`与`=`后面的表达式相连接，这个表达式将四个ASCII码对应的字符进行拼接，并赋值到`lpwang`上!!lpwang is GOAT！!!
## 函数(Functions)
+ 函数是对一系列数据操作的封装（如上面的`chr`就是python的内置函数）
+ 函数可以通过一个简单的表达式省略复杂的操作过程
+ 我们也可以自定义一些函数使程序可读性更强
+ 将会在后续详细讨论
## 对象(Objects)
+ 整合了数据和操作数据的逻辑（可以理解为数据结构+函数）
+ 将会在后续详细讨论
## 解释器(Interpreters)
+ 用于计算复合表达式
+ 在python中，程序的执行依靠的就是解释器
+ 和 **编译器(Compliers)** 的区别：
  + 解释器会将代码一条一条解释并执行，而编译器先将代码编译为目标代码，再在目标平台上执行(如C和C++代码编译成二进制代码(.exe程序)，以在windows平台执行)。
  + 解释器的优点在于跨平台运行代码更加容易，而编译器的优点则是编译后运行效率远高于解释运行的程序。
+ 将会在后续详细讨论
## 报错(Errors)
+ 程序的运行总是会遇到各种各样的错误
+ 遇到错误时，尝试读取错误信息，并进行调试(debug):
  + 在编写完每一个组件后，及时测试以确保组件运行正常；
  + 诊断问题时，精确到一行代码的某一部分；
  + 有时程序员的设想与代码的实现会有偏差，在调试时确保代码符合你的设想；
  + 必要时请教他人
# 值，运算符与表达式(Values, Operators, and Expressions)
+ 对于任何程序语言，它都包含以下三个构建机制：
  + 原始的表达式与语句(构成最简单的代码块)；
  + 组合元素的方式；
  + 抽象元素的方式（即将组合元素作为新的基本单元进行操作）。
+ 在程序中，我们主要处理两种元素：**数据(data)** 和 **函数(functions)** （当然二者之间没有严格界限）
## 表达式的类型(Type of Expressions)
+ 在python中，以下均为表达式：
  + `2`，`1000-7`，`2**64`，`pi`，`'I am stupid.'`，...
  + 对于纯数学符号的表达式，在python中使用中缀表示法(infix notation)，即操作符(如加减乘除)出现在操作数之间
  + 对于包含其他符号的表达式，也有各种不同的复合表示方法，之后会详细介绍
+ 我们通常使用`f(x)`泛指所有的表达式(即函数符号)
### 调用表达式(Call Expressions)
+ 包括两部分：操作符(operator)和操作数(operand)
+ 运算过程：
  1. 将操作符转化为函数(function)
  2. 计算操作数，作为函数的参数(argument)
  3. 将参数代入函数中，得到返回值(value)
+ 有时调用表达式可以嵌套其他调用表达式，运算顺序由里到外（类似递归结构）
  + 具体的例子：
  ```python
    mul(add(4, mul(4, 6)), add(3, 5))
  ```
  其中`add`为加法运算，`mul`为乘法运算，那么运算顺序就是：先计算`mul(4, 6)`得24，再计算`add(4,24)`得28,然后计算`add(3,5)`得8，最后计算`mul(28,8)`得224。
+ 有一些调用表达式中参数有顺序，不得随意调换
+ 在调用表达式中，使用函数符号比使用中缀表示法有以下优点：
  + 函数符号总在参数之前，因而可以一次性接收多个参数而无需写多个函数符号
  + 函数符号表达式的嵌套结构更加清晰（理论上可以嵌套任意多层，但需要先保证可读性）
  + 任何名称的运算符都可以用函数符号表示（如幂、对数、根号等）
### 赋值语句(Assignment Statements)
+ 利用`=`，将`=`右边的表达式取值后赋值到`=`左边的表达式中
+ 当然，我们也可以通过`import`语句赋予一些名称特定值，比如：
  ```python
  >>> from math import pi
  >>> pi*114/514
  0.6967734679168023
  ```
+ 赋值是最简单的 **抽象** 方法，因为我们可以用简单的符号表达复合操作的结果
+ 另外，我们也可以用`=`将函数赋予到特定的名称上，比如：
  ```python
  >>> f = max
  >>> f
  <built-in function max>
  >>> f(2, 3, 4)
  4
  ```
+ 这种被赋值的名称又被称为 **变量名(variable names)** 或 **变量(variables)**,因为它们随时会因为新的赋值语句发生变化。
+ 有时还可以同时为多个变量赋值，变量间和赋值间用`,`隔开：
```python
>>> radius=10
>>> area, circumference = pi * radius * radius, 2 * pi * radius
>>> area
314.1592653589793
>>> circumference
62.83185307179586
```
+ 利用这一特性，我们可以比较方便地交换两个变量的值：
```python
>>> x, y = 3, 4.5
>>> y, x = x, y
>>> x
4.5
>>> y
3
```