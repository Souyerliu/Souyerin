---
title: CS61A Chapter 6
date: 2025-12-04 12:19:33
categories:
 - study
 - [计算机科学,CS61A]
 - [python] 
tags: 
 - python
cover: 61A-header.png
---
本节开始，聚焦的主题由函数转变为数据，包括不同的数据类型、类与对象、数据处理以及面向对象编程等内容。
# 原始数据类型（Native Data types）
+ 在python中，每一个值的类型都由一个类(class)来定义（如`int`,`float`,`str`等）。具有相同类型的值可以执行相同的操作或进行组合（如四则运算等）。
+ 而python中内置了一些数据类型（比如上面的这些例子），其中数值类型包括`int`（整数）、`float`（浮点数）和`complex`（复数）三类。【当然，python语言不需要特别声明原始数据类型，而会根据值的格式自动确定，如`int`为一连串数字，`float`为包含小数点的一串数字，`complex`为`整数+整数j`】
+ 另外，关于“浮点数”名称的由来，是缘于其表示的数值存在精度限制（可参考[IEEE 754](https://zh.wikipedia.org/wiki/IEEE_754)）
+ 当然，还有许多非数值类型的数据，不过我们可以通过某些方法将它们转化为数值型；除此之外，还有一些集合化的数据，如列表、字符串等……
+ 关于数值型的数据类型就不多赘述，下面我们从序列开始讲起：
# 序列（Sequences）
序列是一组有顺序的值的集合，也是一个重要的抽象概念。它的形式并不唯一，且可以包含不同类型的数据（在python中），不过它们也具有一些共同的特征，比如：
+ **长度**：一个序列具有有限的长度，而一个空序列的长度为$0$。
+ **数据-索引对应**：序列中的每个数据元素都与一个索引一一对应。索引一般为非负数，从$0$开始。
## 列表(Lists)
列表是python中最重要的一种内置序列。示例如下：
```python
>>> digits = [1, 1, 4, 5, 1, 4]
>>> len(digits)
6
>>> digits[3] # 当然也可以用getitem(digits,3)
5
>>> digits[-1] # 当查询的索引值为负数，则从后向前返回
4
```
+ 如上面所述，列表`list`内置了返回其长度的函数`len()`，同时也可以通过`列表名[索引]`返回列表中特定位置的值。
+ 除了通用的功能，列表还支持加法与乘法。对多个列表使用加法会返回串联起来的新列表，而对一个列表乘以一个正整数可以返回其复制自身若干次后串联的列表。具体示例如下：
```python
>>> [1,1,4,5,1,4] + [1,9] * 2 + [8,1,0] 
[1, 1, 4, 5, 1, 4, 1, 9, 1, 9, 8, 1, 0]
```
+ 另外，列表本身也可以成为另一个列表的元素（即嵌套列表）。而我们可以通过多次使用`[]`访问嵌套列表内的元素（类似函数嵌套）。示例如下：
```python
>>> pairs = [[33, 16], [44, 6]]
>>> pairs[1]
[33,16]
>>> pairs[1][0]
44
```
+ 我们还可以使用`查询元素 in 列表名`判断元素是否在列表中，如下：
```python
>>> digits = [1, 1, 4, 5, 1, 4]
>>> 1919 in digits
False
>>> 810 not in digits
True
```
## 序列遍历（Sequence Iteration）
有时候，我们需要访问一个列表中的所有元素，最朴素的方法就是使用`while`+索引与序列长度比较判定：
```python
>>> def count(s, value):
        """统计在序列 s 中出现了多少次值为 value 的元素"""
        total, index = 0, 0
        while index < len(s):
            if s[index] == value:
                total = total + 1
            index = index + 1
        return total
>>> count(digits, 8)
2
```
而在python中，可以用`for`循环简化代码：
```python
>>> def count(s, value):
        """统计在序列 s 中出现了多少次值为 value 的元素"""
        total = 0
        for elem in s:
            if elem == value:
                total = total + 1
        return total
>>> count(digits, 8)
2
```
这里使用了下面这样的格式，其中`<expression>`代表了一个可迭代的值（更多内容之后会叙述）
```python
for <name> in <expression>:
    <suite>
```
### 序列分解（Sequence unpacking）
