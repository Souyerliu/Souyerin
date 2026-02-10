---
title: CS61A Chapter 6
date: 2026-02-04 12:19:33
categories:
 - study
 - [计算机科学,CS61A]
 - [python] 
tags: 
 - python
cover: 61A-header.png
---
本节开始，聚焦的主题由函数转变为数据，包括不同的数据类型、类与对象、数据处理以及面向对象编程等内容。
> 注：本节内容大致对应国内的数据结构课程，可结合相关资源进行学习。
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
+ 对于元素本身也是序列的序列（类似数学里的集类），可以在`for`循环的`<name>`里设置多个变量将序列里的序列元素分别映射到对应的变量中，比如下面的函数就可以实现统计序列中两个元素相等的子序列数量：
```python
def same_count(pairs):
    same_count = 0
    for x, y in pairs:
            if x == y:
                same_count = same_count + 1
    return same_count

>>> pairs = [[1, 2], [2, 2], [2, 3], [4, 4]]
>>> same_count(pairs)
2
```
+ 这种功能与使用一个等号对多变量同时赋值原理相同。
### 范围（Ranges）
+ `range`是python内置的一种整数序列，一般形式为`range(a,b,r)`，其中`a`和`b`均为整数，表示序列从`a`开始，到`b`结束（不包括`b`）；而`r`表示步长。
+ 最终得到的序列即为：`[a,a+r,a+2r,...]`，到`b`之前结束；
+ 默认情况下，`a=0`，`r=1`.
+ 常常与`for`循环搭配使用，作为遍历序列的方法。当然，如果不需要获得序列的值，只是为了进行重复执行代码，也可以这样写：`for _ in range(n):`，这样程序就可以重复执行$n$次。
## 序列处理（Sequence Processing）
序列作为程序中的一种重要的数据结构，在程序设计中常常可以进行模块化编写——即将一系列语句进行压缩，得到简洁的表达式。
### 列表推导式（List Comprehension）
+ 列表推导式可以对序列中（满足条件）的元素统一代入表达式求值，并返回一个新的序列，一般形式为：`[<map expression> for <name> in <sequence expression> if <filter expression>]`
+ 列表推导式的推导顺序为：`<sequence expression>`$\Longrightarrow$得到`<name>`$\Longrightarrow$根据`<filter expression>`筛选$\Longrightarrow$代入`<map expression>`进行计算并返回值
+ 示例：
```python
>>> odds = [1,3,5,7,9]
>>> [x for x in odds if 25 % x == 0]
[1, 5]
```
### 聚合（Aggregation）
+ 之前提到的聚合函数（比如`sum`，`max`，`mean`等）都可以直接作用于序列上（相当于集成于数组上的函数）
+ 另外，这里再介绍两个新的聚合函数：
    + `all()`：当且仅当序列为空或序列的所有元素`bool`值均为`True`时返回`True`，其他情况返回`True`
    + `any()`：当且仅当序列为空或序列的所有元素`bool`值均为`False`时返回`False`，其他情况返回`True`
### 高阶函数（Higher-Order Function）
+ 我们也可以将序列处理与高阶函数进行结合，将序列处理的表达式函数作为输入，返回的序列作为输出，比如：
```python
>>> def apply_to_all(map_fn, s):
        return [map_fn(x) for x in s]
```
这样一个函数返回的就是`map_fn`函数应用到`s`的每个元素得到的序列；而下面的函数则对封装了序列元素的筛选：
```python
>>> def keep_if(filter_fn, s):
        return [x for x in s if filter_fn(x)]
```
另外，下面的例子：
```python
>>> def reduce(reduce_fn, s, initial):
        reduced = initial
        for x in s:
            reduced = reduce_fn(reduced, x)
        return reduced
```
则是使用了类似递归的思想，将序列中的元素与当前结果值进行组合计算，反复迭代得到最终返回值（比如累乘等）
+ 实际上，上面提到的三个函数在python都有对应的内置名称：
  + `apply_to_all`对应`map`（当然这里需要再使用`list()`返回列表）
  + `keep_if`对应`filter`（同上）
  + `reduce`也是`functools`的内置函数（不需要指定初始值）
+ 当然，在实际应用中，直接使用列表推导式相比高阶函数更常见。
## 序列抽象（Sequence Abstraction）
+ 这里我们总结一下序列的抽象使用方法：
+ 首先是**成员（Membership）**。前面已经提到，可以用`in`或者`not in`判断某个值是否在序列中；
+ 然后是**切片（Slicing）**。切片是指序列中任意一段连续的片段，格式一般为`<sequence_name>[start:end]`，其中切片范围与`range()`函数类似，`start`为起始位置，`end`为终止位置的后一个位置（当然也可以使用负数，意思及为从右往左数的位置）。`start`默认为`0`，`end`默认为序列的长度。
  + 冷知识：python中切片的`start`和`end`可以超过序列的长度。当`end`超过序列长度，则会自动重置为序列长度；当`start`超过序列长度，则返回空表。
## 字符串（Strings）
+ 字符串是一种非常重要的数据结构，它可以储存所有的文本数据。在python中，字符串可以用`''`，`""`包含任意文本来表示。
+ 之前我们接触到的字符串包括：文档字符串(docstrings)，`print`的文本，以及`assert`的断言文本。
+ 字符串同样满足序列的两个基本特征：长度和数据-索引对应。它和列表一样，也可以通过`len`求长度，通过索引获取指定位置的字符（注：在python中，没有单独的字符类型，字符串总是由若干个单字符串组成）。同样的，字符串也可以进行加法与乘法，在此不赘述。
+ 字符串可以进行 **成员资格（Membership）** 判断，例如：
    ```python
    >>> 'kaguya' in "chokaguyahime"
    True
    ```
    不过这和列表的`in`略有区别，字符串查找的是连续的子串，而不是元素。
+ 字符串也可以表示多行文字（这已经在之前的docstrings中使用），具体使用例：
  ```python
    >>> """
    Iroha is 0 while Kaguya is 1.
    Iroha is 1 while Kaguya is 0.
    """
    '\nIroha is 0 while Kaguya is 1.\nIroha is 1 while Kaguya is 0.\n'
  ```
  其中输出内容里的`\n`为换行符，在字符串中占一个字符。
+ **字符串强制转换（String Coercion）**：在python中使用`str()`可以强制将其他类型的元素（比如整型、浮点型、列表等）转化为字符串，例子如下：
  ```python
  >>> strings = ['i','ro','ha']
  >>> str(168) + ' can be pronounced as ' + str(strings)
    "168 can be pronounced as ['i', 'ro', 'ha']"
  ```
+ 关于计算机如何编码字符串的具体内容，可参见：[Strings](http://getpython3.com/diveintopython3/strings.html)
## 树（Trees） 
+ 列表本身能否作为其他列表的元素，这一特性也被称为**闭包特性（Closure Property）**。利用这一特性，我们可以在一个列表中构建复杂的层次结构。而这种结构我们就可以用**树**表示。
+ 这里简单介绍一下树的一些概念（可能和数据结构的定义略有不同）：树是由 **根标签（root label）** 和 **分支（branches）** 构成的序列。每个分支都可以看作一棵树（被称为**子树（sub-trees）**），没有分支的树被称为**叶子（leaf）**。每棵子树的根被称为**节点（Node）**。
+ 我们先试用下面的函数构建树（基于列表）：
```python
>>> def tree(root_label, branches=[]): # 根据根节点与分支列表构建树
        for branch in branches:
            assert is_tree(branch), '分支必须是树'
        return [root_label] + list(branches)

>>> def label(tree): # 返回根节点值
        return tree[0]

>>> def branches(tree): # 返回除去根节点后的树列表
        return tree[1:]
```
+ 下面的函数可以判定一个列表是否构成一棵树（使用了递归的方法）：
```python
>>> def is_tree(tree):
        if type(tree) != list or len(tree) < 1:
            return False
        for branch in branches(tree):
            if not is_tree(branch):
                return False
        return True
```
+ 而下面这个函数则可以判断当前列表是否是叶子：
```python
>>> def is_leaf(tree):
        return not branches(tree)
```
+ 综合这些函数，我们可以完整地构建并判定一棵树：
```python
>>> t = tree(3, [tree(1), tree(2, [tree(1), tree(1)])])
>>> t
[3, [1], [2, [1], [1]]]
>>> label(t)
3
>>> branches(t)
[[1], [2, [1], [1]]]
>>> label(branches(t)[1])
2
>>> is_leaf(t)
False
>>> is_leaf(branches(t)[0])
True
```
+ 当然，也可以直接利用递归的方式，按照某种规律定义节点，并构建整棵树。同时，利用递归还可以进行一些树的计算，比如下面的函数可以对树的叶子数量进行计算：
```python
>>> fib_tree = [5, [2, [1], [1, [0], [1]]], [3, [1, [0], [1]], [2, [1], [1, [0], [1]]]]]
>>> def count_leaves(tree):
      if is_leaf(tree):
          return 1
      else:
          branch_counts = [count_leaves(b) for b in branches(tree)]
          return sum(branch_counts)
>>> count_leaves(fib_tree)
8
```
+ 在此基础上略作修改，我们就可以得到一棵树的所有叶子组成的列表：
```python
def leaves(tree):
    """Return a list containing the leaf labels of tree.
    >>> leaves(fib_tree(5))
    [1,0,1,0,1,1,0,1]
    """
    if is_leaf(tRee):
        return [label(tree)]
    else:
        return sum([leaves(b) for b in branches(tree)], []) # 这里用了sum函数的一个小技巧：sum(列表,[])可以将列表里的所有子列表进行加法合并
```
### 分割树（Partition Trees）
+ 在之前的[整数分割问题](/2025/11/21/computer-science/CS61A/CS61A-Chapter-5/#%E7%A4%BA%E4%BE%8B%E6%95%B4%E6%95%B0%E6%8B%86%E5%88%86)中，我们也可以使用分割树解决。
+ 首先进行分割树的构建。这里我们定义一棵树`partition_tree(n, m)`，表示所有用不超过$m$的整数分割$n$的方案。对于任意非叶子节点：
  + 它的左子树代表用至少一个$m$分割$n$的所有方案（即`partition_tree(n-m, m)`）；
  + 它的右子树代表用不超过$m-1$的整数分割$n$的所有方案（即`partition_tree(n, m-1)`）。
+ 对于叶子节点，如果$n=0$则设为`True`（即完成分割），如果$n<0$或$m=0$则设为`False`（即无法完成分割）。最终得到的函数定义如下：
```python
>>> def partition_tree(n, m):
        """返回一棵用不超过m的整数分割n的分割树"""
        if n == 0:
            return tree(True)
        elif n < 0 or m == 0:
            return tree(False)
        else:
            left = partition_tree(n-m, m)
            right = partition_tree(n, m-1)
            return tree(m, [left, right])

>>> partition_tree(2, 2)
[2, [True], [1, [1, [True], [False]], [False]]]
```
+ 在统计所有的分割方案时，就可以对这棵树进行遍历（使用深度优先搜索，每次搜索到叶子节点时，如果叶子结点值为`True`则输出这条路径的所有节点值）。具体函数如下：
```python
>>> def print_parts(tree, partition=[]):
        if is_leaf(tree):
            if label(tree):
                print(' + '.join(partition)) # 将路径里的元素用+链接
        else:
            left, right = branches(tree)
            m = str(label(tree))
            print_parts(left, partition + [m]) # 对应(n-m,m)
            print_parts(right, partition) # 对应(n,m-1)

>>> print_parts(partition_tree(6, 4))
4 + 2
4 + 1 + 1
3 + 3
3 + 2 + 1
3 + 1 + 1 + 1
2 + 2 + 2
2 + 2 + 1 + 1
2 + 1 + 1 + 1 + 1
1 + 1 + 1 + 1 + 1 + 1
```
+ 另外，使用列表的切片操作也可以将任意的树转换为二叉树（代码略）。
## 链表（Linked Lists）
+ 在python中，虽然没有直接表示链表的数据类型（如C语言的指针），但可以通过嵌套列表（类似单枝树）表达这一结构。比如：
```python
four = [1, [2, [3, [4, 'empty']]]]
```
+ 即一个链表是由一个元素和一个链表（或者`empty`）组成的列表。我们可以设置以下函数创建并检验链表：
```python
>>> empty = 'empty'
>>> def is_link(s):
        """判断 s 是否为链表"""
        return s == empty or (len(s) == 2 and is_link(s[1]))

>>> def link(first, rest):
        """用 first 和 rest 构建一个链表"""
        assert is_link(rest), " rest 必须是一个链表"
        return [first, rest]

>>> def first(s):
        """返回链表 s 的第一个元素"""
        assert is_link(s), " first 只能用于链表"
        assert s != empty, "空链表没有第一个元素"
        return s[0]

>>> def rest(s):
        """返回 s 的剩余元素"""
        assert is_link(s), " rest 只能用于链表"
        assert s != empty, "空链表没有剩余元素"
        return s[1]
```
+ 链表也可通过以下函数获得其长度以及指定元素的索引，所以也满足序列抽象：
```python
>>> def len_link(s):
        """返回链表 s 的长度"""
        length = 0
        while s != empty:
            s, length = rest(s), length + 1
        return length

>>> def getitem_link(s, i):
        """返回链表 s 中索引为 i 的元素"""
        while i > 0:
            s, i = rest(s), i - 1
        return first(s)
```
+ 当然，我们也可以用递归方法实现上面的操作：
```python
>>> def len_link_recursive(s):
        """返回链表 s 的长度"""
        if s == empty:
            return 0
        return 1 + len_link_recursive(rest(s))

>>> def getitem_link_recursive(s, i):
        """返回链表 s 中索引为 i 的元素"""
        if i == 0:
            return first(s)
        return getitem_link_recursive(rest(s), i - 1)

>>> len_link_recursive(four)
4
>>> getitem_link_recursive(four, 1)
2
```
+ 另外，通过递归还可以对链表设计其他的操作（如转换，拼接等）：
```python
>>> def extend_link(s, t):
        """返回一个在 s 链表的末尾连接 t 链表后的延长链表"""
        assert is_link(s) and is_link(t)
        if s == empty:
            return t
        else:
            return link(first(s), extend_link(rest(s), t))

>>> extend_link(four, four)
[1, [2, [3, [4, [1, [2, [3, [4, 'empty']]]]]]]]

>>> def apply_to_all_link(f, s):
        """应用 f 到 s 中的每个元素"""
        assert is_link(s)
        if s == empty:
            return s
        else:
            return link(f(first(s)), apply_to_all_link(f, rest(s)))

>>> apply_to_all_link(lambda x: x*x, four)
[1, [4, [9, [16, 'empty']]]]

>>> def keep_if_link(f, s):
        """返回 s 中 f(e) 为 True 的元素"""
        assert is_link(s)
        if s == empty:
            return s
        else:
            kept = keep_if_link(f, rest(s))
            if f(first(s)):
                return link(first(s), kept)
            else:
                return kept

>>> keep_if_link(lambda x: x%2 == 0, four)
[2, [4, 'empty']]

>>> def join_link(s, separator):
        """返回由 separator 分隔的 s 中的所有元素组成的字符串"""
        if s == empty:
            return ""
        elif rest(s) == empty:
            return str(first(s))
        else:
            return str(first(s)) + separator + join_link(rest(s), separator)
>>> join_link(four, ", ")
'1, 2, 3, 4'
```
### 递归构造（Recursive Construction）
+ 还是上面的整数分割问题。有了链表的思想以及前面分割树的思路，下面的链表序列构造也就不难理解了：
```python
>>> def partitions(n, m):
        """返回一个链表，包含使用最大为 m 的部分对 n 的分割。
        每个分割都表示为一个链表
        """
        if n == 0:
            return link(empty, empty) # 一个包含空分割的链表
        elif n < 0 or m == 0:
            return empty
        else:
            using_m = partitions(n-m, m)
            with_m = apply_to_all_link(lambda s: link(m, s), using_m)
            without_m = partitions(n, m-1)
            return extend_link(with_m, without_m)
```
构建好的`partitions(n,m)`是由若干个列表组成的列表，每个子列表都是一个分割方案。要将这些方案输出，只需要对每个子列表进行统一的操作即可：
```python
>>> def print_partitions(n, m):
        lists = partitions(n, m)
        strings = apply_to_all_link(lambda s: join_link(s, " + "), lists)
        print(join_link(strings, "\n"))

>>> print_partitions(6, 4)
4 + 2
4 + 1 + 1
3 + 3
3 + 2 + 1
3 + 1 + 1 + 1
2 + 2 + 2
2 + 2 + 1 + 1
2 + 1 + 1 + 1 + 1
1 + 1 + 1 + 1 + 1 + 1
```

!!tbh感觉这里的树和链表用python不如用C语言doge!!