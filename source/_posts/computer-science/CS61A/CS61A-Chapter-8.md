---
title: CS61A Chapter 8
date: 2026-02-19 11:21:40
categories:
 - study
 - [计算机科学,CS61A]
 - [python] 
tags: 
 - python
cover: 61A-header.png
---
> 本节对应教材4.2节内容。   
> 注：本节后半段会涉及到一些类与对象的相关内容，可以先跳过，待学过相关知识后再回来看
# 隐式序列（Implicit Sequences）
+ 与之前的显式序列不同，我们可以建立一个对象，可通过其访问序列中的元素（不需要提前计算元素的值）。
+ 一个直接的例子就是前面使用过的`range()`容器，其对应的序列并不是显式存储在内存中的（不然内存可能会爆），而是在访问/调用时实时计算对应元素的值。这种按需求值的方法被称为**惰性计算（lazy computation）**。
## 迭代器（Iterators）
+ 在python中，迭代器作为一种对象提供对序列中的值进行顺序访问的功能!!大致类似C语言的链表指针!!。其含有两个抽象组件：
  1. 检索并返回下一个元素
  2. 当到达序列末尾时发出停止迭代信号
+ 具体而言，对于python中的列表，都可以调用内置的`iter()`函数设置迭代器，并用`next()`访问迭代器内容，比如：
    ```python
    >>> primes = [2, 3, 5, 7]
    >>> type(primes)
    <class 'list'>
    >>> iterator = iter(primes)
    >>> type(iterator)
    <class 'list-iterator'>
    >>> next(iterator) # 注意是先输出对应位置的值，再将位置后移
    2
    >>> next(iterator)
    3
    >>> next(iterator)
    5
    ```
    当`iterator`到达序列末尾时，使用`next`会触发`StopIteration`异常。通常可以使用`try`与`except`语句进行处理：
    ```python 
    >>> next(iterator)
    7
    >>> next(iterator)
    Traceback (most recent call las):
    File "<stdin>", line 1, in <module>
    StopIteration

    >>> try:
            next(iterator)
        except StopIteration:
            print('No more values')
    No more values
    ```
+ 迭代器会保持其本地状态（local state）以记录对应序列的位置。每次调用`next`都会使迭代器记录位置向后一次。对于一个序列，可以定义不同的迭代器（记录不同的位置）；而当两个变量指向同一个迭代器，则它们记录的位置共享：
  ```python
  >>> r = range(3, 13)
  >>> s = iter(r)  # r 的第一个迭代器
  >>> next(s)
  3
  >>> next(s)
  4
  >>> t = iter(r)  # r 的第二个迭代器
  >>> next(t)
  3
  >>> next(t)
  4
  >>> u = t        # u 绑定到 r 的第二个迭代器
  >>> next(u)
  5
  >>> next(u)
  6
  ```
+ 如果在迭代器上使用`iter()`，则返回的仍然是迭代器本身，而并非其副本（这一点与`list()`函数不同）。
+ 迭代器实际上提供了一种按顺序读取元素而不需要同时存取所有元素的方法。其原理类似`range()`，直接通过序列的起始与终止端点计算对应位置的值。这种顺序访问的方式虽然不如直接通过索引求值（即随机访问）灵活，但其可以节省大量内存空间。
## 可迭代值
+ 任何可以产生迭代器的值都称为可迭代值（iterable value）。在Python中，可迭代值是任何可以传递给内置`iter`函数的值，主要包括**序列值**（如字符串与元组）和其他容器（如集合与字典），当然也包括迭代器本身。
+ 对于集合这种无序容器，其迭代顺序无法由程序员控制，但其顺序确实遵循某种规律（实际上在同一次运行中迭代顺序是相同的）
+ 而对于字典而言，如果对其元素进行添加或删除，则其原有的迭代器会立刻失效（使用`next`会报错）；如果对`list`进行增删操作则不会使迭代器报错（但不建议再使用）；对元素的修改不会对迭代器产生影响。
## 内置迭代器
+ 除了`iter`外，python中还有一些内置的函数可以接收可迭代值并返回迭代器。下面以`map`函数为例，展示其用法：
  ```python
  >>> def double_and_print(x):
        print('***', x, '=>', 2*x, '***')
        return 2*x
  >>> s = range(3, 7)
  >>> doubled = map(double_and_print, s)  # double_and_print 未被调用
  >>> next(doubled)                       # double_and_print 调用一次
  *** 3 => 6 ***
  6
  >>> next(doubled)                       # double_and_print 再次调用
  *** 4 => 8 ***
  8
  >>> list(doubled)                       # double_and_print 再次调用两次
  *** 5 => 10 ***                         
  *** 6 => 12 ***
  [10, 12]
  ```
  注意上面的`list(doubled)`会将迭代器`doubled`直接迭代到结束。（其实`list()`函数本身就是不断调用`next()`直到`StopIteration`）
+ 除`map`函数外，python还有`filter`、`zip`与`reversed`等函数可以返回迭代器。（具体使用方法可参考：[1](https://www.cnblogs.com/xb9259/p/18773798)，[2](https://www.codeleading.com/article/73516066206/)）
## For 语句
+ 我们可以通过迭代器的角度重新审视`for`语句：
  ```python
  for <name> in <expression>:
      <suite>
  ```
  上面的语句中，`<expression>`里往往是可迭代对象。具体而言，`for`语句会先对`<expression>`调用`__iter__`方法（生成迭代器），然后在触发`StopIteration`异常之前，重复调用该迭代器上的`__next__`方法，并将结果绑定到`<name>`，最后执行`<suite>`。
+ 下面我们利用`while`，`try`和`except`语句模拟`for`语句的运行过程：
  ```python
  >>> counts = [1, 2, 3]
  >>> items = counts.__iter__()
  >>> try:
          while True:
              item = items.__next__()
              print(item)
      except StopIteration:
          pass
  1
  2
  3
  >>> '''
      以上语句等价于：
      for item in counts:
              print(item)
      '''
  ```
+ 另一方面，`<expression>`里也可以为迭代器本身（这在面向对象编程里比较常见），所以这需要迭代器本身具有`__iter__`方法（返回其自身）。
## 生成器（Generators）和 Yield 语句
+ 生成器也是一种特殊的迭代器，其由生成函数得到。生成函数与普通函数的区别在于其没有`return`语句（即不返回特定值），取而代之的是`yield`语句。
+ 基于生成函数的生成器使用时，通过`next()`输出生成函数的`yield`语句内容，并转到下一个`yield`语句，直到迭代结束（没有新的`yield`语句）。具体示例如下：
  ```python
  >>> def letters_generator():
        current = 'a'
        while current <= 'd':
            yield current
            current = chr(ord(current) + 1)

  >>> for letter in letters_generator(): # for 循环相当于重复调用__next__
          print(letter)
  a
  b
  c
  d
  >>> '''
      也可以手动使用__next__：
      '''
  >>> letters = letters_generator()
  >>> type(letters)
  <class 'generator'>
  >>> letters.__next__()
  'a'
  >>> letters.__next__()
  'b'
  >>> letters.__next__()
  'c'
  >>> letters.__next__()
  'd'
  >>> letters.__next__()
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  StopIteration
  ```
+ 另外，python还提供一种可简化代码的关键字：`yield from`。具体而言，它等价于`for`循环重复的`yield`语句。其常用于递归生成函数中。示例：
  ```python
  >>> def countdown(k):
        if k > 0:
          yield k
          for x in countdown(k-1): # 这个for循环等价于：yield from countdown(k-1)
              yield x

  >>> t = countdown(3)
  >>> next(t)
  3
  >>> next(t)
  2
  >>> next(t)
  1
  ```
+ 下面我们再回顾一下经典的[整数分割](/2025/11/21/computer-science/CS61A/CS61A-Chapter-5/#%E7%A4%BA%E4%BE%8B%E6%95%B4%E6%95%B0%E6%8B%86%E5%88%86)例子。之前我们都是使用`return`返回值，现在我们可以通过`yield`函数代替`return`+输出，从而大幅简化代码：
  ```python
  def partitions(n,m):
    """
    Yield partitions.

    >>> for p in partitions(6,4):print(p)
    2 + 4
    1 + 1 + 4 
    3 + 3
    1 + 2 + 3
    1 + 1 + 1 + 3
    2 + 2 + 2
    1 + 1 + 2 + 2
    1 + 1 + 1 + 1 + 2
    1 + 1 + 1 + 1 + 1 + 1
    """
    if n > 0 and m > 0:
        if n == m:
            yield str(m)
        for p in partitions(n-m, m):
            yield p + '+' + str(m)
        yield from partitions(n, m-1)
  ```
  而且由于惰性运算，其输出一种分割方法的效率会比先前的列表方法更高（不需要计算所有分割方式即可输出）。
+ 更多关于`yield`语句的解释可参考[官方文档](https://docs.python.org/zh-cn/3/reference/expressions.html#yield-expressions)或[知乎](https://zhuanlan.zhihu.com/p/268605982)。
## 可迭代对象接口（Iterable Interface）
+ 从这里开始，我们将尝试手动实现python内置的`iter()`与`next()`函数，并将其用于类与对象中。
+ 如果一个由类实例化创建的对象使用`__iter__`方法会返回一个迭代器，那么它就可以被称为可迭代对象。可迭代对象和前面的可迭代值本质相同，都表示可以产生多个不同迭代器的数据集合。
+ 以下面这个迭代输出连续字母的类为例：
  ```python
  >>> class Letters:
        def __init__(self, start='a', end='e'):
            self.start = start
            self.end = end
        def __iter__(self):
            return LetterIter(self.start, self.end) # 这里LetterIter属于迭代器（具体实现见下）
  >>> '''具体使用例：'''
  >>> b_to_k = Letters('b', 'k')
  >>> first_iterator = b_to_k.__iter__()
  >>> next(first_iterator)
  'b'
  >>> next(first_iterator)
  'c'
  >>> second_iterator = iter(b_to_k)
  >>> second_iterator.__next__()
  'b'
  >>> first_iterator.__next__()
  'd'
  >>> first_iterator.__next__()
  'e'
  >>> second_iterator.__next__()
  'c'
  >>> second_iterator.__next__()
  'd'
  ```
  在上面这个例子中，`b_to_k`就是可迭代对象（可理解为连续字母的集合），而`first_iterator`与`second_iterator`是两个独立的迭代器。
+ python的一些内置函数也可以接收可迭代对象为参数并返回迭代器（如`map`等函数，具体使用和上面的可迭代值基本相同）
## 迭代器接口（Iterator Interface）
+ 和上面的`__iter__`方法类似，我们也可以在类中实现`__next__`方法。其核心在于，在调用`__next__`方法时，迭代器会检索计算下一个元素，并将对应位置向后移位，直到到达序列末端时返回`StopIteration`。具体例子如下（与上面的示例相配套）：
  ```python
  >>> class LetterIter:
        """依照 ASCII 码值顺序迭代字符的迭代器。"""
        def __init__(self, start='a', end='e'):
            self.next_letter = start
            self.end = end
        def __next__(self):
            if self.next_letter == self.end:
                raise StopIteration
            letter = self.next_letter
            self.next_letter = chr(ord(letter)+1)
            return letter
  ```
  使用例：
  ```python
  >>> letter_iter = LetterIter()
  >>> letter_iter.__next__()
  'a'
  >>> letter_iter.__next__()
  'b'
  >>> next(letter_iter) # 和 __next__() 等价
  'c'
  >>> letter_iter.__next__()
  'd'
  >>> letter_iter.__next__()
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "<stdin>", line 12, in next
  StopIteration
  ```
  注意这种迭代是一次性的，所以在创建迭代器并遍历到序列末尾后，这个迭代器就无法再重新遍历使用，只能重新创建新的迭代器对象。
+ 当然，我们也可以设计不触发`StopIteration`的迭代器对象（即无限迭代），示例如下：
  ```python
  >>> class Positives:
        def __init__(self):
            self.next_positive = 1;
        def __next__(self):
            result = self.next_positive
            self.next_positive += 1
            return result
  >>> p = Positives()
  >>> next(p)
  1
  >>> next(p)
  2
  >>> next(p)
  3
  ```
## 使用 Yield 创建可迭代对象
+ 使用`yield`语句也可以实现可迭代对象接口，即在类中建立`__iter__`方法。比如下面这个例子：
  ```python
  >>> class LettersWithYield:
        def __init__(self, start='a', end='e'):
            self.start = start
            self.end = end
        def __iter__(self):
            next_letter = self.start
            while next_letter < self.end:
                yield next_letter
                next_letter = chr(ord(next_letter) + 1)
  ```
  `LettersWithYield`类在使用`__iter__`方法时，其都会返回一个新的生成器（也是迭代器）。每次调用的迭代器都会从序列开始向后遍历。使用例：
  ```python
  >>> letters = LettersWithYield()
  >>> list(all_pairs(letters))[:5]
  [('a', 'a'), ('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'a') ]
  ```
+ 这样就不需要在类中另外写`__next__`方法了，代码会更加简洁，运行效率也更高。
## Python 流
+ 放在后面结合链表类的相关内容再叙述（也是惰性计算的应用）。


> 拓展：[生成器与协程](https://zhuanlan.zhihu.com/p/267904590)