---
title: CS61A Chapter 7
date: 2026-02-10 16:54:23
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
创建模块化程序时，除了使用数据抽象的方法，给数据增加随时间改变的状态也可以使得特定的对象在程序中独立出来。这是 **面向对象编程（object-oriented programming）** 中的一大核心。
## 对象（Objects）
+ 对象可以理解为函数与数据的结合，将函数看作数据的行为，可以被调用计算，也可以被直接操作。对象将信息与处理绑定在一起，这样就可以以代表复杂事物的属性、交互和行为。
+ 下面介绍一些概念：
  1. **类（class）**：可以理解为一种值的类型，传入参数进行 **实例化（instantiation）** 后就能得到对象（或者称为**实例（instances）**）；
  2. **属性（attributes）**：对象的某个值的名称（可参照C语言结构体的成员变量），可通过`<对象名>.<属性名>`进行访问；
  3. **方法（methods）**：也属于属性，但其本质是函数，通过对象及传入的参数计算返回结果。方法其实不止存在于我们自己定义的类与对象中，python自带的许多处理数字/字符串/列表等的函数同样属于定义好的方法。可以说，python中的所有值都是对象。
## 序列对象（Sequence Objects）
+ 上面提到列表本身也属于一种对象，因而可以通过调用其自身的方法使其数据发生变化。下面举一个具体的例子：
+ 假设有一个列表，记录了(不同国家/地区)扑克牌的花色。我们尝试对列表进行修改：
```python
>>> chinese = ['coin', 'string', 'myriad']  # 字符串列表
>>> suits = chinese                         # 为同一个列表指定了两个不同的变量名
>>> suits.pop()             # 从列表中移除并返回最后一个元素
'myriad'
>>> suits.remove('string')  # 从列表中移除第一个与参数相同的元素（此时列表只剩下coin一个元素）
>>> suits.append('cup')              # 在列表最后插入一个元素
>>> suits.extend(['sword', 'club'])  # 将另外一个列表中的所有元素添加到当前列表最后
>>> suits[2] = 'spade'  # 替换某个元素
>>> suits
['coin', 'cup', 'spade', 'club']
>>> suits[0:2] = ['heart', 'diamond']  # 替换一组数据
>>> suits
['heart', 'diamond', 'spade', 'club']
```
+ 当然，列表的方法还不止这些。但所有这些方法都是在原始列表上进行修改，而不是创建了新的对象。我们可以再重新查看`chinese`的值：
  ```python
  >>> chinese  # 这个变量名与 "suits" 指向的是同一个列表对象
  ['heart', 'diamond', 'spade', 'club']
  ```
  这就是**数据共享（Sharing）**。这与之前我们对变量进行修改完全不同。
+ 当然，如果要将列表数据复制到一个新列表，可以使用`list()`函数。这样，对两个列表的方法调用就互不影响了。
+ 另一种特殊情况是嵌套列表。如果一个列表中嵌套有另一个列表，那么对子列表数据的修改也会影响外部列表。比如下面这个例子：
  ```python
  >>> nest = list(suits)  
  >>> nest[0] = suits     # 创建一个嵌套列表，列表第一项是另一个列表
  >>> suits.insert(2, 'Joker')  # 在下标为 2 的位置插入一条新元素，其余元素相应向后移动
  >>> nest
  [['heart', 'diamond', 'Joker', 'spade', 'club'], 'diamond', 'spade', 'club']
  >>> nest[0].pop(2)
  'Joker'
  >>> suits
  ['heart', 'diamond', 'spade', 'club']
  ```
  那么我们如何区别数据完全相同的两个对象呢？在python中，我们可以用`is`
和`is not`进行分辨：
```python
>>> suits is nest[0]
True
>>> suits is ['heart', 'diamond', 'spade', 'club']
False
>>> suits == ['heart', 'diamond', 'spade', 'club'] # == 与 is 的区别
True
```
+ 如果对列表对象使用列表推导式，那么得到的新列表与原列表独立，不会影响原列表的数据：
```python
>>> from unicodedata import lookup
>>> [lookup('WHITE ' + s.upper() + ' SUIT') for s in suits]
['♡', '♢', '♤', '♧']
```
### 元组（Tuple）
+ 与可变序列对象相对的，元组就是python内置的一种不可变的实例化对象。它的形式为：`(a,b,c,...)`（括号可以不加），其他功能和列表几乎一样，见下例：
```python
>>> code = ("up", "up", "down", "down") + ("left", "right") * 2
>>> len(code)
8
>>> code[3]
'down'
>>> code.count("down")
2
>>> code.index("left")
4
```
+ 如果元组没有元素（空元组），或者只有一个元素，表示如下：
```python
>>> ()    # 0 elements
()
>>> (10,) # 1 element
(10,)
```
+ 当然，列表对象中对列表元素修改的操作元组中就没有了，但如果元组元素中有列表，那还是可以对子列表进行操作的。
+ 实际上，python中的批量赋值也隐性使用了元组。
## 字典（Dictionary）
+ 字典是python另外一种内置的数据类型。其由若干个 **键值对（key-value pairs）** 组成（可以理解为序列中的索引被替换为了键），一般键由字符串构成。
+ 字典的所有修改操作均基于键进行（注：关于字典的顺序，在目前的python版本中，字典会维持键插入的顺序，即新插入的关键字总是会在字典末尾）
+ 如果要直接访问字典的`value`，可以通过`<dictionary>.values()`返回得到由所有值组成的列表。（类似的还有`.keys()`和`.items()`，后者返回的是键值对元组组成的列表）
+ 类似`list()`，我们也可以使用`dict()`将键值对构成的列表转换为字典：
```python
>>> dict([(3, 9), (4, 16), (5, 25)])
{3: 9, 4: 16, 5: 25}
```
+ 字典存在两个限制：
  1. 字典中已有的`key`值不能改变，同时也不能包含可变数据。
  2. 在字典中，一个`key`只能对应一个`value`（当然`value`可以是一个包含多个元素的列表）。
+ 类似列表推导式，字典也有其推导式用法（前面应该已经出现过）
+ 关于列表与字典，还有一个值得注意的特性：当函数中使用这两个对象作为参数时，如果函数有多次调用，那么其参数的数据会共享（即在同一个列表/字典上操作），比如：
  ```python
  def f(s=[]): # 只会在第一次调用时执行赋值
    s.append(3)
    return len(s)

  f() # 1
  f() # 2
  f() # 3
  ```
## 局部状态（Local State）
+ 上面提到的列表、字典，以及函数都有局部状态。局部状态可以理解为在一段封闭的程序部分中数据（状态）会随着时间而改变。
+ 对于函数（为了和之前的那些函数区别，也被称为**非纯函数in-pure function**）而言，它的局部状态可以表现为：当接收相同的参数时，函数可能会返回不同的值。具体而言，函数在调用后除了计算并返回内容之外，还会对其栈帧之外的变量进行修改。
+ 一个具体的例子：
  ```python
  >>> def make_withdraw(balance):
        """返回一个每次调用都会减少 balance 的 withdraw 函数"""
        def withdraw(amount):
            nonlocal balance                 # 声明 balance 是非局部的
            if amount > balance:
                return 'insufficient funds'
            balance = balance - amount       # 重新绑定
            return balance
        return withdraw
  ```
+ 这里的`balance`通过`nonlocal`被声明为非局部变量，即它只会位于上一级被声明时的栈帧，后续局部帧对变量的修改都是在原有变量值的基础上进行变化。（注：如果在`nonlocal`声明前变量没有赋值定义，那么就会报错）
+ 这种`nonlocal`声明方法常用于像上面这种的嵌套函数中。当然，引入这种非局部变量就需要与局部变量进行明确的区分。
+ 注：当然这里也可以绕过使用`nonlocal`变量的方式，比如改为列表：
  ```python
  def MAKE_WITHDRAW_LIST(BALANCE):
      b = [balance]
      def withdraw(amount):
          if amount > balance:
              return 'insufficient funds'
          b[0] = b[0] - amount  
          return b[0]
      return withdraw
  ```
+ 当然，这种非局部声明方法也是python的一个特性。在其他语言中（如C语言），变量的定义往往不需要特别进行声明，其位置已经决定了它的局部性。在python中，下面这样的代码会产生报错（无法直接在局部帧中使用非局部帧的变量值）：
  ```python
  def make_withdraw(balance):
    def withdraw(amount):
        if amount > balance:
            return 'Insufficient funds'
        balance = balance - amount
        return balance
    return withdraw

  wd = make_withdraw(20)
  wd(5)
  # UnboundLocalError: local variable 'balance' referenced before assignment
  ```
  这主要与python解释器会提前考虑内部函数的变量状态有关。
### 非局部（Non-local）赋值的好处
+ 我们以下面这个例子说明：
  ```python
  def make_withdraw(balance):
    def withdraw(amount):
        nonlocal balance
        if amount > balance:
            return 'Insufficient funds'
        balance = balance - amount
        return balance
    return withdraw

  wd = make_withdraw(20)
  wd2 = make_withdraw(7)
  wd2(6) # 1
  wd(8) # 12
  ```
+ 可以看到，程序定义了两个`make_withdraw`框架，分别对应`wd`和`wd2`。这两个框架定义的`balance`相互独立，互不影响，只会受各自函数的调用而变化。
+ 也就是说，我们可以利用非局部赋值设计一个独立的函数框架，其中的变量只会受自身函数调用的控制，而其他任何函数都无法改变其状态。
### 非局部（Non-local）赋值的代价
+ 非局部赋值的引入会让名称与值的关系产生微妙的变化。我们再引入一个例子作为比较：
  ```python
  def make_withdraw(balance):
    def withdraw(amount):
        nonlocal balance
        if amount > balance:
            return 'Insufficient funds'
        balance = balance - amount
        return balance
    return withdraw

  wd = make_withdraw(12)
  wd2 = wd
  wd2(1) # 11
  wd(1) # 10
  ```
  可以看到，这段程序中`wd`和`wd2`对应的都是同一个`balance`变量。由于只有函数调用时才会产生新的帧，而对`nonlocal`变量的赋值取决于其对应的函数帧，所以这里只有一个对应的`balance`变量。
+ 所以说，在`nonlocal`语境下，数据对象的相等就不一定是值的相等，而相同的对象即使值发生变化，对象本身仍然不变。
+ 虽然这会让程序的理解难度上升，但这也是编写模块化程序的重要工具。我们使用的内置列表以及字典都是基于这种设计实现的。
## 实现列表与字典
+ 在python中，我们无法访问列表/字典的实现基础（即底层架构），但我们会尝试使用具有局部状态的函数构建一个可变列表。
+ 下面我们会以[链表](/2026/02/04/computer-science/CS61A/CS61A-Chapter-6/#%E9%93%BE%E8%A1%A8linked-lists)的操作函数为基础，构建一个可变链表。可变链表的形式是一个函数，其内部会嵌套一个`dispatch`函数，用于接收操作名称（如`'len','getitem'`等）以及相关参数，并返回对应的结果。具体构造如下：
  ```python
  >>> def mutable_link():
        """返回一个可变链表的函数"""
        contents = empty
        def dispatch(message, value=None):
            nonlocal contents
            if message == 'len':
                return len_link(contents)
            elif message == 'getitem':
                return getitem_link(contents, value)
            elif message == 'push_first':
                contents = link(value, contents)
            elif message == 'pop_first':
                f = first(contents)
                contents = rest(contents)
                return f
            elif message == 'str':
                return join_link(contents, ", ")
        return dispatch
  ```
  然后我们还可以再构造一个函数，用于接收数据（列表）并构建整个链表：
  ```python
  >>> def to_mutable_link(source):
        """返回一个与原列表相同内容的函数列表"""
        s = mutable_link()
        for element in reversed(source):
            s('push_first', element)
        return s
  ```
+ 这样我们就成功实现了一个可变链表，示例如下：
  ```python
  >>> suits = ['heart', 'diamond', 'spade', 'club']
  >>> s = to_mutable_link(suits)
  >>> print(s('str'))
  heart, diamond, spade, club
  >>> s('pop_first')
  'heart'
  >>> print(s('str'))
  diamond, spade, club
  ```
+ 当然，我们还可以再此基础上再添加更多的功能（如指定位置插入、删除等）。其实现可以通过直接使用原有定义的功能，或者在`dispatch`函数中增加`elif`条件。后者也被称为**消息传递（Message passing）**，其基于`message`的信息执行对应的操作（局部函数）。
+ 类似地，我们也可以模拟字典的实现，示例如下（使用键值对列表存储内容）：
  ```python
  >>> def dictionary():
        """返回一个字典的函数实现"""
        records = []
        def getitem(key):
            matches = [r for r in records if r[0] == key]
            if len(matches) == 1:
                key, value = matches[0]
                return value
        def setitem(key, value):
            nonlocal records
            non_matches = [r for r in records if r[0] != key]
            records = non_matches + [[key, value]]
        def dispatch(message, key=None, value=None):
            if message == 'getitem':
                return getitem(key)
            elif message == 'setitem':
                setitem(key, value)
        return dispatch

  >>> d = dictionary()
  >>> d('setitem', 3, 9)
  >>> d('setitem', 4, 16)
  >>> d('getitem', 3)
  9
  >>> d('getitem', 4)
  16
  ```
  当然这里字典查找键值使用的是顺序查找，效率较低。python的内置字典使用了更高效的查找方式，这里不详细展开。
### 调度字典（Dispatch Dictionaries）
+ 上面的`dispatch`函数使用了条件语句将消息与对应的功能对应。这里我们也可以直接使用内置字典，将消息设为键，对应的变量为值，以实现对应操作。示例如下：
  ```python
  def account(initial_balance):
    def deposit(amount):
        dispatch['balance'] += amount
        return dispatch['balance']
    def withdraw(amount):
        if amount > dispatch['balance']:
            return 'Insufficient funds'
        dispatch['balance'] -= amount
        return dispatch['balance']
    dispatch = {'deposit':   deposit,
                'withdraw':  withdraw,
                'balance':   initial_balance}
    return dispatch

  def withdraw(account, amount):
      return account['withdraw'](amount) 
  def deposit(account, amount):
      return account['deposit'](amount)
  def check_balance(account):
      return account['balance']

  a = account(20) # a 定义为一个字典（对应dispatch/account）
  deposit(a, 5) # balance = 25
  withdraw(a, 17) # balance = 8
  check_balance(a) # 8
  ```
+ 注意到，上面定义的字典中，`'deposit'`和`'withdraw'`对应的值均为函数对象（对应`account`函数内定义的两个函数），所以在下面的两个函数中对其进行了调用。
+ 而且，这两个函数作为字典的值，可以直接使用字典中的`balance`值（不需要`nonlocal`声明）
## 约束传递 (Propagating Constraints)
+ 下面会介绍一个扩展的示例，其综合运用了非局部赋值、列表与字典。
+ 这个示例是一个支持多方向计算的约束系统。为了实现这一系统，我们没有使用传统程序的单项计算（命令式编程，即给定参数→执行操作→输出），而使用声明式编程（即建立若干约束条件与连接器→给定已知量→根据约束关系计算未知量）
+ 具体而言，本例实现的是摄氏度与华氏度的关系网络，使用以下约束关系：
  $$
  9\times c=5\times (f-32)
  $$
  我们将这一约束分解，可以得到由加法器、乘法器以及常量约束组成的网络（图源CS61A官方教材）：
  ![network](https://www.composingprograms.com/img/constraints.png)
  图中所有大方框都属于约束，而所有的连线均属于连接器。
+ 这个网络的计算过程如下：对于每一个连接器，当它的一端被给定一个值，那么它就会将值传到另一端，激活对应的约束框；而每个约束框在被激活后会检查其接收的连接器值是否足以确定未接收到的连接器值，如果能确定则将值通过该连接器传输，以此类推。
+ 下面是程序实现。我们先不考虑约束与连接器的实现细节，对网络的大框架进行构建：
  ```python
  >>> celsius = connector('Celsius')
  >>> fahrenheit = connector('Fahrenheit')
  >>> def converter(c, f):
        """用约束条件连接 c 到 f，将摄氏度转换为华氏度."""
        u, v, w, x, y = [connector() for _ in range(5)]
        multiplier(c, w, u) # 乘法器
        multiplier(v, x, u)
        adder(v, y, f) # 加法器
        constant(w, 9) # 常量
        constant(x, 5)
        constant(y, 32)
  >>> converter(celsius, fahrenheit)
  ```
  其中约束与连接器使用消息传递系统控制。约束与连接器均由字典实现，具体功能如下：
  ```python
  >>> connector ['set_val'](source, value)  """表示 source 在请求连接器将当前值设为 value"""
  >>> connector ['has_val']()  """返回连接器是否已经具有值"""
  >>> connector ['val']  """是连接器的当前值"""
  >>> connector ['forget'](source)  """告诉连接器 source 请求遗忘它的值"""
  >>> connector ['connect'](source)  """告诉连接器参与新的约束，即 source"""
  >>> constraint['new_val']()  """表示与约束相连的某个连接器具有新的值。"""
  >>> constraint['forget']()  """表示与约束相连的某个连接器遗忘了值。"""
  ```
+ 下面是约束与连接器的具体实现：
  1. 约束
  ```python
  >>> def make_ternary_constraint(a, b, c, ab, ca, cb):
        """通用约束 ab(a,b)=c，ca(c,a)=b，cb(c,b)=a"""
        def new_value():
            av, bv, cv = [connector['has_val']() for connector in (a, b, c)]
            if av and bv:
                c['set_val'](constraint, ab(a['val'], b['val']))
            elif av and cv:
                b['set_val'](constraint, ca(c['val'], a['val']))
            elif bv and cv:
                a['set_val'](constraint, cb(c['val'], b['val']))
        def forget_value():
            for connector in (a, b, c):
                connector['forget'](constraint)
        constraint = {'new_val': new_value, 'forget': forget_value}
        for connector in (a, b, c):
            connector['connect'](constraint)
        return constraint # 调度字典
  >>> from operator import add, sub
  >>> def adder(a, b, c):
        """加法器约束 a + b = c"""
        return make_ternary_constraint(a, b, c, add, sub, sub)
  >>> from operator import mul, truediv
  >>> def multiplier(a, b, c):
        """乘法器约束 a * b = c"""
        return make_ternary_constraint(a, b, c, mul, truediv, truediv)
  >>> def constant(connector, value):
        """常量赋值"""
        constraint = {}
        connector['set_val'](constraint, value)
        return constraint
  ```
  2. 连接器
  ```python
  >>> def inform_all_except(source, message, constraints):
        """告知信息除了 source 外的所有约束条件"""
        for c in constraints:
            if c != source:
                c[message]()
  >>> def connector(name=None):
        """限制条件之间的连接器"""
        informant = None
        constraints = []
        def set_value(source, value):
            nonlocal informant
            val = connector['val']
            if val is None:
                informant, connector['val'] = source, value
                if name is not None:
                    print(name, '=', value)
                inform_all_except(source, 'new_val', constraints)
            else:
                if val != value:
                    print('Contradiction detected:', val, 'vs', value)
        def forget_value(source):
            nonlocal informant
            if informant == source:
                informant, connector['val'] = None, None
                if name is not None:
                    print(name, 'is forgotten')
                inform_all_except(source, 'forget', constraints)
        connector = {'val': None,
                     'set_val': set_value,
                     'forget': forget_value,
                     'has_val': lambda: connector['val'] is not None,
                     'connect': lambda source: constraints.append(source)}
        return connector
  ```
+ 使用示例：
  ```python
  >>> celsius['set_val']('user', 25)
  Celsius = 25
  Fahrenheit = 77.0
  >>> fahrenheit['set_val']('user', 212)
  Contradiction detected: 77.0 vs 212
  >>> celsius['forget']('user')
  Celsius is forgotten
  Fahrenheit is forgotten
  >>> fahrenheit['set_val']('user', 212)
  Fahrenheit = 212
  Celsius = 100.0
  ```
+ 在这个约束程序中，我们引入了许多将在面向对象编程中再次出现的思想。
!!看不懂问题不大，多看几遍了解其思想即可，笔者也是一头雾水doge!!