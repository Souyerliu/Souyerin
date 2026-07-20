---
title: CS61B CHAPTER 3
date: 2026-02-05 12:19:59
categories: [计算机科学, CS61B]
tags:
  - JAVA
  - 数据结构
cover: ./61B-header.png
---

# Primitive and Reference Types（原始与引用类型）
本节将会介绍一些基本的数据类型及其使用方式，并围绕列表这一核心进行阐释。

## 原始数据类型（Primitive Types）
- 首先简单提一下变量数据在计算机中的存储方式：无论是数字还是文本，在计算机中均以二进制列进行存储，而程序中的变量类型则指定了二进制列的解释方式。
- 在Java中，定义一个变量需要指定其变量类型，这相当于在内存中分配了一段存储变量的二进制空间（当然，我们无法获取到具体的存储地址，这点和C语言不同）。而对变量进行赋值就相当于在这个空间中存储这个值。
- 上面提到的变量类型可以是：`byte, short, int, long, float, double, boolean, char`，这些也称为原始数据类型。
+ **等号黄金法则（The Golden Rule of Equals）**：对于原始数据类型定义的变量，在进行等号赋值操作时（如`y = x`），其本质是将等号右边的变量对应的值（二进制位）复制到等号左边的变量空间中。

## 引用数据类型（Reference Types）
- 与原始数据类型相对，其他的数据类型都可以称为引用数据类型（包括列表和类）
- 在对象被实例化后，其对应类的变量初始值被赋为`0`或`null`；
- 接着在创建对象时使用`new 对象名(属性1值,属性2值,...)`，相当于在内存空间中分配一个区域，存储其所有的属性值。但这个对象变量在内存中实际存储的并不是属性值本身，而是属性值存储的地址（比如使用64位二进制列存储）。
- 所以在引用数据类型变量上使用等号时，其复制的是属性值的地址，两个变量（对象）指向的是同一个属性值。具体的例子：
```java
public class Walrus {
    public int weight;
    public double tuskSize;

    public Walrus(int w, double ts) {
          weight = w;
          tuskSize = ts;
    }
    @Override
    public String toString() {
        return "Walrus{weight=" + weight + ", tuskSize=" + tuskSize + "}";
    }
    public static void main(String[] args) {
        Walrus a = new Walrus(1000, 8.3);
        Walrus b;
        b = a;
        b.weight = 5;
        System.out.println(a);
        System.out.println(b);
    }
}
```
上面这段代码的输出结果为：
```
Walrus{weight=5, tuskSize=8.3}
Walrus{weight=5, tuskSize=8.3}
```
即`b`和`a`指向的是同一个`Walrus`对象。

## 参数传递（Parameter passing）
- 有了前面的阐述，方法（函数）的参数传递也就容易理解了：参变量将传入变量的值复制给自身（和上面的等号规则一样），得以将变量值进行传递。

# Arrays and Lists（数组与列表）
## Java 内置数据结构
列表作为最基础且最常见的数据结构，在大多数编程语言中都已经内置，且附带一些基本用法，比如插入、删除、修改元素，元素-索引转换等。
- 在Java中，内置列表是通过类定义的，其构建方式如下（当然这是二十多年前早期版本的写法了，后面会介绍更加现代的写法）：
  ```java
  import java.util.List;
  import java.util.ArrayList;
  public class ListDemo {
      public static void main(String[] args) {
          List L = new ArrayList();
          L.add("a");
          L.add("b");
          L.add("c");
          System.out.println(L.get(0)); //输出 a
      }
  }
  ```
  注意上面定义语句左边`List`与右边`ArrayList`的差异：
  - `ArrayList`是`List`的子类，其他子类还包括`LinkedList`，`CopyOnWriteArrayList`等，具体可参见[官方文档](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/util/List.html)。不同种类的列表各自具有独特的优势，这会在之后详细展开。
  - 然而，上述方法却存在一个缺陷：无法直接将列表中的元素赋值到新变量。【需要强制类型转换】
- 为解决这一问题，在更新的版本中，定义列表时使用尖括号明确列表元素的类型，比如：
  ```java
  import java.util.List;
      import java.util.ArrayList;
      public class ListDemo {
          public static void main(String[] args) {
              List<String> L = new ArrayList<>();
              L.add("a");
              L.add("b");
              String x = L.get(0);
          }
      }
  ```
  > 冷知识：直到2011年前，上面定义数组列表时右边都必须写成`new ArrayList<String>()`形式。
  - 由此可见，现代列表中元素的类型必须保持一致。
- 在Java中，数组也属于一种特殊的列表，其在定义时就确定了大小（无法改变），且没有内置方法，使用上与python numpy数组比较类似。
  - 相比列表，数组的读写效率更高，使用也更频繁。（Java语言追求性能至上）
- 除此之外，Java还有另一种类似python字典的数据结构——映射（Map）。示例如下：
  ```java
  import java.util.Map;
  import java.util.TreeMap;
  public class MapDemo {
      public static void main(String[] args) {
          Map<String, String> L = new TreeMap<>();
          L.put("dog", "woof");
          L.put("cat", "meow");
          String sound = L.get("cat");
      }
  }
  ```
  与列表类似，`Map`也有多个子类，如`TreeMap`，`HashMap`等。（这些应该也会在之后涉及）
  > 类似地，Java也有类似python集合的数据结构，也叫集合（Set），同样有`TreeSet`和`HashSet`。

## 构建列表
- 下面我们会从零开始构建一个类似python的列表。
### 数组的声明与实例化
- 数组也属于对象，所以定义数组时也会使用`new`，比如：
```java
Planet p = new Planet(0, 0, 0, "blah.png");
int[] x = new int[]{1，1, 4, 5, 1, 4}; // 当然这也等价于int[] x = {1, 1, 4, 5, 1, 4};
```
上面这两行代码都同时包含了声明（等号左边）、实例化（等号右边）以及赋值（等号）。注意，这里`p`和`x`存储的都是数组对应的地址，而非数组内容本身。
- 另外，在比较两个数组是否相等时，不能直接使用`==`，因为即使两个数组内容完全相同，它们的地址也不一样。所以比较两个数组需要使用`Arrays.equals(x,y)`。

### 初始化列表（IntLists）
- 我们在[CS61A](https://souyerin.netlify.app/posts/computer-science/cs61a/cs61a-chapter-6/#链表linked-lists)里已经提到了链表这一结构。我们可以尝试在Java中也构建这样的列表。
- 一个比较直接的想法如下：
    ```java
    public class IntList {
        public int first;
        public IntList rest;
        public static void main(string[] args) {
            IntList L = new IntList(5, null);
            L.rest = new IntList(10, null);
            L.rest.rest = new IntList(15, null);
        }
    }
    ```
    这样的链表实现比较丑陋，不够简洁。我们也可以尝试反向构建链表（或者说前插法构建）：
    ```java
    public class IntList {
        public int first;
        public IntList rest;
        public IntList(int f, IntList r) { //构造器，等价于插入函数
            first = f;
            rest = r;
        }
        public static void main(String[] args) {
            IntList L = new IntList(15, null);
            L = new IntList(10, L);
            L = new IntList(5, L);
        }
    }
    ```

### 求链表的大小（Size）
- 那么如何获取链表的大小呢？我们需要定义一个获取链表长度的方法`L.size()`。递归或者迭代两种方法都可以。下面是具体的演示：
    ```java
    /** 使用递归法返回链表长度 */
    public int size() {
        if (rest == null) {
            return 1;
        }
        return 1 + this.rest.size();
    }

    /** 使用迭代法返回链表长度 */
    public int iterativeSize() {
        IntList p = this;
        int totalSize = 0;
        while (p != null) {
            totalSize += 1;
            p = p.rest;
        }
        return totalSize;
    }
    ```
### 求链表具体位置的元素
- 类似地，我们也需要编写获取链表指定元素的方法：
    ```java
    /** 获取链表的第i个元素（递归） */
    public int get(int i) {
        if (i == 0) {
            return first;
        }
        return rest.get(i-1);
    }
    ```
- 这种查找方法需要$O(n)$级别的时间，当链表长度很大的时候效率会很低。之后我们会讨论如何进行优化。

> 最后补充一个课程使用的类似CS61A的代码运行可视化工具：[java visualizer](https://cscircles.cemc.uwaterloo.ca/java_visualize/#)