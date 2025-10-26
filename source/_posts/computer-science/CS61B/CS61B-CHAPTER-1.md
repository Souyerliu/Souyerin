---
title: CS61B CHAPTER 1
date: 2025-10-20 08:19:46
categories:
 - study
 - [计算机科学,CS61B]
 - [JAVA] 
tags: 
 - JAVA
cover: 61B-header.png
---
本系列笔记基于官网发布的slides，videos，textbook以及B站Lecture视频编写。
CS61B官网（2025 Fall）：[CS 61B Fall 2025](https://fa25.datastructur.es/)
CS61B Lecture（2024 Spring）：[CS61B Lecture sp24](https://www.bilibili.com/video/BV1hJ4m1M7ZA/)
注：本课程使用的编程语言为Java，但是否熟悉这个语言对学习此课程影响不大。（个人认为它相当于C++去掉C）
关于Java的安装配置，可参考：[Java开发环境配置教程](https://www.runoob.com/java/java-environment-setup.html)。!!如果你和我一样使用的是vscode，并且电脑上已经安装了java版minecraft，那么只需要在vscode中安装对应的java插件就行（）!!
但为了更好理解本课程，下面给出一些Java的基本语法。!!其实笔者也是第一次学QAQ!!{.bulr}
# Getting started - Introduction of Java
从经典的`Hello world`开始：
```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello"+" "+"world!");
        //注意java输出多个内容时中间使用+而不是,
    }
}
```
可以看出，和python相比，Java的语法繁杂很多，但和C/C++语法比较类似。
+ 一些基本语法：
  1. Java的所有语句都必须放在 **类(Classes)** 中。
  2. 需要运行的代码必须放在`public static void main(String[] args)`里（之后会解释具体的含义）
  3. 代码块必须用`{}`包裹，每个语句必须以`;`结尾。
## Java工作流
+ 运行`.java`文件通常经过两个过程：编译(compilation)和解释(interpretation)。
+ 以上面的`HelloWorld.java`文件（java文件名必须和`public class`名称相同）为例，当我们使用命令行操作文件时，先输入：
```shell
javac HelloWorld.java
```
将java文件编译为`.class`文件（如果我们尝试查看文件内容，会发现一堆乱码）。接下来，我们再执行：
```shell
java HelloWorld
```
程序就会开始执行。
使用`.class`文件有若干优点，包括更安全的代码共享，更高效的运行效率，保护编写者的知识产权等。但这里不会详细涉及。
## 更多的Java语法
### 变量
+ 以一段输出0到9之间的整数为例：
```java
public class HelloNumbers {
    public static void main(String[] args) {
        int x = 0;
        while (x < 10) {
            System.out.print(x + " "); 
            //print和println的区别在于println输出后自动换行，而print不换行
            x = x + 1;
        }
    }
}
```
+ 可以发现java定义变量的语法和C/C++类似，需要提前声明并确定变量类型，且不能随意更改变量类型。（这种程序语言被称为 **静态类型语言statically typed language** ）
+ 当然，java还有一个独特的特性。当我们尝试组合输出不同类型的数据时（如下例）：
```java
public class HelloWorld {
    public static void main(String []args) {
        System.out.println(114514+"homo");
    }
}
```
程序会顺利输出`114514homo`。这是C/C++和python都做不到的。这是因为java的编译器在识别到`+`的一边为字符串后，会自动将另一边转换为字符串并合并。
### 函数
+ java定义函数也与C/C++类似，需要声明参数的数据类型，以及返回的数据类型。例子如下：
```java
public class LargerDemo {
    public static int larger(int x, int y) {
        if (x > y) {
            return x;
        }
        return y;
    }

    public static void main(String[] args) {
        System.out.println(larger(8, 10));
    }
}
```
+ 注意所有函数都应该放在`public static`中（这类似于python中的`def`），可以与`public static void main(String[] args){}`并列。（`main`本身也是一个函数）另外，函数只能返回一个值。
+ 当然定义函数还有其他方式，之后也会提到。
### 代码风格与注释
+ 和其他语言一样，在编写程序时我们应尽量提高程序的可读性。
+ 关于代码的编写风格，这因人而异，但核心点在于能让他人轻松读懂你的代码。
+ 注释：和C/C++一样，java使用`//`进行单行注释，`/* */`进行多行注释。
+ 另外，在CS61A课程中，我们提到python支持在函数内编写文档字符串(docstring)。事实上，java也支持类似的操作，只需要在函数体开头加入`/** */`（这被称为javadoc），在其中就可以编写函数的功能描述（尽量控制在一行）。javadoc还支持加入标签(tags)，在这里不详细讨论。

以上便是java的一些基础介绍。更多java用法可见[java菜鸟教程](https://www.runoob.com/java/java-tutorial.html)。
# 关于CS61B
+ 与CS61A不同，本课程着重于高效地编写能够高效运行的程序，包括熟练掌握一些好的算法与数据结构，以及能够完整地设计、编写并调试大型程序，同时能使用包括但不限于git，IntelliJ IDEA（或其他编程环境），断言测试（如JUnit,Truth），大模型等编程工具。
+ 同时，本课程假定你已经掌握了一些编程的基础知识，包括：基于对象的编程，递归，列表，树等。
+ 本课程主要分为三大部分：
  + Java和数据结构基础
  + 数据结构（和一部分软件工程）
  + 算法与软件工程
+ 另外，本课程非常注重实践，所以会有大量的作业，实验与项目。（之后可能会考虑放一些自己做的在博客里）
+ 