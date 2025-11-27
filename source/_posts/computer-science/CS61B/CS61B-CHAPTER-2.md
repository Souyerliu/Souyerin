---
title: CS61B CHAPTER 2
date: 2025-11-24 19:19:43
categories:
 - study
 - [计算机科学,CS61B]
 - [JAVA] 
tags: 
 - JAVA
 - 数据结构
cover: 61B-header.png
---
# 定义与使用类(Defining and Using Classes)
## 静态方法(static methods)
+ 如前面所述，java语言中所有代码都必须放在类(class)中，而大部分代码都放在方法(method)中（可以理解为可执行一系列操作的函数）。而为了运行这些方法，我们需要对其进行调用。
+ 以下面这个`Dog`程序为例：
```java
public class Dog {
    public static void makeNoise() {
        System.out.println("Bark!");
    }
}
```
+ 单独编译这段代码没有问题，但运行这段代码时会报错：
  ```bash
  错误: 在类 Dog 中找不到 main 方法, 请将 main 方法定义为:
  public static void main(String[] args)
  ```
+ 在上一节中，我们提到可以设立一个`main`方法解决这一问题。当然，我们还可以采用另一种方法：在同一目录下创建另一个java文件，并在main方法中调用它：
```java
public class DogLauncher {
    public static void main(String[] args) {
        Dog.makeNoise();
    }
}
```
+ 像这样在调用其他类方法的类也被称为类的“使用者”(client)。（不像python，在同一目录下java不需要import就能直接调用类，这其实比较像matlab）

## 实例变量与对象实例化(Instance Variables and Object Instantiation)
+ 还是以上面这个`Dog`为例，当我们尝试拓展其功能（根据某些条件输出不同狗的叫声）时，一种最直接的方案是设置不同的类，但这样过于累赘。于是，我们会尝试使用变量与条件语句，代码如下：
```java
public class Dog {
    public int weightInPounds;//实例化变量

    public void makeNoise() {//注意这里没有static，因为使用了非静态变量weightInPounds（所以被称为非静态方法，也称为实例化方法）
        if (weightInPounds < 10) {
            System.out.println("yipyipyip!");
        } else if (weightInPounds < 30) {
            System.out.println("bark. bark.");
        } else {
            System.out.println("woof!");
        }
    }    
}
```
而在`DogLauncher.java`中调用时，形式如下：
```java
public class DogLauncher {
    public static void main(String[] args) {
        Dog d;
        d = new Dog();//创建一个新的对象
        d.weightInPounds = 20;
        d.makeNoise();
    }
}
```
运行时就会输出`bark. bark.`。不过，一般我们不在类的使用者中设置变量值（因为多次调用赋值会产生大量冗余代码），而在类中建立一个“构造器”(Constructor)对类的变量进行赋值：
```java
public class Dog {
    public int weightInPounds;

    public Dog(int w) {//构造器Constructor
        weightInPounds = w;
    }

    public void makeNoise() {
        if (weightInPounds < 10) {
            System.out.println("yipyipyip!");
        } else if (weightInPounds < 30) {
            System.out.println("bark. bark.");
        } else {
            System.out.println("woof!");
        }    
    }
}
```
这样在调用类时，只需要传入对应的参数即可：
```java
public class DogLauncher {
    public static void main(String[] args) {
        Dog d = new Dog(20);
        d.makeNoise();
    }
}
```
+ 事实上，java里的Constructor和python类中的`__init__`非常类似。
+ 下面我们再梳理一下类与对象的关系：    
  在`Dog.java`中，我们创建了一个名为`Dog`的类，它规定了基于此创建的对象的蓝图（规则与功能）。而在`DogLauncher.java`中，我们创建了一个名为`d`的对象，根据类的蓝图，在创建时传入参数，并可以根据需要调用类的方法或设置已有的变量（但是不能在类的使用者中设置新的类变量）。
+ 当然，我们也可以创建一系列对象，利用数组进行保存：
```java
public class DogArrayDemo {
    public static void main(String[] args) {
        /* 创建包含两个Dog对象的数组. */
        Dog[] dogs = new Dog[2];
        dogs[0] = new Dog(8);
        dogs[1] = new Dog(20);

        dogs[0].makeNoise();//会输出yipyipyip!
    }
}
```
## 类方法(Class methods) vs 实例方法(Instance methods)
+ 在java中，方法一般分为下面两种：
  1. 类方法，又称为静态方法(Static methods)；
  2. 实例方法，又称为非静态方法(non-static methods)。
+ 这两类方法的根本区别在于：实例方法作用于具体的对象，而类方法作用于所有具有相同类的对象（作为一个整体）
+ 在具体调用上，类方法不需要定义对象，而直接可以通过`类名.方法(参数)`的方式进行调用；与之相对的，实例方法则需要先定义一个对象（如`类名 对象名=new 类名(参数)`），再调用方法（`对象名.方法`）
+ 另一方面，类方法不能直接调用实例变量（除非参数为对象），而实例方法可以。
+ 下面还是通过`Dog`的例子进行演示：
  + 我们先用静态方法构建一个比较狗的重量的函数：
    ```java
    public static Dog maxDog(Dog d1, Dog d2) {
        if (d1.weightInPounds > d2.weightInPounds) {
            return d1;
        }
        return d2;
    }
    ```
  + 由于使用了静态方法，所以调用时直接通过类名调用：
    ```java
    Dog d = new Dog(15);
    Dog d2 = new Dog(100);
    Dog.maxDog(d, d2);
    ```
  + 当然，我们也可以使用非静态方法：
    ```java
    public Dog maxDog(Dog d2) {
        if (this.weightInPounds > d2.weightInPounds) {//这里this.可以省略
            return this;//返回调用这个方法的对象
        }
        return d2;
    }
    ```
  + 相应地，调用时就需要使用一个具体的对象：
    ```java
    Dog d = new Dog(15);
    Dog d2 = new Dog(100);
    d.maxDog(d2);
    ```
  + 可以看出，两种方法的含义略有不同：静态方法是以一个旁观者的视角比较两只狗的重量，而非静态方法以其中一只狗的重量为基准，和另一只狗进行比较。
## 静态变量(Static variables)
+ 当然，与实例变量相对的，在类中也可以定义静态变量。示例如下：
  ```java
  public class Dog {
    public int weightInPounds;
    public static String binomen = "Canis familiaris";
    ...
  }
  ```
+ 与静态方法类似，定义静态变量后，调用变量需要以`类名.变量名`的形式，而不能通过对象名调用变量。另外，静态变量的值不能改变，因此**谨慎使用**（可以将其理解为一个常量，类似C++中的`const`）。
## 关于主函数
+ 利用前面的概念，我们可以对主函数`public static void main(String[] args)`进行分析：
  + `public`：目前我们写的所有函数方法都带有这个关键字；
  + `static`：主函数属于静态方法，不依赖任何实例；
  + `void`：主函数没有返回值；
  + `main`：主函数名称；
  + `String[] args`：传入主函数的参数。（通过命令行输入获取）
