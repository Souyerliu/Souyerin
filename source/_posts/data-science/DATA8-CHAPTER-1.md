---
title: DATA8 CHAPTER 1
date: 2026-02-14 19:54:53
categories:
 - study
 - [数据科学,DATA8] 
tags: 
 - data science
 - jupyter notebook
cover: valentine_logo.png
---
本系列笔记参考：
+ Lecture：[sp23](https://www.bilibili.com/video/BV19eqHYSEDG)
+ 公开资料仓库：[materials](https://github.com/data-8/materials-fds)
+ 官方教材：[Computational and Inferential Thinking](https://inferentialthinking.com/)
+ DATA 8大致对应国内的数据科学导论（或类似课程），内容比较基础，主要讲述一些基本的数据处理与可视化方法，为后续进阶课程（如DATA 100,DATA 140）做铺垫。
+ 本课程使用Python+Jupyter Notebook进行数据分析。安装Anaconda（或类似环境）即可满足要求。使用的库主要为python的[datascience](https://data8.org/datascience/)库（由UCB教授们开发）
# 介绍（Introduction）
+ 什么是数据科学？数据科学是关于通过**探索**、**预测**与**推断**的方式从大而多样的数据集中获得有用结论的学科。
  |方式|目的|使用工具|
  |:--:|:--:|:--:|
  |**探索（exploration）**|识别信息中的模式|数据可视化/描述性统计|
  |**预测（prediction）**|利用已知信息对未知数值进行合理推测|优化方法/机器学习|
  |**推断（inference）**|量化确定程度（在数据中发现的模式是否也会在新观测中重现？预测准确度有多高？）|统计检验/统计模型|
+ 数据科学综合汲取了统计学以及计算机科学的精华，并将其应用与具体的领域中。统计学主要负责运用随机性补全缺失的信息，而计算机科学（编程）则用于处理数据（尤其是大规模数据）并得出有效结论。
## 数据科学中的统计方法
+ 统计学提供了一些连贯且精确的词汇以描述观察结果与结论间的关系。
+ 数据分析中的核心推断问题（假设检验、置信度估计以及未知量预测等）同样源于统计学。
+ 数据科学则在统计学基础上引入计算、数据可视化、机器学习、优化和信息获取等方法；同时加入计算机与网络技术，使得其能够获取并分析庞大的数据集（还可以利用重采样（Resampling）等技术使得数据集满足特定研究要求）。
## 一个简单的数据可视化例子
略，直接上[链接](https://inferentialthinking.com/chapters/01/3/plotting-the-classics/)。（其实数据可视化的例子网上数不胜数，这里只是给一个大概的印象）