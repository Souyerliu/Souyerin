---
title: Kaggle SQL进阶 学习笔记
date: 2025-10-18 16:57:33
categories:
 - [计算机科学,Kaggle-sql-tutorial]
tags:
 - SQL
 - BigQuery
cover: cover.png
---
注：强烈建议看完[SQL入门笔记](/2025/10/15/Kaggle%20SQL入门%20学习笔记/)后再阅读此笔记！
# JOINs and UNIONs
+ 在[入门笔记](/2025/10/15/Kaggle%20SQL%E5%85%A5%E9%97%A8%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/#join)中，我们使用了`INNER JOIN`合并两个表格。这一节我们会使用更多的`JOIN`语句，以及使用`UNIONs`获取多个表格的信息。
+ 这里我们使用以下表格作为示例（若无特别说明，以下图片均源自Kaggle教程）：
![origin2](origin2.png)
和入门笔记中使用的表格略有不同，我们在两个表格中各自设置了一个无匹配对象的数据。
## JOINs
+ 对于上面的表格，如果使用`INNER JOIN`就只会保留两个表格的前4条数据。代码如下：
```python
query="""
      SELECT p.Name AS Pet_Name, o.Name AS Owner_Name
      FROM `bigquery-public-data.pet_records.owners` AS o
      INNER JOIN `bigquery-public-data.pet_records.pets` AS p
      ON p.ID =o.Pet_ID
      """
```
+ 如果我们想加上各自的最后一条数据，可以使用下列语句替换`INNER JOIN`：
  1. `LEFT JOIN`：只加上`JOIN`语句前面表格的无匹配数据，在这里就是`owners`表格的最后一条数据；
  2. `RIGHT JOIN`：只加上`JOIN`语句后面表格的无匹配数据，在这里就是`pets`表格的最后一条数据；
  3. `FULL JOIN`：两个表格的无匹配数据都加上。
+ 三种语句比较图如下：
![joins](joins.png)
+ 注：对于加入后的无匹配数据，其缺失的匹配数据会被设置为`NULL`。
+ 当然，多个`JOINs`语句也适用于多个表格的合并，数据匹配规则类似链表结构。
## UNIONs
+ 与`JOINs`的水平合并表格不同，`UNIONs`语句是将不同表格的相同类型列纵向合并为一列。
+ 使用例：
```python
query="""
    SELECT Age FROM `bigquery-public-data.pet_records.pets`
    UNION ALL
    SELECT Age FROM `bigquery-public-data.pet_records.owners`
    """
``` 
输出结果：
|Age |
|:--:|
|20  |
|45  |
|10  |
|9   |
|8   |
|1   |
|9   |
|7   |
|2   |
|10  |
+ 这里`UNION ALL`代表允许数据中有重复值。如果要去掉重复值，则使用`UNION DISTINCT`。
# Analytic functions
+ 在入门笔记中，已经涉及到了一些聚合函数(aggregate functions)，这一节我们将接触一些分析函数(analytic functions)。
+ 分析函数(analytic functions)与聚合函数(aggregate functions)的一大区别是，分析函数对每一行数据都会返回值，而聚合函数对同一组数据只返回一个值。
## Syntax(句法)
+ 我们直接通过例子进行叙述：
+ 已知现有两名运动员的训练数据如下：
![training](training.png)
其中`id`表示运动员编号，`date`表示训练日期，`time`表示训练时长(单位：分钟)
+ 我们尝试动态地计算每位运动员的平均训练时长，编写query如下：
```python
query="""
      SELECT *,
      AVG(time) OVER(
                    PARTITION BY id
                    ORDER BY date
                    ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
                    ） as avg_time
      FROM `bigquery-public-data.runners.train_time`
      """
```
可以看到，在分析函数`AVG()`之后，有一段`OVER()`语句。`OVER()`有三个可选参数：
+ `PARTITION BY`：类似`GROUP BY`，将数据按指定列分组进行计算（和`GROUP BY`区别：`PARTITION BY`不压缩数据）
+ `ORDER BY`：将数据按指定列排序
+ `ROWS BETWEEN 1 PRECEDING AND CURRENT ROW`：这被称为 **窗口框架子句(window frame clause)**。它选定了需要计算的行范围（亦被称为“窗口”window）（实际上，分析函数也被称为窗口函数window functions）
  + 窗口框架子句的结构一般为`ROWS BETWEEN A AND B`，其中`A`和`B`可以为下列选项之一：
    + `UNBOUNDED PRECEDING`：当前行之前所有行；
    + `n PRECEDING`：当前行之前的n行；
    + `CURRENT ROW`：当前行；
    + `n FOLLOWING`：当前行之后的n行；  
    + `UNBOUNDED FOLLOWING`：当前行之后所有行。
## Three types of analytic functions（三种分析函数）
+ bigQuery支持的分析函数有很多，这里只列举三种，更多分析函数参见[window function calls](https://cloud.google.com/bigquery/docs/reference/standard-sql/window-function-calls)。
### Analytic aggregate functions（分析聚合函数）
+ 上面使用到的`AVG() OVER`就是一种典型的分析聚合函数。这种函数获取窗口内的所有数据并返回一个值。
+ 其他的分析聚合函数例子：`MIN()`,`SUM()`,`COUNT()`（实际上聚合函数基本都能转化为分析聚合函数）
### Analytic navigation function（分析导航函数）
+ 通常会返回窗口内来自其他行的值
+ 典型例子：`FIRST_VALUE()`（返回窗口排序后第一个值）,`LAST_VALUE()`（返回窗口排序后最后一个值）,`LEAD(xx,n)`(和`LAG(xx,n)`)（返回当前行的前(或后)n行的值）
### Analytic numbering functions（分析数值函数）
+ 会给每一行返回一个与其顺序有关的数
+ 典型例子：`ROW_NUMBER()`（返回该行在排序后的顺序编号，每行唯一），`RANK()`（返回该行在排序后的顺序编号，重复值编号相同，之后会跳号，类似运动会排名）
# Nested and Repeated Data
+ 之前我们处理的数据包括数值型，字符串型以及`DATATIME`型。这节我们将处理更加复杂的数据。
## Nested data（嵌套数据）
+ 有时我们在合并两个表格时，会选择将一个表格内的数据嵌套与另一个表格的一列中，类似下图：
![nested](nested.png)
+ 此时这一列的数据类型就被认定为`STRUCT`(或`RECORD`)，这可以通过输出`table.schema`确定。
+ 如果要在query中访问这列数据，可以使用类似下面的形式：
```python
query="""
      SELECT Name AS Pet_Name,
      Toy.Name AS Toy_Name,
      Toy.Type AS Toy_Type
      FROM `bigquery-public-data.pet_records.pets_and_toys`
      """
```
+ 其他操作就和正常的数据没有什么区别了。
## Repeated Data
+ 有时在表格的一格内有多个数据，如下图：
![repeated](repeated.png)
+ 这些数据对应列的数据类型被认定为`REPEATED`，而一格内的数据被称为 **数组(array)**。同一个数组内的所有数据类型都相同。
+ 如果要在query中访问这列数据，我们需要用到`UNNEST()`函数，示例如下：
```python
query="""
      SELECT Name AS Pet_Name,
      Toy_Type
      FROM `bigquery-public-data.pet_records.pets_and_toys_type`,
      UNNEST(Toys) AS Toy_Type
      """
```
注意`UNNEST()`函数紧接在`FROM`语句之后。得到的结果类似如下：
![unnest](unnest.png)
这相当于将所有聚合的数据进行平铺。
## nested and repeated data
+ 如果上面两种数据同时存在在一列数据中，如下图所示：
![nested and repeated](nested_and_repeated.png)
那么这一列的数据类型就同时为`RECORD`和`REPEATED`。
+ 那如何在query中访问这些数据呢？很简单——将上面两个操作合并即可。示例如下：
```python
query="""
      SELECT Name AS Pet_Name,
      t.Name AS Toy_Name,
      t.Type AS Toy_Type
      FROM `bigquery-public-data.pet_records.more_pets_and_toys`,
      UNNEST(Toys) AS t
      """
```
具体而言，就是先用`UNNEST()`将数据平铺，并重命名为新的表格，再用`.`进行嵌套数据的访问。
+ !!以C语言的数据结构理解，nested data就是结构体，repeated data就是数组，那么nested and repeated data就是结构体数组!!
# Writing Efficient Queries
+ 有时，我们需要大量地重复调用query，或者用query获取规模较大的数据集。此时query的处理效率就比较重要了。
+ 因此，这一节我们主要叙述一些query的优化。我们使用的优化指标包括：
  + query查询的数据量大小；
  + query查询的时间。
## Strategies
1. 只选择需要的数据列
   谨慎使用`SELECT * FROM`。这看似很省心，但可能会读取到大量不需要的数据列。
   所以每次query查询前，先明确自己需要查询那些数据列。
2. 读取更少的数据
   有时一些列虽然数据类型不同，但一一对应，在一些条件判断时相互等价。此时就可以不用读取一些不必要的数据列，节省读取数据量。
   例子：
   ```python
    more_data_query = """
                    SELECT MIN(start_station_name) AS start_station_name,
                        MIN(end_station_name) AS end_station_name,
                        AVG(duration_sec) AS avg_duration_sec
                    FROM `bigquery-public-data.san_francisco.bikeshare_trips`
                    WHERE start_station_id != end_station_id 
                    GROUP BY start_station_id, end_station_id
                    LIMIT 10
                    """
    show_amount_of_data_scanned(more_data_query)

    less_data_query = """
                      SELECT start_station_name,
                          end_station_name,
                          AVG(duration_sec) AS avg_duration_sec                  
                      FROM `bigquery-public-data.san_francisco.bikeshare_trips`
                      WHERE start_station_name != end_station_name
                      GROUP BY start_station_name, end_station_name
                      LIMIT 10
                      """
    show_amount_of_data_scanned(less_data_query)
   ```
   输出：
   ```python
   Data processed: 0.076 GB
   Data processed: 0.06 GB
   ```
   在`less_data_query`的`WHERE`语句中，使用`start_station_name`代替`start_station_id`（因为二者一一对应），可以节省读取的数据量。
3. 避免使用 N:N JOINs
   + 什么是“N:N JOIN”？首先，我们之前遇到的`JOIN`的表格中，每个表格的一行数据至多与另一表格的一行数据匹配：
   ![1-1join](1-1join.png)
   + 这被称为“ **1:1 JOIN** ”。如果一个表格的一行数据可与另一个表格的多行数据匹配，如下图：
   ![n-1join](n-1join.png)
   + 那么就被称为“ **N:1 JOIN** ”。同理，如果一个表格的多行数据与另一个表格的多行数据相互匹配，如下图：
   ![n-njoin](n-njoin.png)
   + 就被称为“ **N:N JOIN** ”。
   + 我们直接通过下面两个例子的比较分析“N:N JOIN”的执行效率：
   ```python
   big_join_query = """
                    SELECT repo,
                        COUNT(DISTINCT c.committer.name) as num_committers,
                        COUNT(DISTINCT f.id) AS num_files
                    FROM `bigquery-public-data.github_repos.commits` AS c,
                        UNNEST(c.repo_name) AS repo
                    INNER JOIN `bigquery-public-data.github_repos.files` AS f
                        ON f.repo_name = repo
                    WHERE f.repo_name IN ( 'tensorflow/tensorflow', 'facebook/react', 'twbs/bootstrap', 'apple/swift', 'Microsoft/vscode', 'torvalds/linux')
                    GROUP BY repo
                    ORDER BY repo
                    """
   show_time_to_run(big_join_query)
   small_join_query = """
                      WITH commits AS
                      (
                      SELECT COUNT(DISTINCT committer.name) AS num_committers, repo
                      FROM `bigquery-public-data.github_repos.commits`,
                          UNNEST(repo_name) as repo
                      WHERE repo IN ( 'tensorflow/tensorflow', 'facebook/react', 'twbs/bootstrap', 'apple/swift', 'Microsoft/vscode', 'torvalds/linux')
                      GROUP BY repo
                      ),
                      files AS 
                      (
                      SELECT COUNT(DISTINCT id) AS num_files, repo_name as repo
                      FROM `bigquery-public-data.github_repos.files`
                      WHERE repo_name IN ( 'tensorflow/tensorflow', 'facebook/react', 'twbs/bootstrap', 'apple/swift', 'Microsoft/vscode', 'torvalds/linux')
                      GROUP BY repo
                      )
                      SELECT commits.repo, commits.num_committers, files.num_files
                      FROM commits 
                      INNER JOIN files
                          ON commits.repo = files.repo
                      ORDER BY repo
                      """

   show_time_to_run(small_join_query)
   ```
   输出：
   ```python
   Time to run: 13.028 seconds
   Time to run: 4.413 seconds
   ```
   可以看到，“N:N JOIN”在处理大规模数据时效率会降低。为解决这一问题，我们需要提前将数据进行筛选与聚合，减小数据集规模。（即将筛选过程放在合并过程之前，必要时可使用CTE）
## 更多优化策略
参见[Google BigQuery: The Definitive Guide](https://static1.squarespace.com/static/5530dddfe4b0679504639dc1/t/65e17f819cccca1eeb8a8278/1709277088797/Valliappa+Lakshmanan_+Jordan+Tigani+-+Google+BigQuery_+The+Definitive+Guide_+Data+Warehousing%2C+Analytics%2C+and+Machine+Learning+at+Scale-O%E2%80%99Reilly+Media+%282020%29.pdf)。

[ **再次恭喜！你已经成为SQL高手了！おめでとう！** ]{.green}