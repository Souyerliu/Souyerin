---
title: Kaggle SQL入门 学习笔记
date: 2025-10-15 13:44:52
categories:
 - study
 - [计算机科学,Kaggle-sql-tutorial]
tags:
 - SQL
 - BigQuery
cover: cover.jpg
---
教程链接：<https://www.kaggle.com/learn/intro-to-sql>
注：本教程使用python语言执行sql语句的相关操作，数据均源于Google BigQuery。
Google BigQuery官方文档：<https://cloud.google.com/python/docs/reference/bigquery/latest/index.html>
Kaggle上收录的Google BigQuery数据集：<https://www.kaggle.com/datasets?fileType=bigQuery>

# Getting started
1. 使用以下代码加载BigQuery数据库：
```python
from google.cloud import bigquery
```
注：如果是在本地python或jupyter notebook中运行，需要先确保库中已经包含google.cloud.bigquery，否则就先进行以下安装：
```shell
pip install google-cloud-bigquery
```
安装完后还需要设置Google cloud认证：
+ 在<https://console.cloud.google.com/iam-admin/serviceaccounts>里建立一个service account（可能需要先创建一个project）
+ 创建后点击`Actions`，选择`Manage keys`
+ 点击`Add key`，选择`Create new key`
+ `Key type`选择JSON
+ 网站会自动下载JSON文件到本地，将文件存放路径记住
+ 对于windows系统，需要在系统环境变量中设置：
```shell
[System.Environment]::SetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS", "xxx\<project>-<key>.json", "User")
```
其中xxx为JSON存放路径。
+ 设置完成后，重新运行python脚本，运行以下代码：
```python
from google.cloud import bigquery
client = bigquery.Client()
print(client.project)
```
若能打印出project的ID，说明连接成功。
+ 为了从数据库上获取数据，需要创建一个`client`对象：
```python
client = bigquery.Client()
```
2. 我们选择获取bigQuery上的[hacker news](https://news.ycombinator.com/)数据集，这个数据库包含于`bigquery-public-data`项目下。
我们需要执行以下代码：
```python
# 使用dataset()建立一个到hacker news数据集的引用(reference)
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")

# 使用get_dataset()建立API请求以获取数据集
dataset = client.get_dataset(dataset_ref)
```
3. 每个数据集都由多个表格组成。我们可以使用`list_tables()`将数据集中的表格列出来：
```python
# 列举所有在hacker_news数据集里的表格
tables = list(client.list_tables(dataset))

# 打印出所有表格的名称
for table in tables:  
    print(table.table_id)
```
注：笔者这里输出结果只有一个表格`full`，个人推测是这个数据集对表格进行了整合，将内容都集中在一个表格里了。

4. 与获取数据集的方法类似，我们也可以获取一个表格：
```python
# 建立对full表格的引用
table_ref = dataset_ref.table("full")

# 使用get_table()建立API请求以获取表格
table = client.get_table(table_ref)
```
下图展示了`client`,`bigquery-public-data`,`hacker_news`和`full`之间的关系：
![dataset-table](table.png)

## 表模式(table schema)
+ 表模式反映了数据各列的一些基本信息，这些信息有助于我们确定应该获取哪些数据：
```python
# 打印hacker_news数据集里full表格各列的信息
print(table.schema)
```
输出结果如下：
```python
[SchemaField('title', 'STRING', 'NULLABLE', None, 'Story title', (), None), 
SchemaField('url', 'STRING', 'NULLABLE', None, 'Story url', (), None), 
SchemaField('text', 'STRING', 'NULLABLE', None, 'Story or comment text', (), None), 
SchemaField('dead', 'BOOLEAN', 'NULLABLE', None, 'Is dead?', (), None), 
SchemaField('by', 'STRING', 'NULLABLE', None, "The username of the item's author.", (), None), 
SchemaField('score', 'INTEGER', 'NULLABLE', None, 'Story score', (), None), 
SchemaField('time', 'INTEGER', 'NULLABLE', None, 'Unix time', (), None), 
SchemaField('timestamp', 'TIMESTAMP', 'NULLABLE', None, 'Timestamp for the unix time', (), None), 
SchemaField('type', 'STRING', 'NULLABLE', None, 'type of details (comment comment_ranking poll story job pollopt)', (), None), 
SchemaField('id', 'INTEGER', 'NULLABLE', None, "The item's unique id.", (), None), 
SchemaField('parent', 'INTEGER', 'NULLABLE', None, 'Parent comment ID', (), None), 
SchemaField('descendants', 'INTEGER', 'NULLABLE', None, 'Number of story or poll descendants', (), None), 
SchemaField('ranking', 'INTEGER', 'NULLABLE', None, 'Comment ranking', (), None), 
SchemaField('deleted', 'BOOLEAN', 'NULLABLE', None, 'Is deleted?', (), None)]
```
+ 每个SchemaField都包含以下信息：
  + 列的名称(name)
  + 列的数据类型(field type)
  + 列的数据模式(`'NULLABLE'`代表该列允许数据空值，也是默认选项)
  + 列的数据描述(description)
+ 另外，我们还可以用`list_rows()`查看表格的前几行，以确保数据描述的准确性：
```python
# 查看full表格的前5行
print(client.list_rows(table, max_results=5).to_dataframe())
```
输出表格如下：
```python
  title   url  text  dead    by  score        time                 timestamp   type       id  parent  descendants  ranking  deleted
0  None  None  None  <NA>  None   <NA>  1437576722 2015-07-22 14:52:02+00:00  story  9929939    <NA>         <NA>     <NA>     <NA>
1  None  None  None  True  None   <NA>  1437577213 2015-07-22 15:00:13+00:00  story  9929995    <NA>         <NA>     <NA>     <NA>
2  None  None  None  <NA>  None   <NA>  1437577406 2015-07-22 15:03:26+00:00  story  9930011    <NA>         <NA>     <NA>     <NA>
3  None  None  None  <NA>  None   <NA>  1437577665 2015-07-22 15:07:45+00:00  story  9930036    <NA>         <NA>     <NA>     <NA>
4  None  None  None  <NA>  None   <NA>  1437578250 2015-07-22 15:17:30+00:00  story  9930094    <NA>         <NA>     <NA>     <NA>
```

+ 当然，也可以设定`selected_fields`查看特定的几列数据（如果要根据列名查找对应的列数据，需要下面的SELECT...FROM指令）：
```python
# 查看full表格time列的前5行
client.list_rows(table, selected_fields=table.schema[6:7], max_results=5).to_dataframe()
```
输出：
```python
         time
0  1437576722
1  1437577213
2  1437577406
3  1437577665
4  1437578250
```
# Select, From & Where
+ 使用`Select`,`From`和`Where`可以更精细地获取想要的数据。
## SELECT ... FROM
+ 为了选择特定表格的特定列的数据，需要编写`query`查询指令：
```python
query="""
      SELECT Name
      FROM `bigquery-public-data.pet_records.pets`
      """
```
+ 这里`Name`是列名，`pets`是表格名，`pet_records`是数据集。
## WHERE
+ 有时对同一列的数据需要进行进一步筛选，此时就要用到`WHERE`指令：
```python
query="""
      SELECT Name
      FROM `bigquery-public-data.pet_records.pets`
      WHERE Animal='Cat'
      """
```
+ 这里我们筛选了`pets`表格里满足`Animal`为`Cat`对应行的`Name`列
## 实操
1. 前述准备代码如下：
```python
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()
#fetch the dataset
dataset_ref = client.dataset("openaq", project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)

tables = list(client.list_tables(dataset))
for table in tables:  
    print(table.table_id)
#fetch the table
table_ref = dataset_ref.table("global_air_quality")
table = client.get_table(table_ref)

print(client.list_rows(table, max_results=5).to_dataframe())
```
2. 在获取表格之后，我们编写`query`：
```python
query = """
        SELECT city
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """
```
注：在本地运行`query`时，最好可以安装相应的BigQuery Storage API的客户端库：
```shell
pip install google-cloud-bigquery-storage
```
这样可以显著加快`.to_dataframe()`的速度。
3. 接着，我们利用`query`获取特定的数据：
```python
# 设置查询query
query_job = client.query(query)

# API请求 - 运行查询, 返回dataframe
us_cities = query_job.to_dataframe()
```
在`us_cities`里，就已经存放好了我们想要的数据。此时我们就可以以`dataframe`的方式处理这个表格了：
```python
# 输出前五行
print(us_cities.head())

# 输出频数最高的前五个城市
print(us_cities.city.value_counts().head())
```
输出：
```python
     city
0  HOWARD
1  HOWARD
2  HOWARD
3  HOWARD
4  HOWARD
city
Phoenix-Mesa-Scottsdale                     39414
Los Angeles-Long Beach-Santa Ana            27479
Riverside-San Bernardino-Ontario            26887
New York-Northern New Jersey-Long Island    25417
San Francisco-Oakland-Fremont               22710
Name: count, dtype: int64
```
注：有时同一列中会有大量重复数据，如果我们想要去除重复数据，可以在query中进行修改：
```python
query = """
        SELECT DISTINCT city
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """
```
这样重复的数据只会出现一次。
## 查询多个列
+ 如果需要查询更多的列数据，可以将`SELECT`后的参数进行修改：
```python
query = """
        SELECT city, country
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """
```
+ 还可以一次性查询所有列：
```python
query = """
        SELECT *
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """
```
## 获取查询数据的大小
+ 有时，我们需要查询的数据量较大（可能达到GB甚至TB量级）。为了能提前估计我们处理的数据大小，我们可以运行以下代码：
```python
# 创建QueryJobConfig对象，在运行query查询前估计数据量大小
dry_run_config = bigquery.QueryJobConfig(dry_run=True)

# dry run query
dry_run_query_job = client.query(query, job_config=dry_run_config)

print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))
```
+ 当然，还可以设置最大的查询数据大小，当数据量过大时停止查询：
```python
# 当数据量小于1MB时运行查询
ONE_MB = 1000*1000
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_MB)

# 设置query
safe_query_job = client.query(query, job_config=safe_config)

# 尝试运行query(会报错并终止运行)
safe_query_job.to_dataframe()
```
# Group By, Having & Count
+ 这一节，我们将在查询选择特定数据后对数据进行初步统计。
## COUNT()
+ 顾名思义，这个函数返回就是括号内指定列的数据个数。
+ 使用例：
```python
# 获取global_air_quality表格中city列的数据总个数
query = """
        SELECT COUNT(city)
        FROM `bigquery-public-data.openaq.global_air_quality`
        """
```
+ 它会输出类似下面这样的表格：
```python
        f0_
0       5594614
```
+ `COUNT()`是一个典型的聚合函数（多个输入，一个输出）。其他聚合函数包括`SUM()`, `AVG()`, `MIN()`和`MAX()`。
+ 另外，可以看到输出的表格列名是默认的`f0_`，关于如何修改这个名称会在后面叙述。
## GROUP BY
+ `GROUP BY`函数可以根据指定列内容对数据进行分类（指定列内容相同的数据归为一类），配合`COUNT()`可以对每一组进行数量统计。
使用例：
```python
# 获取global_air_quality表格中每个country的city个数
query = """
        SELECT country,COUNT(city)
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY country
        """
```
输出类似如下（输出前5行，下同）：
```python
  country    f0_
0      ME  13466
1      HR  13428
2      DK  11190
3      BA    177
4      CY   3903
```
## GROUP BY ... HAVING
+ `HAVING`在`GROUP BY`的基础上再增加筛选条件（如筛选有指定范围数量数据的组）
+  使用例：
```python
query = """
        SELECT country,COUNT(city)
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY country
        HAVING COUNT(city)<10000
        """
```
输出类似如下：
```python
  country   f0_
0      AD  2920
1      BA   177
2      CY  3903
3      AR  2620
4      LU  7072
```
+ 注意与`WHERE`的区别：`WHERE`是对数据集整体进行筛选（一般在`SELECT...FROM`之后，`GROUP BY`之前），而`HAVING`是在分组后对组的聚合函数（频数，均值，最大值等）进行筛选（在`GROUP BY`之后，`ORDER BY`之前）
## 改进
+ 上面说到，`f0_`可以进行修改。具体来说，我们可以通过在`COUNT()`后加入`AS XXX(名称)`来重命名列名。例如：
```python
query_improved = """
                 SELECT parent, COUNT(1) AS NumPosts
                 FROM `bigquery-public-data.hacker_news.full`
                 GROUP BY parent
                 HAVING COUNT(1) > 10
                 """
```
这一技巧被称为 **“混叠”(aliasing)** ，后续会详细讲述。
+ 另一方面，如果不确定`COUNT()`括号里面的内容（即不确定计数的列）时，使用`COUNT(1)`或`COUNT(*)`也是一个较好的选择，而且运行速度更快！（这两个与`COUNT(列名)`的区别在于前者考虑了`NULL`值，而后者不考虑）
## 注意事项
+ 使用`GROUP BY`时，在`SELECT`语句里选择的列名应该都满足下述两者条件之一：
  + 在`GROUP BY`中；
  + 在聚合函数（如`COUNT()`）中。
+ 可以理解为，使用`GROUP BY`就相当于对数据进行了压缩，不可避免地会产生一定信息的损失。
+ 另外，由于`GROUP BY`为关键字，所以如果列名中有`by`，`group`，那么就需要再`SELECT`语句中对其加\`\`号，最好再使用`AS`语句进行重命名。
# Order by
## ORDER BY
+ `ORDER BY`指令常放在query的最后，用于将前面得到的数据根据一定的规则进行排序。
+ 可以对数进行排序（默认升序），也可以对字符串进行排序（以字母表顺序）
+ 使用例：
```python
# 获取global_air_quality表格中的country列，并按字母表排序
query = """
        SELECT city,longitude
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        ORDER BY longitude
        """
```
输出：
```python
       city  longitude
0    BETHEL  -161.7670
1    BETHEL  -161.7670
2    BETHEL  -161.7670
3  Honolulu  -158.0886
4  Honolulu  -158.0886
```
如果需要降序排列，则在`ORDER BY 列名`后加上`DESC`(descending)
## Dates
+ 在数据库中，日期Date出现非常频繁
+ 在BigQuery中，日期可以以两种方式存储：
  + **DATE**：格式为年-月-日(YYYY-\[M\]M-\[D\]D)
    + 例：2025-10-17
  + **DATETIME**：格式为年-月-日 时:分:秒(YYYY-\[M\]M-\[D\]D HH:mm:ss)
    + 例：2025-10-17 23:35:23
+ 当我们需要筛选特定日期范围内的数据时，可以直接使用不等号进行日期比较，比如：
```python
# 已知trip_start_timestamp是DATETIME格式
query = """
        SELECT trip_start_timestamp,trip_miles,trip_seconds
        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
        WHERE trip_seconds > 0 and trip_miles > 0 
              and trip_start_timestamp > "2016-01-01" 
              and trip_start_timestamp < "2016-04-01"
        """
```
就可以筛选出时间在2016年1月1日到2016年4月1日之间的数据。
## EXTRACT
+ 在query中使用`EXTRACT`可以获取时间列中的特定元素（如年份，月份等）
+ 使用例（没有合适的数据库，所以直接复制kaggle教程里的了）：
```python
query="""
    SELECT Name, EXTRACT(DAY from Date) AS Day
    FROM `bigquery-public-data.pet_records.pets_with_date`
    """
```
输出：
```python
                Name      Day
0 Dr. Harris Bonkers      18
1               Tom       16
2               Moon      7
3               Ripley    23
```
+ 其他可调用日期相关函数详见[文档](https://cloud.google.com/bigquery/docs/reference/legacy-sql?hl=zh-cn#datetimefunctions)
# With & As
+ 在之前我们提到，使用`AS`语句可以对列名进行重命名。事实上，配合`WITH`，可以将query内的代码模块化，进一步增强可读性。这种模块被称为**公共表表达式(Common Table Expression,CTE)**。
## 公共表表达式（CTE）
+ 一个典型的CTE形式如下：
```python
query = """
        WITH Seniors AS
        (
          SELECT ID, Name
          FROM'bigquery-public-data.pet_records.pets
          WHERE Years_old > 5
        )
        ...
        """
```
+ 这实际上将括号内查询获取的结果保存为一个名为`Seniors`的表格（但不会被query直接输出），以供后续调用，比如：
```python
query = """
        WITH Seniors AS 
        (
          SELECT ID, Name
          FROM'bigquery-public-data.pet_records.pets
          WHERE Years_old > 5
        )
        SELECT ID
        FROM Seniors
        """
```
+ 可以看到，我们可以将创建好的CTE在后续query语句中当作数据集进行使用。这对一些大型SQL项目来说非常有用。而且它还可以提高数据处理的效率。
+ 注：CTE只能在其定义所在的query中使用，所以每次在使用前必须先定义好CTE。
# Joining Data
+ 有时我们需要的数据来自多个数据集（表格），因而需要合并多个表格。这时我们就可以使用`JOIN`语句。
## JOIN
+ 我们直接通过一个例子进行阐述（注：以下所有图片与数据均源于Kaggle教程）：
+ 原始表格：
![origin](origin.png)
+ 可以看到，`owners`表格里的`Pet_ID`与`pets`表格里的`ID`一一对应，因而我们可以将两个表格合并。
+ 代码如下：
```python
query="""
      SELECT p.Name AS Pet_Name, o.Name AS Owner_Name
      FROM `bigquery-public-data.pet_records.pets` AS p
      INNER JOIN `bigquery-public-data.pet_records.owners` AS o
      ON p.ID =o.Pet_ID
      """
```
输出：
+ ![join](join.png)
+ 可以看到，`ON`语句决定了合并两个表格使用的规则。当然，在合并表格前，应该提前明确参考的列数据
+ 另外注意到，我们在这里使用了`INNER JOIN`，这代表着合并后的表格只会取`owners`表格里的`Pet_ID`与`pets`表格里的`ID`相等的行（即如果某个表格里的`ID`在另一个表格中找不到对应编号，那么这条数据就被废除）
  + 当然也有保留无匹配数据的方式，这会在后续进阶笔记中呈现。
+ 实际上，两个表格的行数不一定相等（可能存在一对多的情况），最终得到的表格的行数应为两个表格行数的较大者（前提是所有数据都能配对）
# 最后一个技巧：LIKE
+ 在使用`WHERE`筛选字符串数据时，可以用`LIKE`配合`%`进行模糊筛选。
+ 使用例：
```python
query = """
        SELECT id,title,owner_user_id
        FROM `bigquery-public-data.stackoverflow.posts_questions`
        WHERE tags LIKE '%bigquery%'
        """
```
这里query会查询`tags`中所有包含`"bigquery"`字段的数据。

[ **恭喜完成SQL入门！おめでとう！** ]{.red}