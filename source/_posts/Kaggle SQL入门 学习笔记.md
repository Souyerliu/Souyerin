---
title: Kaggle SQL入门 学习笔记
date: 2025-10-15 13:44:52
categories:
 - 计算机科学
 - SQL
tags:
 - SQL
 - BigQuery
cover: cover.jpg
---
教程链接：<https://www.kaggle.com/learn/intro-to-sql>
注：本教程使用python语言执行sql语句的相关操作，数据均源于Google BigQuery。
参考链接：<https://cloud.google.com/python/docs/reference/bigquery/latest/index.html>

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
