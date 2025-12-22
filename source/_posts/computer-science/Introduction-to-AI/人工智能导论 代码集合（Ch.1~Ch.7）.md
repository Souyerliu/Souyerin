---
title: 人工智能导论 代码集合(Ch.1~Ch.7)
date: 2025-12-06 22:46:00
categories: 
  - [study]
  - [计算机科学,人工智能导论]
tags:
  - 人工智能
cover: cover.jpg
---
老师还是听劝，终于把代码端了上来!!有没有用另说!!   
笔者进行了一番品鉴，取其精华，汇总于此，以作参考（部分运行结果请参考所附链接，使用了在线python IDE）   
另外，笔者认为，就目前阶段而言，不必掌握所有代码的细节，知道应该调用哪个库解决即可（去查官方文档！）
# Ch.2 人工智能数据基础
## 数据相似性度量
以下算法用于比较预测值与真实值间的差异：
1. 欧氏距离  
手搓版（[euclidean_distance](https://www.onlineide.pro/playground/share/23edff2d-2869-4727-ab0d-d27f1bd1e0f8)）：
```python
import math
def euclidean_distance(p,q):
  if len(p) != len(q):
    raise ValueError("The two vectors must have the same length")
  distance =0.0
  for i in range(len(p)):
    distance +=(p[i]- q[i])**2
  return math.sqrt(distance)

#示例
p = [2,4,6]
q = [1,3,5]
print(euclidean_distance(p,q))
``` 
使用numpy实现（[euclidean_distance_numpy](https://www.onlineide.pro/playground/share/0c53297c-8056-4843-a24d-d1f933d88a1b)）：
```python
import numpy as np
def euclidean_distance_numpy(p,q):
  return np.sqrt(np.sum((np.array(p)-np.array(q))** 2))

#示例
p=[2,4,6]
q=[1,3,5]
print(euclidean_distance_numpy(p,q))
```
2. 余弦相似度（[cos_sim](https://www.onlineide.pro/playground/share/8eda894a-bc3b-4860-a51e-acaf8e9f1e49)）
```python
import numpy as np
vec1= np.array([1,2,3,4])
vec2= np.array([5,6,7,8])
cos_sim =vec1.dot(vec2)/ (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(cos_sim)
```
3. SMC & Jaccard系数   
手搓版（[smc_and_jaccard](https://www.onlineide.pro/playground/share/b872bb01-980d-4161-8494-af8a9c706005)）：
```python
def smc_similarity(A, B, U):
    match = 0
    for x in U: 
        match += 1 if (x in set(A)) == (x in set(B)) else 0
    smc = match / len(U)
    return smc
def jaccard_similarity(A,B):
  intersection = len(set(A) & set(B))
  union = len(set(A) | set(B))
  jaccard_coefficient = intersection/union
  return jaccard_coefficient
#定义两个集合的列表表示
A=[1,2,3,4,5]
B=[4,5,6,7,8]
U=[1,2,3,4,5,6,7,8,9,10]
#计算smc与jaccard相似系数
smc_coefficient=smc_similarity(A,B,U)
jaccard_coefficient=jaccard_similarity(A,B)
print("SMC系数:",smc_coefficient)
print("Jaccard相似系数:",jaccard_coefficient)
```
使用sklearn.metrics的[Jaccard_score](https://scikit-learn.cn/stable/modules/generated/sklearn.metrics.jaccard_score.html)（[jaccard_with_sklearn](https://onecompiler.com/python/446ujsdqc)）：
```python
from sklearn.metrics import jaccard_score
#定义两个集合的列表表示
A=[1,2,3,4,5]
B=[4,5,6,7,8]
U = sorted(set(A) | set(B))
# 将集合转为布尔向量
A_vec = [1 if x in A else 0 for x in U]
B_vec = [1 if x in B else 0 for x in U]
#使用sklearn中的jaccard_score函数计算Jaccard相似系数
jaccard_coefficient=jaccard_score(A_vec,B_vec)
print("Jaccard相似系数:",jaccard_coefficient)
```
## 数据可视化
略，可参考[matplotlib.pyplot官方文档](https://matplotlib.org/stable/api/pyplot_summary.html#module-matplotlib.pyplot)与[matplotlib.pyplot官方教程](https://matplotlib.org/stable/tutorials/pyplot.html#sphx-glr-tutorials-pyplot-py)。

# Ch.5 机器学习——监督学习
## 回归分析
一元线性回归（[Simple linear regression](http://matplotlib.codeutility.io/en/a733l9)）：
```python
import numpy as np
import matplotlib.pyplot as plt
#-------------------1.准备数据---------------------
data=np.array([[32,31],[53,68],[61,62],[47,71],[59,87],[55,78],[52,79],[39,59],[48,75],[52,71],
[45,55],[54,82],[44,62],[58,75],[56,81],[48,60],[44,82],[60,97],[45,48],[38,56],
[66,83],[65,118],[47,57],[41,51],[51,75],[59,74],[57,95],[63,95],[46,79],[50,83]])
#提取data中的两列数据，分别作为x，y
x = data[:,0]
y = data[:,1]
#用plt画出散点图
#plt.scatter(x, y)
#plt.show()
#-------------------2．定义损失函数---------------------
#损失函数是系数的函数，另外还要传入数据的x，y
def compute_cost(w,b,points):
  total_cost = 0
  M = len(points)
  #逐点计算平方损失误差，然后求平均数
  for i in range(M):
    x = points[i,0]
    y = points[i,1]
    total_cost += (y - w * x - b) ** 2
  return total_cost / M
#-------------------3.定义算法拟合函数（最小二乘法）----------------
#先定义一个求均值的函数
def average(data):
  sum = 0
  num = len(data)
  for i in range(num):
    sum += data[i]
  return sum / num
  #定义核心拟合函数
def fit(points):
  M = len(points)
  x_bar = average(points[:,0])
  sum_yx = 0
  sum_x2 = 0
  sum_delta =0
  for i in range(M):
    x = points[i,0]
    y = points[i,1]
    sum_yx += y*(x - x_bar)
    sum_x2 += x ** 2
  #根据公式计算w
  w = sum_yx / (sum_x2- M * (x_bar ** 2))
  for i in range(M):
    x = points[i,0]
    y = points[i,1]
    sum_delta += (y - w * x)
    b = sum_delta/M
  return w,b
#-------------------4.测试----------------------
w,b = fit(data)
print("w is: ",w)
print("b is:",b)
cost = compute_cost(w,b,data)
print("cost is: ",cost)
#------------------5．画出拟合曲线-------------------------
plt.scatter(x,y)
#针对每一个x，计算出预测的y值
pred_y = w * x + b
plt.plot(x,pred_y,c='r')
plt.show()
```
输出结果：
```
w is:  1.5933633756656984
b is: -8.5604260548949
cost is:  117.28701351904957
```
![regression](regression.png)
## 决策树
基于下述数据，对银行是否给予贷款建立决策树模型（使用[sklearn.tree.DecisionTreeClassifier](https://scikit-learn.cn/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)）：
![case](case.png)
```python
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import math
#创建数据集
data={
  '年龄':['>30','>30','20~30','<20','<20','<20','20~30','>30','>30','<20','>30','20~30','20~30','<20'],
  '银行流水':['高','高','高','中','低','低','低','中','低','中','中','中','高','中'],
  '是否结婚':['否','否','否','否','否','否','是','是','是','是','是','是','否','否'],
  '拥有房产':['是','否','是','是','是','是','是','是','是','是','是','否','是','是'],
  '是否给子女贷款':['否','否','是','是','是','否','是','否','是','是','是','是','是','否']
}

# 转换为dataframe
df=pd.DataFrame(data)

#将特征列转换为数值类型
df['年龄'] = df['年龄'].map({'<20':0,'20~30':1,'>30':2})
df['银行流水']=df['银行流水'].map({'低':0,'中':1,'高':2})
df['是否结婚']=df['是否结婚'].map({'否':0,'是':1})
df['拥有房产']=df['拥有房产'].map({'否':0,'是':1})
df['是否给子女贷款']=df['是否给子女贷款'].map({'否':0,'是':1})

# 定义特征和标签
X=df[['年龄','银行流水','是否结婚','拥有房产']]#特征
y=df['是否给子女贷款'] #标签

#计算熵值
def calcEntropy(dataSet):
  # 1. 获取所有样本数
  exampleNum =len(dataSet)
  # 2. 计算每个标签值的出现数量
  labelcount={}
  for featVec in dataSet:
    curLabel= featVec[-1]
    if curLabel in labelcount.keys():
      labelcount[curLabel]+=1
    else:
      labelcount[curLabel]=1
  # 3. 计算熵值（对每个类别求熵值求和）
  entropy= 0
  for key,value in labelcount.items():
    # 概率值
    p =labelcount[key] /exampleNum
    #当前标签的熵值计算并追加
    curEntropy =-p * math.log(p,2)
    entropy += curEntropy
  # 4．返回
  return entropy

# 计算整体熵值
dataList = df.values.tolist()
entropy_value = calcEntropy(dataList)
print("数据集的整体信息熵为：", entropy_value)

# 构建决策树（ID3）
clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)

# 决策树可视化
plt.figure(figsize=(16,10))
tree.plot_tree(clf, 
               feature_names=['年龄','银行流水','是否结婚','拥有房产'],
               class_names=['否','是'],
               filled=True)
plt.show()
```
输出结果：
```
数据集的整体信息熵为： 0.9402859586706309
```
![decisiontree](decisiontree.png)
## 线性判别分析LDA
1. numpy手搓版（以iris数据集为例）（[lda_with_numpy](https://onecompiler.com/python/446va4rrw)）：
```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
import numpy as np
# 加载数据集（这里以Iris数据集为例）
data =load_iris()
X = data.data
y = data.target
# 转换标签为数字（如果标签为字符串）
# label_encoder=LabelEncoder()
# y=label_encoder.fit_transform(y)

# 1. 计算数据样本本集中每个类别样本的均值
class_labels = np.unique(y)
mean_vectors=[]
for label in class_labels:
  mean_vectors.append(np.mean(X[y==label],axis=0))

# 2. 计算类内散度矩阵Sw和类间散度矩阵Sb
d = X.shape[1] #特征数
n = X.shape[0] #样本总数
k = len(class_labels) #类别数
# 类内散度矩阵Sw
Sw = np.zeros((d,d))
for label,mean_vec in zip(class_labels,mean_vectors):
  class_scatter = np.zeros((d,d))
  class_samples= X[y==label]
  for sample in class_samples:
    sample = sample.reshape(d,1) # 转化为列向量
    mean_vec = mean_vec.reshape(d,1) # 同上
    class_scatter += (sample - mean_vec).dot((sample - mean_vec).T)
  Sw += class_scatter
# 类间散度矩阵sb
mean_all= np.mean(X,axis=0).reshape(d,1)
Sb = np.zeros((d,d))
for label,mean_vec in zip(class_labels,mean_vectors):
    n_i=X[y==label].shape[0]
    mean_vec = mean_vec.reshape(d,1)
    Sb += n_i *(mean_vec - mean_all).dot((mean_vec - mean_all).T)

# 3.求解 Sw^-1 * Sb的特征向量
eigvals,eigvecs=np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
# 按照特征值的大小排序特征向量
sorted_indices=np.argsort(eigvals)[::-1]
eigvecs_sorted=eigvecs[:,sorted_indices]

# 4. 选择前r个特征向量
W=eigvecs_sorted[:,:2] #选择前两个特征向量

# 5. 将每个样本映射到低维空间
X_lda= X.dot(W)
# 输出降维后的结果
print("降维后的数据：")
print(X_lda)
```
2. 使用sklearn.discriminant_analysis的[LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)：
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
# 1．生成三类二维数据
X,y=make_classification(n_samples=150,n_features=2,n_redundant=0,n_informative=2,
n_clusters_per_class=1,n_classes=3,random_state=42)

# 2. 划分训练集和测试集
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# 3. 训练LDA模型
lda=LinearDiscriminantAnalysis()
lda.fit(X_train,y_train)

# 4. 预测与评估
y_pred =lda.predict(X_test)
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)

# 5. 可视化LDA降维结果
X_lda = lda.transform(X)
plt.scatter(X_lda[:,0],np.zeros_like(X_lda[:,0]),c=y,cmap=plt.cm.Set1,edgecolor='k')
plt.title('LDA 1D Projection(3-class)')
plt.xlabel('LDA Component 1')
plt.yticks([])
plt.show()
print("LDA Test Accuracy:",acc)
print("Confusion Matrix:\n",cm)
```
输出结果：
```
LDA Test Accuracy: 0.9777777777777777
Confusion Matrix:
 [[14  1  0]
 [ 0 15  0]
 [ 0  0 15]]
```
![lda](lda.png)

## Ada Boosting（自适应提升）
使用sklearn.ensmble的[AdaBoostClassifier](https://scikit-learn.cn/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)（[Adaboosting](https://onecompiler.com/python/446vbqwhm)）：
```python
#导入所需库
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 创建一个示例数据集
X,y = make_classification(n_samples=1000,n_features=20,n_classes=2,random_state=42)

# 划分数据集为训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 创建基学习器（弱学习器），使用深度为1的决策树
base_learner = DecisionTreeClassifier(max_depth=1)

# 创建AdaBoost分类器
adaboost = AdaBoostClassifier(estimator=base_learner,n_estimators=50,random_state=42)

# 训练模型
adaboost.fit(X_train,y_train)

# 使用测试集进行预测
y_pred = adaboost.predict(X_test)

# 计算模型准确度
accuracy= accuracy_score(y_test,y_pred)
print("真实值：",y_test)
print("预测值：",y_pred)
print(f"AdaBoost模型准确度:{accuracy:.4f}")
```
## 支持向量机
使用sklearn.svm的[SVC](https://scikit-learn.cn/stable/modules/generated/sklearn.svm.SVC.html)（[SVM](https://onecompiler.com/python/446vcbnbj)）
```python
#导入所需库
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
# 创建一个示例数据集
X,y = make_classification(n_samples=1000,n_features=20,n_classes=2,random_state=42)

# 划分数据集为训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 创建支持向量机分类器（线性内核）
svm_linear = SVC(kernel='linear',random_state=42)

# 训练模型
svm_linear.fit(X_train,y_train)

# 使用测试集进行预测
y_pred_linear = svm_linear.predict(X_test)

# 计算模型准确度
accuracy_linear = accuracy_score(y_test,y_pred_linear)
print(f"线性SVM模型准确度:{accuracy_linear:.4f}")

# 创建支持向量机分类器（RBF内核）
svm_rbf = SVC(kernel='rbf',random_state=42)

# 训练模型
svm_rbf.fit(X_train,y_train)

# 使用测试集进行预测
y_pred_rbf = svm_rbf.predict(X_test)

# 计算模型准确度
accuracy_rbf = accuracy_score(y_test,y_pred_rbf)
print(f"RBF SVM模型准确度:{accuracy_rbf:.4f}")
```
# Ch.6 机器学习——非监督学习
## K-means聚类
1. 手搓版：
```python
import numpy as np
import matplotlib.pyplot as plt # 可视化结果,便于演示
from sklearn.datasets import make_blobs # 生成模拟聚类数据
# 1. 定义基础参数（可直接调整）
k = 3 # 聚类数（对应c={c_1,c_2,...,c_k}）
max_iter = 100 # 最大迭代次数（终止条件1）
tol = 1e-4 # 质心变化阀值（终止条件2）
n_samples = 200 # 模拟样本数n
n_features = 2 # 特征维度m（对应c_j∈R^m)
random_state = 42 # 随机种子,保证结果可复现

# 2.生成模拟数据集（已适配真实场景）
# 生成带真实聚类的模拟数据（2维方便可视化）,X∈R^(n×m)
X,_=make_blobs(n_samples=n_samples,
                n_features=n_features,
                centers=k,
                random_state=random_state)
n = len(X) # 样本数（1≤i≤n)
m = X.shape[1] #特征维度（c_j∈R^m)

# 步骤1：初始化k个聚类质心c={c_1,c_2,...,c_k}
# 随机选择k个不重复样本作为初始质心（避免重复选）
random_idx = np.random.choice(n,k,replace=False)
c = X[random_idx].copy() # 初始质心集合,c[j]对应c_j
# 迭代核心（步骤2-步骤4循环）
for iter_num in range(max_iter):
  # 步骤2：计算欧氏距离+分配样本到聚类集合G_j
  G =[[]for _ in range(k)] # 初始化/重置聚类集合G={G1,G2,...,G_k}
  # 计算距离矩阵d：d[i][j]=d(x_i,c_j) (1≤i≤n,1≤j≤k)
  distances = np.sqrt(np.sum((X[:,np.newaxis]- c)**2,axis=2))
  # 分配每个x_i到最近质心的G_j
  for i in range(n):
    j_min = np.argmin(distances[i]) # argmin_{c_j∈C} d(x_i,c_j)
    G[j_min].append(X[i]) # 将x_i放入对应聚类集合
  
  # 步骤3:更新聚类质心c_j=(1/|G_j|)∑(x_i∈G_j)x_i
  new_c=np.zeros((k,m)) # 存储更新后的质心
  for j in range(k):
    G_j= np.array(G[j]) # 第j个聚类的样本集合
    len_Gj =len(G_j) #|G_j|：聚类G_j的样本数量
    if len_Gj > 0:
      new_c[j]=(1/len_Gj)*np.sum(G_j,axis=0) #质心更新公式
    else:
      new_c[j]=c[j]    # 空聚类保留原质心（避免除以0)

  # 步骤4：判断终止条件（质心变化<阈值则停止）
  # 计算质心平均变化量
  centroid_shift = np.mean(np.sqrt(np.sum((new_c- c)**2,axis=1)))
  if centroid_shift <tol:
    break # 满足终止条件,退出迭代
  c = new_c.copy() # 更新质心,进入下一轮迭代

# 可视化最终聚类结果
plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'orange', 'purple', 'cyan']

# 绘制每个聚类的样本点
for j in range(k):
    G_j = np.array(G[j])
    if len(G_j) > 0:
        plt.scatter(G_j[:, 0], G_j[:, 1], 
                    s=30, color=colors[j % len(colors)], 
                    label=f"Cluster {j+1}")

# 绘制最终 k 个聚类质心
plt.scatter(c[:, 0], c[:, 1], 
            s=200, marker='*', color='red', 
            label='Centroids')

plt.title(f"K-Means Clustering Result (iter={iter_num+1})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
```
2. 使用sklearn.cluster的[KMeans](https://scikit-learn.cn/stable/modules/generated/sklearn.cluster.KMeans.html)：
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. 定义基础参数
k = 3
n_samples = 200
n_features = 2
random_state = 42

# 2. 生成模拟数据
X, _ = make_blobs(n_samples=n_samples,
                  n_features=n_features,
                  centers=k,
                  random_state=random_state)

# 3. 使用 sklearn KMeans 进行聚类
kmeans = KMeans(
    n_clusters=k,
    init='random',        
    max_iter=100,
    tol=1e-4,
    random_state=random_state
)

kmeans.fit(X)

# 获取聚类标签和质心
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 4. 可视化（同上）
plt.figure(figsize=(8, 6))

colors = ['blue', 'green', 'orange', 'purple', 'cyan']

for j in range(k):
    cluster_points = X[labels == j]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                s=30, color=colors[j % len(colors)],
                label=f"Cluster {j+1}")

# 绘制质心
plt.scatter(centers[:, 0], centers[:, 1],
            s=200, marker='*', color='red', label='Centroids')

plt.title(f"K-Means Clustering Result (iter={kmeans.n_iter_})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
```

输出结果：
![kmeans](kmeans.png)

## 主成分分析PCA
1. 手搓版（[pca](https://python-fiddle.com/saved/57394eeb-cc3b-4243-b7f2-4e25271deacc)）
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
# 1、定义基础参数（贴合数学符号）
n = 200 # 样本数量（对应n个样本x_i，1≤i≤n)
d = 5 # 原始特征维度（x_i∈R^d)
l = 2 # 降维后目标维度（映射矩阵W为d×l）
random_state=42 # 随机种子，保证结果可复现

# 2、生成模拟数据集+标准化（保证各维度均值为0）
# 生成n个d维样本（模拟真实数据）
X,_ = make_blobs(n_samples=n,n_features=d,centers=3,random_state=random_state)
print(f"原始数据矩阵x维度:nxd={X.shape}")

# 标准化：确保每一维度特征均值为0（贴合“假定均值为零”前提）
X_mean=np.mean(X,axis=0) # 计算各维度均值
X_standard = X - X_mean # 中心化（均值归零)
print(f"标准化后数据均值（各维度）:{np.round(np.mean(X_standard,axis=0),6)}") # 验证均值≈0

# 3、核心步骤：求解PCA映射矩阵W
# 步骤3.1：计算协方差矩阵（dxd），反映维度间的相关性
cov_matrix=np.cov(X_standard,rowvar=False) #rowvar=False:每行是样本,每列是特征
print(f"协方差矩阵维度:dxd={cov_matrix.shape}")

# 步骤3.2：求解协方差矩阵的特征值（λ）和特征向量(v）
eigenvalues,eigenvectors = np.linalg.eig(cov_matrix)
# 特征值λ：反映主成分的方差贡献；特征向量v：每一列是一个特征向量（v_j∈R^d）

# 步骤3.3：按特征值从大到小排序,选择前1个特征向量构建映射矩阵W（dx1）
sorted_idx =np.argsort(eigenvalues)[::-1] # 特征值降序索引
W =eigenvectors[:,sorted_idx[:l]] # 构建dxl的映射矩阵W
print(f"映射矩阵w维度:dxl={W.shape}")

# 4.数据降维：Y=X·W（标准化后数据映射）
# 所有样本降维：Y(nxl）=X_standard(nxd)·W(dx1)
Y = np.dot(X_standard,W)
print(f"降维后数据矩阵Y维度:nxl={Y.shape}")

#单个样本映射验证：x_i(1×d）·W(dxl)→降维为l维向量
x_i=X_standard[0] # 取第一个样本x_1（1xd)
y_i=np.dot(x_i,W)# 单个样本降维
print(f"\n单个样本x_1降维验证:")
print(f"原始x_1维度:1xd={x_i.shape}→降维后y_1维度:1xl={y_i.shape}")
```
2. 使用sklearn.decomposition的[PCA](https://scikit-learn.cn/stable/modules/generated/sklearn.decomposition.PCA.html)（加入降维后数据可视化）
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

plt.rcParams['font.family']=['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
# 1. 参数定义
n = 200
d = 5
l = 2
random_state = 42

# 2. 生成数据 + 中心化
X, labels = make_blobs(n_samples=n, n_features=d, centers=3, random_state=random_state)
print("原始数据 x 维度:", X.shape)

X_mean = X.mean(axis=0)
X_standard = X - X_mean
print("标准化后各维度均值:", np.round(X_standard.mean(axis=0), 6))

# 3. 使用 sklearn PCA
pca = PCA(n_components=l)
Y = pca.fit_transform(X_standard)

print("降维后 Y 维度:", Y.shape)
print("映射矩阵 W 维度:", pca.components_.T.shape)

# 验证单个样本降维
x_i = X_standard[0]
y_i = pca.transform([x_i])
print("\n单个样本 x_1 降维验证：")
print("原始 x_1 维度:", x_i.shape, "→ 降维后 y_1 维度:", y_i.shape)

plt.figure(figsize=(6, 5))
plt.scatter(
    Y[:, 0], 
    Y[:, 1], 
    c=labels, 
    cmap='viridis', 
    s=40, 
    edgecolors='k'
)
plt.title("PCA 降维至 2 维后的数据可视化")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)
plt.show()
```
输出结果：
```
原始数据 x 维度: (200, 5)
标准化后各维度均值: [ 0.  0.  0. -0. -0.]
降维后 Y 维度: (200, 2)
映射矩阵 W 维度: (5, 2)

单个样本 x_1 降维验证：
原始 x_1 维度: (5,) → 降维后 y_1 维度: (1, 2)
```
![pca](pca.png)

### 应用：特征人脸方法
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA # 也可手动实现PcA，此处用库简化核心逻辑
# 1. 加载并预处理人脸数据集
# 加载olivetti人脸数据集（400张64x64灰度人脸，40人x每人10张）
dataset = fetch_olivetti_faces(shuffle=True,random_state=42)
faces=dataset.data # 原始数据：（400，4096）→ 400个样本，每个样本64x64=4096维
labels=dataset.target # 人脸标签（对应40个人)
n_samples,n_features = faces.shape # n_samples=40e,n_features=64x64=4096
image_shape=(64,64) # 人脸图像原始尺寸
print(f"人脸数据集维度：{faces.shape}({n_samples}张人脸，每张展平为{n_features}维向量)")

# 数据中心化（减去均值脸，对应PCA“各维度均值为0”的前提）
mean_face=np.mean(faces,axis=0)  #计算均值脸（4096维)
faces_centered=faces-mean_face # 中心化后的人脸数据

# 2. 核心：求解特征人脸（基于PCA）
# 设定降维后维度l（保留的主成分数，即特征人脸数）
l=100#可调整，1越小降维越明显，保留信息越少
pca = PCA(n_components=l,random_state=42)
faces_pca=pca.fit_transform(faces_centered) # 人脸投影到特征人脸空间（400xl)

# 提取特征人脸（PCA的。components_对应主成分，即特征人脸）
eigenfaces=pca.components_#特征人脸：（l，4096)→每个主成分是一张64x64的特征人脸
print(f"特征人脸矩阵维度：{eigenfaces.shape}({l}张特征人脸，每张展平为{n_features}维)")
print(f"前{l}个主成分累计方差贡献率：{np.round(pca.explained_variance_ratio_.sum(),4)*100}%")

# 3. 人脸投影与重建
# 3.1 人脸投影：将中心化人脸映射到特征人脸空间（低维表示）
# faces_pca=faces_centered·eigenfaces.T→等价于pca.fit_transform的结果
face_idx=0 # 选择第1张人脸演示
face_original=faces[face_idx] #原始人脸（4096维)
face_centered=faces_centered[face_idx] # 中心化人脸
face_proj=faces_pca[face_idx] # 投影后的低维表示（l维)
print(f"\n单张人脸投影维度：原始{n_features}维→低维{l}维")
# 3.2人脸重建：从低维空间还原回原始维度
face_recon = np.dot(face_proj,eigenfaces)+ mean_face # 还原+均值脸
face_recon = np.clip(face_recon,0,1) # 限制像素值在0~1（避免异常值)

# 4. 可视化原图与重建图
plt.figure(figsize=(8,4))

# 原始人脸
plt.subplot(1, 2, 1)
plt.imshow(face_original.reshape(image_shape), cmap='gray')
plt.title("Original Face")
plt.axis("off")

# 重建人脸
plt.subplot(1, 2, 2)
plt.imshow(face_recon.reshape(image_shape), cmap='gray')
plt.title(f"Reconstructed Face (l={l})")
plt.axis("off")

plt.tight_layout()
plt.show()
```
结果：
![eigenface](eigenface.png)

# 潜在语义分析LSA
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# 1. 加载并预处理文木数据集
# 加载20个新闻组的子集（简化计算，聚焦核心语义）
categories = ['sci.space','comp.graphics','rec.sport.baseball']
dataset = fetch_20newsgroups(
            subset='all',categories=categories,shuffle=True,random_state=42,
            remove=('headers','footers','quotes') # 移除无关内容，聚焦正文
)
docs = dataset.data # 文档集合（共n篇文档）
labels = dataset.target # 文档标签（对应3个类别）
n_docs = len(docs)
print(f"文档数量：{n_docs}，类别数：{len(categories)}")
print(f"类别名称：{dataset.target_names}")

# 构建TF-IDF文档-术语矩阵（核心输入：X∈R^(n_docs × n_terms)）
# 去停用词+限制词汇量，避免维度过高

tfidf = TfidfVectorizer(
        stop_words='english',# 移除英文停用词(the/a/an等)
        max_features=2000, # 保留前2000个高频词，控制维度
        max_df=0.95, # 过滤出现频率>95%的词（无区分度）
        min_df=2 # 过滤出现次数<2的词（稀有词）
)
X = tfidf.fit_transform(docs).toarray()#文档-术语矩阵(n_docs × n_terms)
n_terms =X.shape[1]
print(f"\n文档-术语矩阵X维度：n_docs × n_terms={X.shape}")
print(f"词汇表大小(特征数）：{n_terms}")

# 2．LSA核心：SVD奇异值分解
# LSA对TF-IDF矩阵做SVD分解：X = U · ∑ · V^T
#U ∈ R^(n_docsxn_docs): 文档-主题矩阵
#∑ ∈ R^(min(n_docs,n_terms)×min(n_docs,n_terms)): 奇异值矩阵(对角阵)
#V^T ∈ R^(n_termsxn_terms)：术语-主题矩阵
U,Sigma,Vt = np.linalg.svd(X,full_matrices=False)
print(f"\nSVD分解结果维度：")
print(f"U（文档-主题）：{U.shape},∑（奇异值）：{Sigma.shape},Vt（术语-主题)：{Vt.shape}")

# 3. 语义降维：保留前k个奇异值（核心步骤）
k = 50 # 降维后主题数（k<<n_terms，捕捉核心语义）
# 保留前k个奇异值，构建降维矩阵
U_k = U[:,:k] # 文档-主题降维矩阵（n_docs × k)
Sigma_k = np.diag(Sigma[:k]) # 前k个奇异值对角矩阵（k × k)
Vt_k = Vt[:k,] # 术语-主题降维矩阵（k × n_terms）

# 降维后的文档-语义矩阵：X_lsa=U_k · ∑_k(n_docs × k)
X_lsa = np.dot(U_k,Sigma_k)
# 降维后的术语-语义矩阵：T_lsa=∑_k · Vt_k(k × n_terms)
T_lsa = np.dot(Sigma_k,Vt_k)
print(f"\n降维后维度：")
print(f"文档语义矩阵X_lsa:{X_lsa.shape}(n_docs × k)")
print(f"术语语义矩阵T_lsa:{T_lsa.shape}(k × n_terms)")

# 4. 语义相似度计算（LSA核心应用）------—
# 4.1文档间语义相似度（余弦相似度）
doc_idx1 = 0 # 第1篇文档（航天类）
doc_idx2 = 10 # 第11篇文档（图形类)
doc_idx3 = 50 # 第51篇文档（棒球类)

# 计算降维后文档的余弦相似度
sim_doc1_doc2=cosine_similarity([X_lsa[doc_idx1]],[X_lsa[doc_idx2]])[0][0]
sim_doc1_doc3=cosine_similarity([X_lsa[doc_idx1]],[X_lsa[doc_idx3]])[0][0]
print(f"\n文档语义相似度：")
print(f"文档{doc_idx1}（{dataset.target_names[labels[doc_idx1]]}）与文档{doc_idx2}（{dataset.target_names[labels[doc_idx2]]}）：{sim_doc1_doc2:.4f}")
print(f"文档{doc_idx1}（{dataset.target_names[labels[doc_idx1]]}）与文档{doc_idx3}（{dataset.target_names[labels[doc_idx3]]}）：{sim_doc1_doc3:.4f}")

# 4.2术语间语义相似度（比如“ space"和“graphics"）
vocab=tfidf.get_feature_names_out()#词汇表（术语列表）
term1 ="space"
term2 ="graphics"
term3 ="baseball"

#找到术语对应的索引
def get_term_idx(term, vocab):
    try:
        return np.where(vocab==term)[0][0]
    except:
        return -1

idx1 = get_term_idx(term1,vocab)
idx2 = get_term_idx(term2,vocab)
idx3 =get_term_idx(term3,vocab)
if idx1 != -1 and idx2 != -1 and idx3 != -1:
    # 术语的语义向量：取T_lsa的列
    term_vec1=T_lsa[:,idx1]
    term_vec2 =T_lsa[:,idx2]
    term_vec3=T_lsa[:,idx3]
    sim_term1_term2 =cosine_similarity([term_vec1], [term_vec2])[0][0]
    sim_term1_term3=cosine_similarity([term_vec1], [term_vec3])[0][0]
    print(f"\n术语语义相似度：")
    print(f"{term1} vs {term2}: {sim_term1_term2:.4f}")
    print(f"{term1} vs {term3}: {sim_term1_term3:.4f}")
```
输出结果：
```
文档数量：2954，类别数：3
类别名称：['comp.graphics', 'rec.sport.baseball', 'sci.space']

文档-术语矩阵X维度：n_docs × n_terms=(2954, 2000)
词汇表大小(特征数）：2000

SVD分解结果维度：
U（文档-主题）：(2954, 2000),∑（奇异值）：(2000,),Vt（术语-主题)：(2000, 2000)

降维后维度：
文档语义矩阵X_lsa:(2954, 50)(n_docs × k)
术语语义矩阵T_lsa:(50, 2000)(k × n_terms)

文档语义相似度：
文档0（sci.space）与文档10（rec.sport.baseball）：-0.0446
文档0（sci.space）与文档50（comp.graphics）：0.0556

术语语义相似度：
space vs graphics: -0.0073
space vs baseball: 0.0117
```
# 期望最大化（EM）算法
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# 1. 生成模拟数据（混合高斯分布）
# 设定3个高斯分量的参数
np.random.seed(42) # 固定随机种子，结果可复现
n_components = 3   # 高斯分量数（隐变量类别数）
n_samples = 500 # 总样本数
# 分量1：均值[2,2]，协方差[[1,0],[0,1]]，混合系数0.3
mean1 = np.array([2,2])
cov1 =np.array([[1,0],[0,1]])
data1 = np.random.multivariate_normal(mean1,cov1,int(n_samples * 0.3))
# 分量2：均值[8,3]，协方差[[1,0.5],[0.5,2]]，混合系数0.5
mean2 =np.array([8,3])
cov2 = np.array([[1,0.5],[0.5,2]])
data2 = np.random.multivariate_normal(mean2,cov2,int(n_samples * 0.5))
# 分量3：均值[4,7]，协方差[[2,-0.5],[-0.5,1]]，混合系数0.2
mean3 = np.array([4,7])
cov3 = np.array([[2,-0.5],[-0.5,1]])
data3 = np.random.multivariate_normal(mean3,cov3,int(n_samples * 0.2))
# 合并所有样本（含隐变量：样本所属高斯分量未知）
X = np.vstack([data1,data2,data3])
np.random.shuffle(X) # 打乱样本顺序
n_samples_total = X.shape[0]
n_features = X.shape[1]
print(f"模拟数据维度：{X.shape}（{n_samples_total}个样本，{n_features}维特征）")

# 2. EM算法初始化（GMM参数）
# 初始化混合系数π（满足∑π_k=1)
pi = np.ones(n_components)/ n_components
# 初始化高斯分量均值（随机选样本作为初始均值）
mean = X[np.random.choice(n_samples_total,n_components,replace=False)]
# 初始化高斯分量协方差（单位矩阵）
cov =[np.eye(n_features)for _ in range(n_components)]
# 初始化责任度γ（n_samples × n_components）：γ_ik表示样本i属于分量k的后验概率
gamma = np.zeros((n_samples_total,n_components))

# 迭代参数
max_iter = 200 # 最大迭代次数
tol = 1e-4 # 收敛阀值（参数变化量<tol则终止）
log_likelihoods=[] # 存储每轮对数似然（判断收敛）

# 3. EM核心迭代
for iter_num in range(max_iter):

    # ========== E-step ==========
    pdfs = np.zeros((n_samples_total, n_components))
    for k in range(n_components):
        pdfs[:,k] = pi[k] * multivariate_normal.pdf(X, mean=mean[k], cov=cov[k])

    # 保存未归一化的分母用于 log-likelihood
    gamma_sum = np.sum(pdfs, axis=1) + 1e-12  # avoid division by zero

    # 归一化得到责任度
    gamma = pdfs / gamma_sum[:,None]

    # ===== log-likelihood (must use unnormalized) =====
    log_likelihood = np.sum(np.log(gamma_sum))
    log_likelihoods.append(log_likelihood)

    #print(f"iter {iter_num+1:3d} | log-likelihood = {log_likelihood:.4f}")
    if iter_num > 0 and abs(log_likelihoods[-1]-log_likelihoods[-2]) < tol:
        print(f"EM 收敛于第 {iter_num+1} 轮, 对数似然变化量： {abs(log_likelihoods[-1]-log_likelihoods[-2]):.6f}")
        break

    # ========== M-step ==========
    N_k = gamma.sum(axis=0)

    # 更新均值 μ_k
    for k in range(n_components):
        mean[k] = np.sum(gamma[:,k][:,None] * X, axis=0) / N_k[k]

    # 更新协方差 Σ_k
    for k in range(n_components):
        diff = X - mean[k]
        cov[k] = (gamma[:,k][:,None] * diff).T @ diff / N_k[k]
        cov[k] += 1e-6 * np.eye(n_features)  # jitter

    # 更新混合系数 π_k
    pi = N_k / n_samples_total

print("\n最终参数：")
print("π =", pi)
print("mean =\n", mean)
print("covariance matrices =")
for k in range(n_components):
    print(f"Component {k}:\n{cov[k]}\n")

plt.figure(figsize=(7,5))
plt.plot(log_likelihoods, marker='o')
plt.title("EM Algorithm Log-Likelihood Curve")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.grid(True)
plt.show()

from matplotlib.patches import Ellipse
# 1. 使用责任度得到硬分类
labels = np.argmax(gamma, axis=1)
# 颜色映射表
colors = np.array(['red', 'blue', 'green'])
plt.figure(figsize=(8,6))
# 2. 绘制散点（按标签染色）
plt.scatter(X[:,0], X[:,1], c=colors[labels], s=20, alpha=0.6)
# 3. 绘制每个高斯分量的均值 μ（画成大点）
plt.scatter(mean[:,0], mean[:,1], c='black', s=200, marker='x', label='means')
# 4. 为每个协方差矩阵绘制椭圆（等密度线）
def plot_cov_ellipse(cov, mean, ax, n_std=2.0, **kwargs):
    """绘制协方差椭圆（n_std=2 表示约 95% 区域）"""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:,order]
    angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)

ax = plt.gca()

for k in range(n_components):
    plot_cov_ellipse(
        cov[k],
        mean[k],
        ax,
        n_std=2.0,
        alpha=0.25,
        color=colors[k]
    )

plt.title("GMM Clustering Visualization (with Means and Covariance Ellipses)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.legend()
plt.show()
```
输出结果：
```
模拟数据维度：(500, 2)（500个样本，2维特征）
EM 收敛于第 154 轮, 对数似然变化量： 0.000096

最终参数：
π = [0.50076611 0.43545578 0.06377811]
mean =
 [[2.73358262 4.08420667]
 [8.1037552  2.91961301]
 [7.23390983 3.28381396]]
covariance matrices =
Component 0:
[[2.10363066 2.2291511 ]
 [2.2291511  7.33122449]]

Component 1:
[[0.90769644 0.52201529]
 [0.52201529 2.18915771]]

Component 2:
[[0.43147839 0.46232308]
 [0.46232308 0.60313388]]
```
![EM-likelihood](EM-likelihood.png)
![GMM](GMM.png)

# 深度学习
## 误差反向传播（BP）
```python
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/ (1 +np.exp(-1 * x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s*(np.ones(s.shape)- s)

def mean_square_loss(s,y):
    return np.sum(np.square(s- y) / 2)

def d_mean_square_loss(s, y):
    return s - y

def forward(w1,w2,b1,b2,X,y):
    # 输入层到隐藏层
    y1=np.matmul(X,w1)+b1 # [2，3]
    z1=sigmoid(y1) # [2，3]
    # 隐藏层到输出层
    y2 = np.matmul(z1,w2)+ b2 # [2，2]
    z2 =sigmoid(y2) # [2，2]
    # 求均方差损失
    loss = mean_square_loss(z2,y)
    return y1,z1,y2,z2,loss

def backward_update(epochs,lr=0.01):
    np.random.seed(42)  # 保证每次运行一致，可删
    X = np.random.rand(2, 2)       # 输入：2 个样本，每个 2 维
    y = np.random.randint(0, 2, 2) # 标签：0/1 二分类
    W1 = np.random.randn(2, 3) * 0.1   # 输入层 → 隐藏层 (2×3)
    b1 = np.zeros(3)                   # 隐藏层偏置 (3,)
    W2 = np.random.randn(3, 2) * 0.1   # 隐藏层 → 输出层 (3×2)
    b2 = np.zeros(2)  
    loss_list = []
    epoch_list = []
    #先进行一次前向传播
    y1,z1,y2,z2,loss = forward(W1,W2,b1,b2,X,y)
    for i in range(epochs):
        # 求得隐藏层的学习信号（损失函数导数乘激活函数导数）
        ds2 =d_mean_square_loss(z2,y)*d_sigmoid(y2)
        # 根据上面推导结果式子（2.4不看学习率）--->（学习信号乘隐藏层z1的输出结果），注意形状需要转置
        dW2 =np.matmul(z1.T,ds2)
        # 对隐藏层的偏置梯度求和（式子2.6），注意是对列求和
        db2 = np.sum(ds2,axis=0)
        # 式子（2.5)前两个元素相乘
        dx = np.matmul(ds2,W2.T)
        # 对照式子(2.3)
        ds1=d_sigmoid(y1)* dx
        # 式子(2.5）
        dW1 = np.matmul(X.T,ds1)
        # 对隐藏层的偏置梯度求和（式子2.7），注意是对列求和
        db1 = np.sum(ds1,axis=0)
        # 参数更新
        W1 = W1- lr * dW1
        b1 = b1- lr * db1
        W2 = W2- lr * dW2
        b2 = b2- lr * db2
        y1,z1,y2,z2,loss = forward(W1,W2,b1,b2,X,y)
        loss_list.append(loss)
        epoch_list.append(i)
        #每隔100次打印一次损失
        if i%100==0:
            print('第%d批次'%(i/100))
            print('当前损失为:{:.4f}'.format(loss))
            print(z2)
            #sigmoid激活函数将结果大于0.5的值分为正类，小于0.5的值分为负类
            z2[z2 > 0.5] = 1
            z2[z2 < 0.5] = 0
            print(z2)
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_list, loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()


if __name__=='__main__':
    backward_update(epochs=50001,lr=0.01)
```
损失曲线图：
![loss](loss.png)
