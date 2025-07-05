## 基于k-means 的用户分群与画像洞察
## 无监督学习可以帮我从海量用户中自动发现部落，用户群体画像
## 任务描述：
## 1、数据加载 使用上节课用户数据集
## 2、执行聚类， 使用kmeans 算法，将数据集划分成 K 个群体
## 3、对于算法划分出来的每个用户群体，计算群个体质心
## 扎厚道离该群体质心最近的真实用户
## 将3个最具代表性的用户完整展示
### 分析与解读：这 3 位用户代表的可能是什么样的用户

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# 加载数据
df = pd.read_csv("user_data.csv")

# 假设 "用户" 是索引列，删除该列
df.drop(columns=["用户"], inplace=True, errors='ignore')  # 防止列不存在时报错

# 查看数据类型
print(df.dtypes)

# 检查是否有缺失值
print(df.isnull().sum())

# 填充缺失值（如果有）
df.fillna(0, inplace=True)

# 分离数值型和类别型列
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# 创建预处理器：对类别型变量 one-hot 编码，数值型标准化
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 构建聚类流水线
kmeans = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('cluster', KMeans(n_clusters=3, random_state=42))
])

# 训练模型并预测
X = df
kmeans.fit(X)
labels = kmeans.predict(X)

# 获取转换后的特征矩阵（用于计算距离）
X_processed = kmeans.named_steps['preprocessor'].transform(X)

# 如果是稀疏矩阵，转成密集数组
from scipy.sparse import csr_matrix, issparse

if issparse(X_processed):
    X_processed = X_processed.toarray()

# 获取聚类中心
centers = kmeans.named_steps['cluster'].cluster_centers_

# 确保 centers 是 numpy.ndarray 类型
if isinstance(centers, csr_matrix):
    centers = centers.toarray()

# 计算每个样本到各个质心的距离
distances = cdist(X_processed, centers, 'euclidean')

# 找出离每个质心最近的样本索引
closest_indices = distances.argmin(axis=0)

# 获取这些用户在原始 DataFrame 中的信息
closest_users = df.iloc[closest_indices]

print("Closest users to each cluster center:\n", closest_users)

