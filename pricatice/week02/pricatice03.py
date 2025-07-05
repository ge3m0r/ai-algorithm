## 基于pca 的用户数据可视化探索
### 高维数据对人类来说极度抽象，降维和可视化是我们更加直观查看
## 任务描述：
## 1. 数据加载
## 2、执行pca 降维
## 3、可视化 绘制二维散点图和三位散点图
## 在散点图中用不同颜色标注出k-means 的分出的不同用户群体

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于3D绘图

# 加载数据
df = pd.read_csv("user_data.csv")

# 删除“用户”列
df.drop(columns=["用户"], inplace=True, errors='ignore')

# 填充缺失值
df.fillna(0, inplace=True)

# 分离数值型和类别型列
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# 预处理 + KMeans 流水线
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

kmeans_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('cluster', KMeans(n_clusters=3, random_state=42))
])

# 拟合模型并预测
X = df
kmeans_pipeline.fit(X)
labels = kmeans_pipeline.predict(X)

# 获取处理后的特征矩阵（用于PCA）
X_processed = kmeans_pipeline.named_steps['preprocessor'].transform(X).toarray()

# PCA 降维到2维
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_processed)

# PCA 降维到3维
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_processed)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.title('PCA 2D Visualization of User Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                     c=labels, cmap='viridis', s=50, alpha=0.7)

plt.title('PCA 3D Visualization of User Clusters')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.colorbar(scatter, label='Cluster')
plt.show()