# 将你的用户数据“向量化”• 任务描述： 使用你在练习1中生成的完整数据集。
# • 特征向量化 (Feature Engineering):
# • 类别特征处理： 对所有的类别特征（如性别, 城市, 消费水平）作为文本，使用BGE-m3计算 Embedding（空间坐标）
# • 向量拼接： 将处理好的所有新特征拼接起来，为每一个用户生成一个长长的、纯数字的向量。此时，每个用户都被一个高维向量所唯一表示。
# • 降维与可视化 (Dimensionality Reduction & Visualization): 由于你生成的用户向量维度很高（可能超过10维），无法直接在平面上展示。请使用一种降维算法，例如主成分分析 (PCA)，将高维的用户向量降至二维。
# • 创建一个二维散点图，图上的每一个点代表一个用户，其(x, y)坐标就是由PCA降维后得到的两个新维度。

### BGE-M3 模型使用
from FlagEmbedding import BGEM3FlagModel
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df = pd.read_csv('user_data.csv')

# 初始化模型（支持 GPU 加速）
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

gender_embeddings = model.encode(df['性别'].tolist())['dense_vecs']
city_embeddings = model.encode(df['所在城市'].tolist())['dense_vecs']
consumption_embeddings = model.encode(df['消费水平'].tolist())['dense_vecs']

user_vectors = np.hstack([
    gender_embeddings,
    city_embeddings,
    consumption_embeddings
])

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(user_vectors)
print(reduced_vectors)

plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('User Vectors after PCA')
plt.show()

