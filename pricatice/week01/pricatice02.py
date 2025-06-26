# 简单留出法体验• 实验材料: 使用你在练习1中生成的包含500名用户的数据集。• 关键指标: 我们将关注 “消费水平” 这一类别特征。请先统计一下在原始数据中，高、中、低三个等级的消费水平各占多少比例。这将是我们的“黄金标准”
# 。• 第一步：进行一次“有风险”的简单随机划分
# • 操作: 使用 sklearn 对你的数据集执行一次常规的 80/20 训练集/测试集划分。• 统计与对比:
# • 计算划分后，训练集中的“消费水平”分布比例。
# • 计算划分后，测试集中的“消费水平”分布比例。
# • 观察思考: 对比训练集、测试集与“黄金标准”的比例，它们一致吗？有多大的偏差？这可能会导致什么问题？

#### 首先计算出来原始数据的 高、中、低的消费比例 ---- 黄金标准
#### 使用sklearn 对数据集进行拆分，分别计算训练集、测试集的比例  ----
#### 对比三个比例的差别

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('user_data.csv')

consumption_counts = df['消费水平'].value_counts(normalize=True) * 100
print("消费水平分布比例:")
print(consumption_counts.map(lambda x: f"{x:.2f}%"))

x = df.drop(columns=['消费水平'])
y = df['消费水平']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

train_consumption_counts = y_train.value_counts(normalize=True) * 100
print("\n训练集消费水平比例")
print(train_consumption_counts.map(lambda x : f"{x:.2f}%"))

test_consumption_counts = y_test.value_counts(normalize=True) * 100
print("\n测试集消费水平比例")
print(test_consumption_counts.map(lambda x : f"{x:.2f}%"))

print("\n 三个维度对比")
# 构造成 DataFrame
comparison = pd.DataFrame({
    '黄金标准': consumption_counts,
    '训练集': train_consumption_counts,
    '测试集': test_consumption_counts
})

# 转置后变成指标为行，消费等级为列
comparison_t = comparison.T

print(comparison_t)