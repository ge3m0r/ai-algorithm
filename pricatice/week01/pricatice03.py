# 分层抽样体验• 第二步：进行一次“高保真”的分层划分
# • 操作：使用 sklearn，再次对原始数据集执行 80/20 划分，但这一次我们要求划分过程必须“尊重”原始的消费水平分布。
# • 再次统计：
# • 重新计算划分后，训练集和测试集中“消费水平”的分布比例。• 加分题：通过 C/C++/Java 实现随机分层采样算法（高频面试题）


import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("user_data.csv")

consumption_counts = df['消费水平'].value_counts(normalize=True) * 100
print("消费水平分布比例:")
print(consumption_counts.map(lambda x: f"{x:.2f}%"))

high_consumption = df[df['消费水平'] == '高']
middle_consumption = df[df['消费水平'] == '中']
low_consumption = df[df['消费水平'] == '低']

train_high, test_high = train_test_split(high_consumption, test_size=0.2, random_state=42)
train_middle, test_middle = train_test_split(middle_consumption, test_size=0.2, random_state=42)
train_low, test_low = train_test_split(low_consumption, test_size=0.2, random_state=42)

df_train = pd.concat([train_high, train_middle, train_low])
df_test = pd.concat([test_high, test_middle, test_low])

train_consumption_counts = df_train['消费水平'].value_counts(normalize=True) * 100
print("\n训练集消费水平比例")
print(train_consumption_counts.map(lambda x : f"{x:.2f}%"))

test_consumption_counts = df_test['消费水平'].value_counts(normalize=True) * 100
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