## 基于语义的文本分类器
## 目标： 训练一个能够理解文本内涵的文本分类器
## 任务描述： 有以下三种分类器选择
## 1、情感分析，文本是积极的还是消极的
## 2、文本是理性的/客观的还是 感性的/主观的
## 3、用户评论是在咨询反馈，还是在感谢赞美
## 数据准备：为你的文本编写 10-15 条正面样本和负面样本
## 特征提取：将文本通过bge-m3 模型转化为特征向量
## 模型训练和评估： 将你的语义向量作为 x ,，分类标签作为y
## 使用逻辑回归模型训练模型
## 尝试评估你的模型

from FlagEmbedding import BGEM3FlagModel
from sklearn.linear_model import LogisticRegression
import numpy as np

obj_text = [
  "产品的性能稳定，符合我的预期。",
  "这款产品在同类中性价比很高。",
  "设计合理，操作简单，适合新手使用。",
  "功能齐全，能够满足日常工作需求。",
  "耐用性不错，使用一段时间后没有明显问题。",
  "包装完好，物流迅速，服务态度很好。",
  "细节处理到位，做工精细。",
  "电池续航能力强，充电速度快。",
  "使用体验良好，响应速度很快。",
  "产品质量可靠，值得信赖。",
  "提供了完善的用户手册和技术支持。",
  "体积小巧，便于携带和存储。",
  "兼容性强，可以与其他设备无缝连接。",
  "客服专业耐心，解决了我遇到的问题。",
  "售后服务完善，退换货流程便捷。"
]

sub_text = [
  "第一次使用就爱上了这款产品，非常贴心！",
  "外观设计太吸引人了，每次使用都心情愉悦。",
  "感觉它就像为我量身定做的一样，完全满足了我的需求。",
  "用起来特别顺手，感觉生活因为这个产品变得更美好了。",
  "颜值高，性能好，简直是完美的结合。",
  "每次打开包装时都有一种莫名的兴奋感。",
  "真心推荐给身边的朋友，他们都觉得很不错。",
  "它让我第一次感受到科技带来的温暖与关怀。",
  "无论从哪个角度看，它都充满了惊喜。",
  "用过之后才发现，原来生活可以这么简单又幸福。",
  "仿佛有了它，一切烦恼都变得微不足道。",
  "精致的外观和细腻的质感让人爱不释手。",
  "每次看到它都觉得自己的选择太明智了。",
  "它不仅是一件产品，更是我生活中的一部分。",
  "用完之后真的有种“相见恨晚”的感觉。"
]

texts = obj_text + sub_text
labels = [1] * len(obj_text) + [0] * len(sub_text)

print("Loading BGE-M3 model ....")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
embeddings = model.encode(texts, batch_size=5, max_length=128)['dense_vecs']

### 训练回归模型
clf = LogisticRegression(max_iter=1000)
clf.fit(embeddings, labels)

weights = clf.coef_.flatten()
top_k = 10
top_indices = np.argsort(np.abs(weights))[-top_k:][::-1]

test_text = [
  "这款产品设计得太棒了，使用起来让人感到无比愉悦！",
  "根据实验数据，该产品的性能提升了20%，具备明显优势。",
  "我非常喜欢这个功能，它完全符合我的个人需求。",
  "用户反馈表明，70%的人认为此产品优于同类竞品。",
  "看到这么出色的外观设计，我立刻决定购买。"
]

test_labels = [0, 1, 0, 1, 0]
test_embeddings = model.encode(test_text, batch_size=5, max_length=128)['dense_vecs']
test_predictions = clf.predict(test_embeddings)
print("预测结果为：", test_predictions)
sum = 0
for i in range(len(test_predictions)):
  if test_predictions[i] == test_labels[i]:
    sum += 1
print("模型测试准确率为: ", sum / len(test_labels))

accuracy = clf.score(test_embeddings, test_labels)
print("模型测试准确率为: ", accuracy)
























