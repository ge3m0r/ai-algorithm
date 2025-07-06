# AI Algorithm Learning Project 🚀

> 从传统机器学习到现代大模型的完整AI算法学习项目

## 📖 项目简介

本项目是一个系统性的AI算法学习平台，旨在帮助学习者从基础的传统机器学习算法逐步过渡到现代大模型技术。项目采用周计划学习模式，每周包含理论知识和实践练习，确保理论与实践相结合。

### 🎯 学习目标
- 掌握传统机器学习算法的核心原理
- 理解现代大模型的技术架构
- 学习工业级AI系统的部署方法
- 通过实际项目提升算法应用能力

## 📁 项目结构

### 整体架构
```
ai-algorithm/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包列表
├── .gitignore                  # Git忽略文件配置
├── LICENSE                     # 开源许可证
├── docs/                       # 项目文档目录
│   ├── getting-started.md      # 入门指南
│   ├── algorithms/             # 算法详细说明
│   └── deployment/             # 部署指南
├── practice/                   # 练习内容目录
│   ├── week01/                 # 第一周：基础算法与数据处理
│   ├── week02/                 # 第二周：进阶算法与应用
│   ├── week03/                 # 第三周：深度学习基础
│   ├── week04/                 # 第四周：大模型入门
│   └── ...                     # 更多周次内容
├── examples/                   # 完整示例项目
│   ├── recommendation-system/  # 推荐系统示例
│   ├── nlp-project/           # 自然语言处理项目
│   └── computer-vision/       # 计算机视觉项目
├── utils/                      # 通用工具函数
│   ├── data_loader.py         # 数据加载工具
│   ├── visualization.py       # 可视化工具
│   └── evaluation.py          # 评估指标工具
└── tests/                      # 单元测试
    ├── test_algorithms.py     # 算法测试
    └── test_utils.py          # 工具函数测试
```

### 每周学习内容结构
```
weekXX/                         # 第XX周学习内容
├── XX.md                      # 理论知识文档
├── slides/                    # 演示文稿（可选）
│   ├── lecture01.pdf
│   └── lecture02.pdf
├── practice/                  # 练习目录
│   ├── practice01.py         # 练习1：基础实现
│   ├── practice02.py         # 练习2：进阶应用
│   ├── practice03.py         # 练习3：综合项目
│   └── ...                   # 更多练习
├── answer/                    # 参考答案目录
│   ├── solution01.py         # 练习1答案
│   ├── solution02.py         # 练习2答案
│   ├── solution03.py         # 练习3答案
│   ├── README.md             # 答案说明文档
│   └── bonus/                # 扩展练习答案
├── data/                      # 数据文件目录
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后数据
│   └── sample/               # 示例数据
├── models/                    # 训练好的模型（可选）
│   ├── model01.pkl
│   └── model02.h5
├── results/                   # 实验结果
│   ├── figures/              # 图表输出
│   ├── logs/                 # 训练日志
│   └── reports/              # 实验报告
└── config/                    # 配置文件
    ├── config.yaml           # 主配置文件
    └── hyperparameters.json  # 超参数配置
```

### 当前已实现内容

#### Week 01: 基础算法与数据处理
```
week01/
├── 01.md                     # 机器学习基础理论
├── answer/                   # 练习答案
│   ├── O1_reservoir_sampling_demo.py      # 水库抽样演示
│   ├── R1_generate_user_profiles.py       # 用户画像生成
│   ├── R2_split_and_analyze_consumption.py # 消费数据分析
│   ├── R3_stratified_split_consumption.py # 分层抽样策略
│   ├── R4_user_vectorization_and_visualization.py # 用户向量化
│   └── README.md             # 答案说明
├── practice01.py             # 练习1：基础算法实现
├── practice02.py             # 练习2：数据处理
├── practice03.py             # 练习3：模型训练
├── practice04.py             # 练习4：结果分析
└── user_data.csv             # 用户数据样本
```

#### Week 02: 进阶算法与应用
```
week02/
├── 02.md                     # 深度学习基础理论
├── practice01.py             # 练习代码
├── practice02.py             # 练习代码
├── practice03.py             # 练习代码
└── user_data.csv             # 数据文件
```

### 文件命名规范

#### 理论文档
- `XX.md`: 第XX周的理论知识文档
- 使用Markdown格式，支持数学公式和代码块

#### 练习文件
- `practiceXX.py`: 第XX个练习的代码文件
- 文件名应简洁明了，体现练习内容

#### 答案文件
- `O1_*.py`: 示例代码（O = Overview）
- `R1_*.py`: 参考答案（R = Reference）
- 文件名应包含算法或功能描述

#### 数据文件
- `*_data.csv`: 数据文件
- `*_config.yaml`: 配置文件
- `*_model.pkl`: 模型文件

### 扩展计划

#### 即将添加的内容
- **Week 03**: 深度学习基础（CNN、RNN）
- **Week 04**: 大模型入门（Transformer、BERT）
- **Week 05**: 工业级部署（Docker、Kubernetes）
- **Week 06**: 模型优化（量化、剪枝）
- **Week 07**: 多模态学习（图像+文本）
- **Week 08**: 强化学习基础

#### 工具和框架
- **MLflow**: 实验跟踪
- **Weights & Biases**: 模型监控
- **Streamlit**: 快速原型开发
- **FastAPI**: API服务开发

## 🛠️ 技术栈

### 核心语言
- **Python 3.8+**: 主要编程语言

### 数据处理
- **Pandas**: 数据分析和处理
- **NumPy**: 数值计算
- **Matplotlib/Seaborn**: 数据可视化

### 机器学习
- **Scikit-learn**: 传统机器学习算法
- **TensorFlow/PyTorch**: 深度学习框架

### 算法实现
- **水库抽样 (Reservoir Sampling)**: 大数据流处理
- **用户画像生成**: 用户行为分析
- **分层抽样**: 数据采样策略
- **向量化技术**: 特征工程

## 📚 学习路径

### 第一周：基础算法与数据处理
- **理论学习**: 机器学习基础概念
- **实践练习**:
  - 水库抽样算法实现
  - 用户画像生成
  - 消费数据分析
  - 分层抽样策略
  - 用户向量化与可视化

### 第二周：进阶算法与应用
- **理论学习**: 深度学习基础
- **实践练习**: 待补充具体内容

## 🚀 快速开始

### 环境要求
```bash
Python >= 3.8
pip >= 20.0
```

### 安装依赖
```bash
# 克隆项目
git clone https://github.com/your-username/ai-algorithm.git
cd ai-algorithm

# 安装依赖包
pip install -r requirements.txt
```

### 运行练习
```bash
# 运行第一周练习1
python practice/week01/practice01.py

# 运行答案示例
python practice/week01/answer/O1_reservoir_sampling_demo.py
```

## 📖 学习指南

### 1. 理论学习
- 每周开始前先阅读对应的 `.md` 文档
- 理解核心概念和算法原理
- 做好笔记，记录关键知识点

### 2. 实践练习
- 独立完成练习代码
- 对比答案，理解实现差异
- 尝试优化和改进算法

### 3. 项目应用
- 将学到的算法应用到实际项目中
- 参与开源项目，贡献代码
- 建立个人项目组合

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目！

### 贡献方式
1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 学习笔记

### 重要概念
- **水库抽样**: 用于处理大数据流的随机采样算法
- **用户画像**: 基于用户行为数据的特征提取
- **分层抽样**: 保证样本代表性的抽样策略
- **向量化**: 将非数值数据转换为数值向量的过程

### 最佳实践
- 始终验证数据的完整性和准确性
- 使用适当的评估指标
- 注意算法的可扩展性
- 保持代码的可读性和可维护性

## 📞 联系方式

- **项目维护者**: [您的姓名]
- **邮箱**: [您的邮箱]
- **GitHub**: [您的GitHub主页]

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和学习者！

---

⭐ 如果这个项目对您有帮助，请给它一个星标！
