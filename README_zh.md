# 基于多头注意力机制的文本相似度模型

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.13+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## 📋 项目介绍

本项目实现了一个基于**多头注意力机制（Multi-Head Attention）**的文本相似度计算模型。该模型能够有效地捕获文本序列中的语义信息，并计算两个文本之间的相似度分数。

### 🌟 主要特性

- **多头注意力机制**：采用 Transformer 架构的核心组件，能够从不同的表示子空间捕获文本特征
- **灵活的配置系统**：支持 YAML/JSON 配置文件，方便进行实验管理
- **完整的训练流程**：包含训练、验证、早停、检查点保存等功能
- **易用的命令行接口**：支持丰富的命令行参数，方便快速实验
- **详细的中文注释**：代码包含完整的中文文档和注释，便于学习和理解

## 🏗️ 项目结构

```
.
├── configs/                 # 配置文件目录
│   └── default.yaml        # 默认配置文件
├── src/                    # 源代码目录
│   ├── configs/           # 配置管理模块
│   │   ├── __init__.py
│   │   └── config.py      # 配置类定义
│   ├── data/              # 数据处理模块
│   │   ├── __init__.py
│   │   └── dataset.py     # 数据集和数据加载器
│   ├── models/            # 模型定义模块
│   │   ├── __init__.py
│   │   ├── attention.py   # 多头注意力实现
│   │   └── similarity_model.py  # 文本相似度模型
│   └── utils/             # 工具模块
│       ├── __init__.py
│       ├── trainer.py     # 训练器实现
│       └── utils.py       # 通用工具函数
├── scripts/               # 脚本目录
│   └── train.py          # 训练脚本
├── tutorial/             # 教程和原始代码
│   └── mha-lstm/        # 原始 notebook 和数据
├── requirements.txt      # 项目依赖
└── README_zh.md         # 中文说明文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd <project-directory>

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

项目使用 STSbenchmark 数据集，数据文件应放置在 `tutorial/mha-lstm/data/` 目录下：
- `sts-kaggle-train.csv`：训练数据
- `sts-kaggle-test.csv`：验证/测试数据

数据格式示例：
```csv
id,sentence_a,sentence_b,similarity
0,"A kitten is playing with a toy.","A kitten is playing with a blue rope toy.",4.4
1,"A dog is running in a field.","A white and brown dog runs in a field.",2.83
```

### 3. 开始训练

#### 使用默认配置
```bash
python scripts/train.py
```

#### 使用自定义配置文件
```bash
python scripts/train.py --config configs/default.yaml
```

#### 常用命令行参数
```bash
# 调整批次大小和学习率
python scripts/train.py --batch-size 64 --learning-rate 0.001

# 使用预训练词嵌入
python scripts/train.py --use-pretrained-embeddings --embeddings-name glove.6B.300d

# 指定实验名称和输出目录
python scripts/train.py --experiment-name my_experiment --output-dir experiments

# 使用 GPU 训练
python scripts/train.py --device cuda

# 仅评估模式
python scripts/train.py --eval-only --resume checkpoints/best_model.pt
```

## 📊 模型架构

详细的架构图请查看 [架构文档](docs/architecture.md)

### 整体架构图

```
输入文本对 (Text A, Text B)
    ↓
分词器 (Tokenizer)
    ↓
词嵌入层 (Embedding Layer)
    ↓
位置编码 (Positional Encoding)
    ↓
多头注意力层 (Multi-Head Attention) × N
    ↓
池化层 (Pooling Layer)
    ↓
输出投影 (Output Projection)
    ↓
余弦相似度 (Cosine Similarity)
    ↓
相似度分数 (Similarity Score)
```

### 核心组件说明

#### 1. 多头注意力机制 (Multi-Head Attention)

多头注意力是模型的核心组件，其计算过程如下：

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**关键参数：**
- `embed_dim`: 嵌入维度 (默认 256)
- `num_heads`: 注意力头数 (默认 8)
- `dropout`: Dropout 概率 (默认 0.1)

#### 2. 文本编码器 (Text Encoder)

编码器将输入文本序列转换为固定维度的向量表示：

```python
输入序列 → 词嵌入 → 位置编码 → 多层注意力 → 池化 → 文本表示
```

**池化策略：**
- `mean`: 平均池化（默认）
- `max`: 最大池化
- `cls`: 使用 [CLS] 标记的表示

#### 3. 相似度计算

使用余弦相似度计算两个文本表示之间的相似程度：

```python
similarity = cosine_similarity(embedding_a, embedding_b)
```

## 🔧 配置说明

### 模型配置 (model)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| embed_dim | int | 256 | 嵌入维度 |
| num_heads | int | 8 | 注意力头数 |
| num_layers | int | 2 | 编码器层数 |
| dropout | float | 0.1 | Dropout 概率 |
| pooling_strategy | str | mean | 池化策略 |
| max_seq_len | int | 200 | 最大序列长度 |

### 训练配置 (training)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| batch_size | int | 32 | 批次大小 |
| learning_rate | float | 1e-4 | 学习率 |
| num_epochs | int | 50 | 训练轮数 |
| optimizer | str | adam | 优化器类型 |
| early_stopping | bool | true | 是否使用早停 |
| patience | int | 10 | 早停耐心值 |

### 数据配置 (data)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| train_path | str | - | 训练数据路径 |
| val_path | str | - | 验证数据路径 |
| tokenizer | str | basic_english | 分词器类型 |
| normalize_scores | bool | true | 是否归一化分数 |
| score_range | list | [0, 5] | 分数范围 |

## 📈 实验结果

### 评估指标

- **Pearson 相关系数**: 衡量预测值和真实值之间的线性相关性
- **Spearman 相关系数**: 衡量预测值和真实值之间的单调相关性
- **MSE (均方误差)**: 预测误差的平方均值
- **MAE (平均绝对误差)**: 预测误差的绝对值均值

### 训练日志

训练过程中会生成以下文件：
- `output/exp_*/config.yaml`: 实验配置
- `output/exp_*/train.log`: 训练日志
- `output/exp_*/checkpoints/`: 模型检查点
- `output/exp_*/history.json`: 训练历史
- `output/exp_*/report.md`: 实验报告

### 可视化

如果安装了 TensorBoard，可以查看训练曲线：
```bash
tensorboard --logdir runs
```

## 🎯 应用场景

1. **文本匹配**：判断两个文本是否表达相同含义
2. **问答系统**：匹配问题和答案的相关性
3. **文档检索**：根据查询找到最相关的文档
4. **重复检测**：识别重复或相似的内容
5. **语义搜索**：基于语义相似度的搜索系统

## 🔍 设计模式说明

### 1. 工厂模式 (Factory Pattern)
在配置管理中使用工厂模式创建不同的配置对象：
```python
config = Config.from_file("config.yaml")  # 从文件创建
config = Config.from_dict(config_dict)    # 从字典创建
```

### 2. 策略模式 (Strategy Pattern)
池化策略的实现采用策略模式，支持不同的池化方法：
```python
if self.pooling_strategy == 'mean':
    return sequence.mean(dim=1)
elif self.pooling_strategy == 'max':
    return sequence.max(dim=1)[0]
```

### 3. 模板方法模式 (Template Method Pattern)
训练器类定义了训练流程的模板，子类可以重写特定步骤：
```python
def train(self):
    for epoch in range(num_epochs):
        self._train_epoch()
        self._validate()
        self._save_checkpoint()
```

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- 感谢 PyTorch 团队提供的深度学习框架
- 感谢 Hugging Face 团队的 Transformers 库提供的灵感
- 感谢所有贡献者的努力

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至 [your-email@example.com]

---

**注意**：本项目仅供学习和研究使用，商业使用请确保遵守相关许可证要求。