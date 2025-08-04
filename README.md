# 基于多头注意力机制的文本相似度模型

> [中文文档](README_zh.md) | English

## 项目介绍

本项目实现了一个基于多头注意力机制（Multi-Head Attention）的文本相似度计算模型。该模型使用 Transformer 架构的核心组件，能够有效地捕获文本序列中的语义信息，并计算两个文本之间的相似度分数。

## 主要特性

- 多头注意力机制实现
- 灵活的配置系统
- 完整的训练流程
- 易用的命令行接口
- 详细的中文注释和文档

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 使用默认配置训练
python scripts/train.py

# 使用自定义配置
python scripts/train.py --config configs/default.yaml
```

更多详细信息请参阅[中文文档](README_zh.md)。
