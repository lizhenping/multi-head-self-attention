
# 系统架构图

## 1. 整体架构

```mermaid
flowchart TB
    subgraph input ["输入层"]
        A[文本A] 
        B[文本B]
    end
    
    subgraph preprocess ["预处理层"]
        C[分词器]
        D[词汇表]
    end
    
    subgraph embedding ["嵌入层"]
        E[词嵌入]
        F[位置编码]
    end
    
    subgraph encoder ["编码器层"]
        G[多头注意力]
        H[前馈网络]
        I[层归一化]
        J[残差连接]
    end
    
    subgraph pooling ["池化层"]
        K[池化策略]
        K1[平均池化]
        K2[最大池化]
        K3[CLS池化]
    end
    
    subgraph output ["输出层"]
        L[文本表示A]
        M[文本表示B]
        N[余弦相似度]
        O[相似度分数]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> G
    J --> K
    K --> K1
    K --> K2
    K --> K3
    K --> L
    K --> M
    L --> N
    M --> N
    N --> O
```

## 2. 多头注意力机制详解

```mermaid
flowchart LR
    A[输入序列X] --> B[线性变换Q]
    A --> C[线性变换K]
    A --> D[线性变换V]
    
    B --> E[多头分割Q]
    C --> F[多头分割K]
    D --> G[多头分割V]
    
    E --> H[缩放点积注意力]
    F --> H
    G --> H
    
    H --> I[拼接多头结果]
    I --> J[输出投影]
```

## 3. 训练流程图

```mermaid
flowchart TD
    A[开始] --> B[加载配置]
    B --> C[准备数据]
    C --> D[构建词汇表]
    D --> E[创建数据加载器]
    E --> F[初始化模型]
    F --> G[设置优化器]
    G --> H[设置损失函数]
    
    H --> I{开始训练}
    I --> J[前向传播]
    J --> K[计算损失]
    K --> L[反向传播]
    L --> M[更新参数]
    M --> N[记录指标]
    
    N --> O{验证时机?}
    O -->|是| P[验证模型]
    O -->|否| Q{结束epoch?}
    P --> R[计算验证指标]
    R --> S{是否最佳?}
    S -->|是| T[保存最佳模型]
    S -->|否| U{早停检查}
    T --> U
    U -->|继续| Q
    U -->|停止| V[结束训练]
    
    Q -->|否| J
    Q -->|是| W{所有epoch完成?}
    W -->|否| I
    W -->|是| V
    
    V --> X[生成报告]
    X --> Y[结束]
```

## 4. 数据流程图

```mermaid
flowchart LR
    A[CSV文件] --> B[分词处理]
    B --> C[构建词汇表]
    C --> D[序列编码]
    D --> E[填充截断]
    E --> F[张量转换]
    F --> G[批次处理]
    G --> H[模型输入]
```

## 5. 模块依赖关系

```mermaid
flowchart TD
    A[train.py] --> B[config.py]
    A --> C[similarity_model.py]
    A --> D[dataset.py]
    A --> E[trainer.py]
    A --> F[utils.py]
    
    C --> G[attention.py]
    E --> C
    E --> D
    E --> F
    
    B -.-> E
    B -.-> D
```

## 6. 类关系图

```mermaid
classDiagram
    class MultiHeadAttention {
        +embed_dim: int
        +num_heads: int
        +forward()
    }
    
    class TextEncoder {
        +embed_dim: int
        +num_layers: int
        +forward()
    }
    
    class TextSimilarityModel {
        +pooling_strategy: str
        +encode_text()
        +compute_similarity()
    }
    
    class Trainer {
        +model: Module
        +train()
        +validate()
    }
    
    class Config {
        +model_config: dict
        +from_file()
        +validate()
    }
    
    TextSimilarityModel --> TextEncoder
    TextEncoder --> MultiHeadAttention
    Trainer --> TextSimilarityModel
    Trainer --> Config
```

## 7. 注意力计算序列图

```mermaid
sequenceDiagram
    participant I as 输入序列
    participant L as 线性变换
    participant A as 注意力计算
    participant O as 输出层
    
    I->>L: 输入嵌入
    L->>L: 生成Q,K,V
    L->>A: 传递Q,K,V
    A->>A: 计算注意力权重
    A->>A: 加权求和
    A->>O: 输出结果
    O->>O: 线性投影
```

## 架构说明

### 主要组件
1. **输入层**: 处理文本A和文本B的输入
2. **预处理层**: 分词和词汇表构建
3. **嵌入层**: 词嵌入和位置编码
4. **编码器层**: 多头注意力和前馈网络
5. **池化层**: 多种池化策略选择
6. **输出层**: 计算文本相似度

### 关键特性
- 支持多种池化策略（平均、最大、CLS）
- 采用残差连接和层归一化
- 可配置的多头注意力机制
- 灵活的训练和验证流程

### 使用方式
1. 配置模型参数
2. 准备训练数据
3. 执行训练流程
4. 评估模型性能
5. 生成相似度预测
