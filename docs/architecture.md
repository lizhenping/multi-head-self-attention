
# 系统架构图

## 1. 整体架构

```mermaid
flowchart TB
    subgraph input ["输入层"]
        A["文本A"] 
        B["文本B"]
    end
    
    subgraph preprocess ["预处理层"]
        C["分词器 Tokenizer"]
        D["词汇表 Vocabulary"]
    end
    
    subgraph embedding ["嵌入层"]
        E["词嵌入 Word Embedding"]
        F["位置编码 Positional Encoding"]
    end
    
    subgraph encoder ["编码器层"]
        G["多头注意力 Multi-Head Attention"]
        H["前馈网络 Feed Forward"]
        I["层归一化 Layer Norm"]
        J["残差连接 Residual Connection"]
    end
    
    subgraph pooling ["池化层"]
        K["池化策略 Pooling Strategy"]
        K1["平均池化 Mean"]
        K2["最大池化 Max"]
        K3["CLS池化"]
    end
    
    subgraph output ["输出层"]
        L["文本表示A"]
        M["文本表示B"]
        N["余弦相似度 Cosine Similarity"]
        O["相似度分数"]
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
    subgraph input_section ["输入"]
        A["输入序列 X"]
    end
    
    subgraph linear_transform ["线性变换"]
        B["Q = XWQ"]
        C["K = XWK"]
        D["V = XWV"]
    end
    
    subgraph multi_head ["多头分割"]
        E["Q1, Q2, ..., Qh"]
        F["K1, K2, ..., Kh"]
        G["V1, V2, ..., Vh"]
    end
    
    subgraph attention ["缩放点积注意力"]
        H["Attention_i = softmax(QiKi^T/√dk)Vi"]
    end
    
    subgraph concat ["拼接与投影"]
        I["Concat(head1, ..., headh)"]
        J["MultiHead = Concat·WO"]
    end
    
    A --> B
    A --> C
    A --> D
    B --> E
    C --> F
    D --> G
    E --> H
    F --> H
    G --> H
    H --> I
    I --> J
```

## 3. 训练流程图

```mermaid
flowchart TD
    A["开始"] --> B["加载配置"]
    B --> C["准备数据"]
    C --> D["构建词汇表"]
    D --> E["创建数据加载器"]
    E --> F["初始化模型"]
    F --> G["设置优化器"]
    G --> H["设置损失函数"]
    
    H --> I{"开始训练"}
    I --> J["前向传播"]
    J --> K["计算损失"]
    K --> L["反向传播"]
    L --> M["更新参数"]
    M --> N["记录指标"]
    
    N --> O{"验证时机?"}
    O -->|是| P["验证模型"]
    O -->|否| Q{"结束epoch?"}
    P --> R["计算验证指标"]
    R --> S{"是否最佳?"}
    S -->|是| T["保存最佳模型"]
    S -->|否| U{"早停检查"}
    T --> U
    U -->|继续| Q
    U -->|停止| V["结束训练"]
    
    Q -->|否| J
    Q -->|是| W{"所有epoch完成?"}
    W -->|否| I
    W -->|是| V
    
    V --> X["生成报告"]
    X --> Y["结束"]
```

## 4. 数据流程图

```mermaid
flowchart LR
    subgraph raw_data ["原始数据"]
        A["CSV文件"]
        A1["sentence_a"]
        A2["sentence_b"]
        A3["similarity"]
    end
    
    subgraph preprocessing ["数据预处理"]
        B["分词"]
        C["构建词汇表"]
        D["序列编码"]
        E["填充/截断"]
    end
    
    subgraph tensor_convert ["张量转换"]
        F["input_ids"]
        G["attention_mask"]
        H["labels"]
    end
    
    subgraph batch_process ["批处理"]
        I["DataLoader"]
        J["批次数据"]
    end
    
    A --> A1
    A --> A2
    A --> A3
    A1 --> B
    A2 --> B
    B --> C
    C --> D
    D --> E
    E --> F
    E --> G
    A3 --> H
    F --> I
    G --> I
    H --> I
    I --> J
```

## 5. 模块依赖关系

```mermaid
flowchart TD
    subgraph script_layer ["脚本层"]
        A["train.py"]
    end
    
    subgraph config_layer ["配置层"]
        B["config.py"]
    end
    
    subgraph model_layer ["模型层"]
        C["attention.py"]
        D["similarity_model.py"]
    end
    
    subgraph data_layer ["数据层"]
        E["dataset.py"]
    end
    
    subgraph tool_layer ["工具层"]
        F["trainer.py"]
        G["utils.py"]
    end
    
    A --> B
    A --> D
    A --> E
    A --> F
    A --> G
    
    D --> C
    F --> D
    F --> E
    F --> G
    
    B -.-> F
    B -.-> E
```

## 6. 类图

```mermaid
classDiagram
    class MultiHeadAttention {
        +int embed_dim
        +int num_heads
        +int head_dim
        +float scale
        +Linear q_linear
        +Linear k_linear
        +Linear v_linear
        +Linear out_proj
        +Dropout dropout
        +forward(query, key, value, mask)
    }
    
    class TextEncoder {
        +int embed_dim
        +int num_layers
        +Embedding embedding
        +PositionalEncoding position_encoding
        +ModuleList attention_layers
        +LayerNorm final_norm
        +forward(input_ids, attention_mask)
    }
    
    class TextSimilarityModel {
        +str pooling_strategy
        +TextEncoder encoder
        +Sequential output_proj
        +pool_sequence(sequence, attention_mask)
        +encode_text(input_ids, attention_mask)
        +forward(input_ids_a, input_ids_b, ...)
        +compute_similarity(embedding_a, embedding_b)
    }
    
    class Trainer {
        +Module model
        +Config config
        +DataLoader train_loader
        +DataLoader val_loader
        +Optimizer optimizer
        +LRScheduler scheduler
        +Loss criterion
        +train()
        +_train_epoch()
        +_validate()
        +_save_checkpoint()
        +load_checkpoint()
        +predict()
    }
    
    class Config {
        +ModelConfig model
        +TrainingConfig training
        +DataConfig data
        +LoggingConfig logging
        +from_file(file_path)
        +from_dict(config_dict)
        +to_dict()
        +save(file_path)
        +update(updates)
        +validate()
    }
    
    TextSimilarityModel --> TextEncoder
    TextEncoder --> MultiHeadAttention
    Trainer --> TextSimilarityModel
    Trainer --> Config
```

## 7. 注意力计算流程

```mermaid
sequenceDiagram
    participant Input as 输入序列
    participant Linear as 线性变换层
    participant MHA as 多头注意力
    participant Scale as 缩放
    participant Softmax as Softmax
    participant Output as 输出
    
    Input->>Linear: 输入嵌入 (B, L, D)
    Linear->>Linear: Q = XWQ
    Linear->>Linear: K = XWK  
    Linear->>Linear: V = XWV
    Linear->>MHA: Q, K, V (B, L, h, dk)
    MHA->>MHA: 转置为 (B, h, L, dk)
    MHA->>Scale: QKT (B, h, L, L)
    Scale->>Scale: 除以 √dk
    Scale->>Softmax: 注意力分数
    Softmax->>Softmax: 归一化权重
    Softmax->>MHA: 与V相乘
    MHA->>MHA: 转置并拼接
    MHA->>Output: 线性投影 WO
```
