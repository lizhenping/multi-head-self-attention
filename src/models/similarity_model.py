"""
文本相似度模型

该模块实现了基于多头注意力机制的文本相似度计算模型。
模型使用预训练的词嵌入，通过注意力机制捕获文本的语义信息，
最后通过余弦相似度计算两个文本的相似程度。

作者: AI Assistant
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention import MultiHeadAttention


class TextEncoder(nn.Module):
    """
    文本编码器
    
    该编码器使用词嵌入和多头注意力机制将文本序列编码为固定维度的向量表示。
    
    参数:
        vocab_size (int): 词汇表大小
        embed_dim (int): 嵌入维度
        num_heads (int): 注意力头数
        num_layers (int): 注意力层数
        max_seq_len (int): 最大序列长度
        dropout (float): Dropout 概率
        pretrained_embeddings (Tensor, optional): 预训练的词嵌入矩阵
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int = 1,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        super(TextEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # 词嵌入层
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, 
                freeze=False,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size, 
                embed_dim, 
                padding_idx=0
            )
        
        # 位置编码（可选）
        self.position_encoding = PositionalEncoding(
            embed_dim, 
            dropout, 
            max_seq_len
        )
        
        # 多层注意力块
        self.attention_layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终的层归一化
        self.final_norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            input_ids (Tensor): 输入的词索引 [batch_size, seq_len]
            attention_mask (Tensor, optional): 注意力掩码 [batch_size, seq_len]
        
        返回:
            output (Tensor): 编码后的表示 [batch_size, seq_len, embed_dim]
        """
        # 词嵌入
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # 添加位置编码
        x = self.position_encoding(x)
        
        # 创建注意力掩码
        if attention_mask is not None:
            # 将 2D 掩码转换为 4D 掩码
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.float()
        
        # 通过多层注意力
        for attention_layer in self.attention_layers:
            x = attention_layer(x, attention_mask)
        
        # 最终归一化
        x = self.final_norm(x)
        
        return x


class AttentionBlock(nn.Module):
    """
    注意力块，包含多头注意力和前馈网络
    
    参数:
        embed_dim (int): 嵌入维度
        num_heads (int): 注意力头数
        dropout (float): Dropout 概率
        ff_dim (int): 前馈网络的隐藏层维度
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.1,
        ff_dim: Optional[int] = None
    ):
        super(AttentionBlock, self).__init__()
        
        # 多头自注意力
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.attention_dropout = nn.Dropout(dropout)
        
        # 前馈网络
        if ff_dim is None:
            ff_dim = 4 * embed_dim  # 默认为 4 倍的嵌入维度
            
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff_dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量 [batch_size, seq_len, embed_dim]
            mask (Tensor, optional): 注意力掩码
        
        返回:
            output (Tensor): 输出张量 [batch_size, seq_len, embed_dim]
        """
        # 自注意力 + 残差连接
        attn_output = self.attention(x, x, x, mask)
        attn_output = self.attention_dropout(attn_output)
        x = self.attention_norm(x + attn_output)
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        ff_output = self.ff_dropout(ff_output)
        x = self.ff_norm(x + ff_output)
        
        return x


class PositionalEncoding(nn.Module):
    """
    位置编码
    
    使用正弦和余弦函数为序列中的每个位置生成位置编码。
    
    参数:
        embed_dim (int): 嵌入维度
        dropout (float): Dropout 概率
        max_len (int): 最大序列长度
    """
    
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """添加位置编码到输入张量"""
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TextSimilarityModel(nn.Module):
    """
    文本相似度模型
    
    该模型计算两个文本之间的语义相似度。
    使用共享的文本编码器对两个文本进行编码，然后计算余弦相似度。
    
    参数:
        vocab_size (int): 词汇表大小
        embed_dim (int): 嵌入维度
        num_heads (int): 注意力头数
        num_layers (int): 编码器层数
        max_seq_len (int): 最大序列长度
        pooling_strategy (str): 池化策略 ('mean', 'max', 'cls')
        dropout (float): Dropout 概率
        pretrained_embeddings (Tensor, optional): 预训练的词嵌入
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int = 2,
        max_seq_len: int = 200,
        pooling_strategy: str = 'mean',
        dropout: float = 0.1,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        super(TextSimilarityModel, self).__init__()
        
        self.pooling_strategy = pooling_strategy
        
        # 文本编码器（共享权重）
        self.encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings
        )
        
        # 输出投影层（可选）
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )
        
    def pool_sequence(
        self, 
        sequence: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        对序列进行池化操作
        
        参数:
            sequence (Tensor): 序列表示 [batch_size, seq_len, embed_dim]
            attention_mask (Tensor, optional): 注意力掩码 [batch_size, seq_len]
        
        返回:
            pooled (Tensor): 池化后的表示 [batch_size, embed_dim]
        """
        if self.pooling_strategy == 'mean':
            if attention_mask is not None:
                # 仅对非填充位置进行平均
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence.size())
                sum_embeddings = torch.sum(sequence * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return sequence.mean(dim=1)
                
        elif self.pooling_strategy == 'max':
            if attention_mask is not None:
                # 将填充位置设为极小值
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence.size())
                sequence = sequence.masked_fill(mask_expanded == 0, -1e9)
            return sequence.max(dim=1)[0]
            
        elif self.pooling_strategy == 'cls':
            # 使用第一个位置（CLS token）的表示
            return sequence[:, 0, :]
            
        else:
            raise ValueError(f"未知的池化策略: {self.pooling_strategy}")
    
    def encode_text(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码文本为固定维度的向量
        
        参数:
            input_ids (Tensor): 输入词索引 [batch_size, seq_len]
            attention_mask (Tensor, optional): 注意力掩码
        
        返回:
            text_embedding (Tensor): 文本嵌入 [batch_size, embed_dim]
        """
        # 编码序列
        sequence_output = self.encoder(input_ids, attention_mask)
        
        # 池化
        pooled_output = self.pool_sequence(sequence_output, attention_mask)
        
        # 输出投影
        text_embedding = self.output_proj(pooled_output)
        
        return text_embedding
    
    def forward(
        self,
        input_ids_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_a: Optional[torch.Tensor] = None,
        attention_mask_b: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播，计算两个文本的相似度
        
        参数:
            input_ids_a (Tensor): 第一个文本的词索引 [batch_size, seq_len]
            input_ids_b (Tensor): 第二个文本的词索引 [batch_size, seq_len]
            attention_mask_a (Tensor, optional): 第一个文本的注意力掩码
            attention_mask_b (Tensor, optional): 第二个文本的注意力掩码
            return_embeddings (bool): 是否返回文本嵌入
        
        返回:
            similarity (Tensor): 相似度分数 [batch_size]
            embeddings (Tuple[Tensor, Tensor], optional): 文本嵌入
        """
        # 编码两个文本
        embedding_a = self.encode_text(input_ids_a, attention_mask_a)
        embedding_b = self.encode_text(input_ids_b, attention_mask_b)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(embedding_a, embedding_b, dim=1)
        
        if return_embeddings:
            return similarity, (embedding_a, embedding_b)
        else:
            return similarity
    
    def compute_similarity(
        self, 
        embedding_a: torch.Tensor, 
        embedding_b: torch.Tensor
    ) -> torch.Tensor:
        """
        计算两个嵌入向量的余弦相似度
        
        参数:
            embedding_a (Tensor): 第一个嵌入 [batch_size, embed_dim]
            embedding_b (Tensor): 第二个嵌入 [batch_size, embed_dim]
        
        返回:
            similarity (Tensor): 相似度分数 [batch_size]
        """
        return F.cosine_similarity(embedding_a, embedding_b, dim=1)


if __name__ == "__main__":
    # 测试代码
    batch_size = 4
    seq_len = 20
    vocab_size = 10000
    embed_dim = 256
    num_heads = 8
    
    # 创建模型
    model = TextSimilarityModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=2,
        pooling_strategy='mean'
    )
    
    # 创建随机输入
    input_ids_a = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_ids_b = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    similarity = model(input_ids_a, input_ids_b)
    print(f"输入形状: {input_ids_a.shape}")
    print(f"相似度分数: {similarity}")
    print(f"相似度分数形状: {similarity.shape}")