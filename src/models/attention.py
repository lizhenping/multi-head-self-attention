"""
多头注意力机制模块

该模块实现了 Transformer 架构中的核心组件 - 多头注意力机制。
多头注意力能够让模型同时关注来自不同位置的不同表示子空间的信息。

作者: AI Assistant
日期: 2024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现
    
    多头注意力机制通过并行计算多个注意力头，能够从不同的子空间中提取信息，
    从而增强模型的表达能力。这是 Transformer 模型的核心组件。
    
    数学公式:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    参数:
        embed_dim (int): 输入嵌入的维度，必须能被 num_heads 整除
        num_heads (int): 注意力头的数量
        dropout (float): Dropout 概率，默认为 0.1
        bias (bool): 是否在线性变换中使用偏置，默认为 True
    
    输入形状:
        - query: [batch_size, seq_len, embed_dim]
        - key: [batch_size, seq_len, embed_dim]
        - value: [batch_size, seq_len, embed_dim]
        - mask: [batch_size, seq_len, seq_len] 或 None
    
    输出形状:
        - output: [batch_size, seq_len, embed_dim]
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super(MultiHeadAttention, self).__init__()
        
        # 验证嵌入维度是否能被头数整除
        assert embed_dim % num_heads == 0, f"嵌入维度 {embed_dim} 必须能被头数 {num_heads} 整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        self.scale = math.sqrt(self.head_dim)   # 缩放因子，防止点积结果过大
        
        # 定义 Q、K、V 的线性变换层
        # 这些层将输入投影到查询、键、值空间
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # 输出投影层，将多头的输出合并后投影回原始维度
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout 层，用于正则化
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._reset_parameters()
    
    def _reset_parameters(self):
        """使用 Xavier 均匀分布初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, query, key, value, mask=None, return_attention=False):
        """
        前向传播
        
        参数:
            query (Tensor): 查询张量 [batch_size, seq_len, embed_dim]
            key (Tensor): 键张量 [batch_size, seq_len, embed_dim]
            value (Tensor): 值张量 [batch_size, seq_len, embed_dim]
            mask (Tensor, optional): 注意力掩码 [batch_size, seq_len, seq_len]
            return_attention (bool): 是否返回注意力权重
        
        返回:
            output (Tensor): 输出张量 [batch_size, seq_len, embed_dim]
            attention_weights (Tensor, optional): 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 1. 线性变换并重塑为多头形式
        # [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
        Q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 2. 转置以便进行注意力计算
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 3. 计算缩放点积注意力
        # [batch_size, num_heads, seq_len, head_dim] @ [batch_size, num_heads, head_dim, seq_len]
        # -> [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 4. 应用掩码（如果提供）
        if mask is not None:
            # 扩展掩码维度以匹配注意力分数
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            # 将掩码位置填充为极小值，使得 softmax 后接近 0
            attention_scores.masked_fill_(mask == 0, float('-inf'))
        
        # 5. 应用 softmax 获得注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 6. 应用注意力权重到值矩阵
        # [batch_size, num_heads, seq_len, seq_len] @ [batch_size, num_heads, seq_len, head_dim]
        # -> [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attention_weights, V)
        
        # 7. 重塑多头输出
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        context = context.transpose(1, 2).contiguous()
        
        # 8. 合并多头
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, embed_dim]
        context = context.view(batch_size, seq_len, self.embed_dim)
        
        # 9. 最终的线性投影
        output = self.out_proj(context)
        
        if return_attention:
            return output, attention_weights
        return output


class SelfAttention(MultiHeadAttention):
    """
    自注意力机制
    
    自注意力是多头注意力的特殊情况，其中查询、键和值都来自同一个输入。
    这允许序列中的每个位置都能关注序列中的所有位置。
    """
    
    def forward(self, x, mask=None, return_attention=False):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量 [batch_size, seq_len, embed_dim]
            mask (Tensor, optional): 注意力掩码
            return_attention (bool): 是否返回注意力权重
        
        返回:
            output (Tensor): 输出张量 [batch_size, seq_len, embed_dim]
        """
        return super().forward(x, x, x, mask, return_attention)


def create_padding_mask(seq, pad_idx=0):
    """
    创建填充掩码
    
    参数:
        seq (Tensor): 输入序列 [batch_size, seq_len]
        pad_idx (int): 填充标记的索引
    
    返回:
        mask (Tensor): 掩码张量 [batch_size, 1, 1, seq_len]
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size):
    """
    创建前瞻掩码（用于解码器自注意力）
    
    参数:
        size (int): 序列长度
    
    返回:
        mask (Tensor): 下三角掩码 [1, 1, size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask.eq(0).unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
    # 测试代码
    batch_size = 2
    seq_len = 10
    embed_dim = 512
    num_heads = 8
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # 创建多头注意力层
    mha = MultiHeadAttention(embed_dim, num_heads)
    
    # 前向传播
    output = mha(x, x, x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试带掩码的情况
    mask = create_padding_mask(torch.randint(0, 2, (batch_size, seq_len)))
    output_masked = mha(x, x, x, mask)
    print(f"带掩码输出形状: {output_masked.shape}")