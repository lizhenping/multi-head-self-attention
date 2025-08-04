"""
模型模块

包含多头注意力机制和文本相似度模型的实现。
"""

from .attention import (
    MultiHeadAttention,
    SelfAttention,
    create_padding_mask,
    create_look_ahead_mask
)

from .similarity_model import (
    TextSimilarityModel,
    TextEncoder,
    AttentionBlock,
    PositionalEncoding
)

__all__ = [
    'MultiHeadAttention',
    'SelfAttention', 
    'create_padding_mask',
    'create_look_ahead_mask',
    'TextSimilarityModel',
    'TextEncoder',
    'AttentionBlock',
    'PositionalEncoding'
]