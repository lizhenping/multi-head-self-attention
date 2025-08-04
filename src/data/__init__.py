"""
数据处理模块

提供数据集加载、预处理和词汇表构建功能。
"""

from .dataset import (
    TextSimilarityDataset,
    build_vocab_from_dataset,
    create_data_loaders,
    DataCollator
)

__all__ = [
    'TextSimilarityDataset',
    'build_vocab_from_dataset',
    'create_data_loaders',
    'DataCollator'
]