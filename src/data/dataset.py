"""
数据集模块

用于加载和处理文本相似度数据集，支持多种数据格式和预处理选项。

作者: AI Assistant
日期: 2024
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Union
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from tqdm import tqdm
import numpy as np


class TextSimilarityDataset(Dataset):
    """
    文本相似度数据集
    
    该数据集用于加载和处理文本对及其相似度标签。
    支持 CSV 格式的数据，包含两个文本列和一个相似度分数列。
    
    参数:
        data_path (str): 数据文件路径
        tokenizer: 分词器函数
        vocab (Vocab): 词汇表对象
        max_length (int): 序列最大长度
        normalize_scores (bool): 是否将分数归一化到 [0, 1]
        score_range (Tuple[float, float]): 原始分数范围，用于归一化
        cache_dir (str): 缓存目录路径
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        vocab: Vocab,
        max_length: int = 200,
        normalize_scores: bool = True,
        score_range: Tuple[float, float] = (0, 5),
        cache_dir: Optional[str] = None,
        is_test: bool = False
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length
        self.normalize_scores = normalize_scores
        self.score_range = score_range
        self.cache_dir = cache_dir
        self.is_test = is_test
        
        # 特殊标记
        self.pad_idx = vocab['<pad>']
        self.unk_idx = vocab['<unk>']
        self.cls_idx = vocab.get('<cls>', self.unk_idx)
        self.sep_idx = vocab.get('<sep>', self.unk_idx)
        
        # 加载数据
        self.data = self._load_data()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"加载数据集: {data_path}")
        self.logger.info(f"数据集大小: {len(self.data)} 条")
    
    def _load_data(self) -> List[Tuple]:
        """加载并预处理数据"""
        # 检查缓存
        if self.cache_dir and not self.is_test:
            cache_path = os.path.join(
                self.cache_dir, 
                f"{os.path.basename(self.data_path)}.cache"
            )
            if os.path.exists(cache_path):
                self.logger.info(f"从缓存加载数据: {cache_path}")
                return torch.load(cache_path)
        
        # 读取数据
        df = pd.read_csv(self.data_path)
        
        # 验证数据列
        required_cols = ['sentence_a', 'sentence_b']
        if not self.is_test:
            required_cols.append('similarity')
            
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"数据集缺少必需的列: {col}")
        
        # 清理数据
        df = df.dropna(subset=required_cols)
        
        # 处理数据
        examples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理数据"):
            # 分词
            tokens_a = self._tokenize(str(row['sentence_a']))
            tokens_b = self._tokenize(str(row['sentence_b']))
            
            # 获取标签
            if not self.is_test:
                label = float(row['similarity'])
                if self.normalize_scores:
                    label = self._normalize_score(label)
            else:
                label = None
            
            examples.append((tokens_a, tokens_b, label))
        
        # 保存缓存
        if self.cache_dir and not self.is_test:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(examples, cache_path)
            self.logger.info(f"保存数据到缓存: {cache_path}")
        
        return examples
    
    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词"""
        tokens = self.tokenizer(text.lower())
        # 截断到最大长度
        if len(tokens) > self.max_length - 2:  # 留出 CLS 和 SEP 的位置
            tokens = tokens[:self.max_length - 2]
        return tokens
    
    def _normalize_score(self, score: float) -> float:
        """将分数归一化到 [0, 1]"""
        min_score, max_score = self.score_range
        return (score - min_score) / (max_score - min_score)
    
    def _denormalize_score(self, score: float) -> float:
        """将归一化的分数还原到原始范围"""
        min_score, max_score = self.score_range
        return score * (max_score - min_score) + min_score
    
    def _tokens_to_ids(self, tokens: List[str]) -> torch.Tensor:
        """将词序列转换为 ID 序列"""
        # 添加特殊标记
        tokens = ['<cls>'] + tokens + ['<sep>']
        
        # 转换为 ID
        ids = [self.vocab.get(token, self.unk_idx) for token in tokens]
        
        # 填充或截断
        if len(ids) < self.max_length:
            ids += [self.pad_idx] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]
        
        return torch.tensor(ids, dtype=torch.long)
    
    def _create_attention_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """创建注意力掩码"""
        return (ids != self.pad_idx).long()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        tokens_a, tokens_b, label = self.data[idx]
        
        # 转换为 ID
        input_ids_a = self._tokens_to_ids(tokens_a)
        input_ids_b = self._tokens_to_ids(tokens_b)
        
        # 创建注意力掩码
        attention_mask_a = self._create_attention_mask(input_ids_a)
        attention_mask_b = self._create_attention_mask(input_ids_b)
        
        sample = {
            'input_ids_a': input_ids_a,
            'input_ids_b': input_ids_b,
            'attention_mask_a': attention_mask_a,
            'attention_mask_b': attention_mask_b,
        }
        
        if label is not None:
            sample['label'] = torch.tensor(label, dtype=torch.float)
        
        return sample


def build_vocab_from_dataset(
    data_paths: List[str],
    tokenizer,
    min_freq: int = 2,
    max_vocab_size: Optional[int] = None,
    special_tokens: Optional[List[str]] = None
) -> Vocab:
    """
    从数据集构建词汇表
    
    参数:
        data_paths (List[str]): 数据文件路径列表
        tokenizer: 分词器
        min_freq (int): 最小词频
        max_vocab_size (int): 最大词汇表大小
        special_tokens (List[str]): 特殊标记列表
    
    返回:
        vocab (Vocab): 词汇表对象
    """
    if special_tokens is None:
        special_tokens = ['<pad>', '<unk>', '<cls>', '<sep>']
    
    # 收集所有词汇
    def yield_tokens():
        for data_path in data_paths:
            df = pd.read_csv(data_path)
            for _, row in df.iterrows():
                if pd.notna(row['sentence_a']):
                    yield tokenizer(str(row['sentence_a']).lower())
                if pd.notna(row['sentence_b']):
                    yield tokenizer(str(row['sentence_b']).lower())
    
    # 构建词汇表
    vocab = build_vocab_from_iterator(
        yield_tokens(),
        min_freq=min_freq,
        specials=special_tokens,
        special_first=True
    )
    
    # 设置默认索引
    vocab.set_default_index(vocab['<unk>'])
    
    # 限制词汇表大小
    if max_vocab_size and len(vocab) > max_vocab_size:
        # 保留最常见的词
        from collections import Counter
        counter = Counter()
        for tokens in yield_tokens():
            counter.update(tokens)
        
        # 获取最常见的词
        most_common = counter.most_common(max_vocab_size - len(special_tokens))
        vocab_tokens = special_tokens + [token for token, _ in most_common]
        
        # 重建词汇表
        vocab = build_vocab_from_iterator(
            [vocab_tokens],
            min_freq=1,
            specials=special_tokens,
            special_first=True
        )
        vocab.set_default_index(vocab['<unk>'])
    
    return vocab


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: Optional[str],
    tokenizer,
    vocab: Vocab,
    batch_size: int = 32,
    max_length: int = 200,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    创建数据加载器
    
    参数:
        train_path (str): 训练数据路径
        val_path (str): 验证数据路径
        test_path (str): 测试数据路径
        tokenizer: 分词器
        vocab (Vocab): 词汇表
        batch_size (int): 批次大小
        max_length (int): 最大序列长度
        num_workers (int): 数据加载线程数
        **kwargs: 其他数据集参数
    
    返回:
        train_loader, val_loader, test_loader
    """
    # 创建数据集
    train_dataset = TextSimilarityDataset(
        train_path, tokenizer, vocab, max_length, **kwargs
    )
    val_dataset = TextSimilarityDataset(
        val_path, tokenizer, vocab, max_length, **kwargs
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 测试数据加载器（如果有）
    test_loader = None
    if test_path:
        test_dataset = TextSimilarityDataset(
            test_path, tokenizer, vocab, max_length, is_test=True, **kwargs
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


class DataCollator:
    """
    数据整理器，用于批处理数据
    """
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """整理批次数据"""
        # 收集所有字段
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            if key.startswith('input_ids') or key.startswith('attention_mask'):
                # 堆叠张量
                collated[key] = torch.stack([item[key] for item in batch])
            elif key == 'label':
                # 标签
                collated[key] = torch.stack([item[key] for item in batch])
        
        return collated


if __name__ == "__main__":
    # 测试代码
    import tempfile
    
    # 创建临时测试数据
    test_data = pd.DataFrame({
        'sentence_a': ['This is a test.', 'Another test sentence.'],
        'sentence_b': ['This is a test!', 'Different sentence.'],
        'similarity': [4.5, 1.0]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    # 测试数据集
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_dataset([temp_path], tokenizer)
    
    dataset = TextSimilarityDataset(
        temp_path,
        tokenizer,
        vocab,
        max_length=50
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"词汇表大小: {len(vocab)}")
    print(f"第一个样本: {dataset[0]}")
    
    # 清理临时文件
    os.unlink(temp_path)