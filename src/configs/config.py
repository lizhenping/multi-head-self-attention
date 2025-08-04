"""
配置管理模块

提供统一的配置管理接口，支持从文件加载配置和命令行覆盖。
使用 dataclass 定义配置结构，支持类型检查和默认值。

作者: AI Assistant
日期: 2024
"""

import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from pathlib import Path


@dataclass
class ModelConfig:
    """模型配置"""
    # 模型架构
    embed_dim: int = 256                # 嵌入维度
    num_heads: int = 8                  # 注意力头数
    num_layers: int = 2                 # 编码器层数
    ff_dim: Optional[int] = None        # 前馈网络维度（None 表示 4*embed_dim）
    dropout: float = 0.1                # Dropout 概率
    
    # 池化策略
    pooling_strategy: str = 'mean'      # 池化策略: 'mean', 'max', 'cls'
    
    # 词汇表
    vocab_size: int = 50000             # 词汇表大小
    max_vocab_size: Optional[int] = None  # 最大词汇表大小限制
    min_freq: int = 2                   # 最小词频
    
    # 序列长度
    max_seq_len: int = 200              # 最大序列长度
    
    # 预训练嵌入
    use_pretrained_embeddings: bool = False  # 是否使用预训练嵌入
    embeddings_path: Optional[str] = None     # 预训练嵌入文件路径
    embeddings_name: str = 'glove.6B.100d'   # GloVe 嵌入名称


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    batch_size: int = 32                # 批次大小
    learning_rate: float = 1e-4         # 学习率
    num_epochs: int = 50                # 训练轮数
    gradient_clip: float = 1.0          # 梯度裁剪值
    weight_decay: float = 1e-5          # 权重衰减
    
    # 优化器
    optimizer: str = 'adam'             # 优化器类型: 'adam', 'adamw', 'sgd'
    adam_betas: tuple = (0.9, 0.999)    # Adam beta 参数
    adam_eps: float = 1e-8              # Adam epsilon
    
    # 学习率调度
    scheduler: str = 'cosine'           # 调度器类型: 'cosine', 'linear', 'constant'
    warmup_steps: int = 1000            # 预热步数
    warmup_ratio: float = 0.1           # 预热比例
    
    # 训练策略
    early_stopping: bool = True         # 是否使用早停
    patience: int = 10                  # 早停耐心值
    min_delta: float = 1e-4             # 最小改进阈值
    
    # 检查点
    save_steps: int = 500               # 保存检查点的步数间隔
    save_total_limit: int = 3           # 保存的最大检查点数
    save_best_only: bool = True         # 仅保存最佳模型
    
    # 验证
    eval_steps: int = 100               # 验证步数间隔
    eval_strategy: str = 'steps'        # 验证策略: 'steps', 'epoch'
    
    # 设备
    device: str = 'cuda'                # 设备: 'cuda', 'cpu'
    num_workers: int = 4                # 数据加载器工作进程数
    pin_memory: bool = True             # 是否固定内存
    
    # 混合精度训练
    fp16: bool = False                  # 是否使用 FP16
    fp16_opt_level: str = 'O1'          # FP16 优化级别


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    train_path: str = 'data/train.csv'  # 训练数据路径
    val_path: str = 'data/val.csv'      # 验证数据路径
    test_path: Optional[str] = 'data/test.csv'  # 测试数据路径
    
    # 数据处理
    tokenizer: str = 'basic_english'    # 分词器类型
    normalize_scores: bool = True       # 是否归一化分数
    score_range: tuple = (0, 5)         # 分数范围
    
    # 缓存
    cache_dir: Optional[str] = 'cache'  # 缓存目录
    
    # 数据增强
    augment_data: bool = False          # 是否进行数据增强
    augment_prob: float = 0.1           # 数据增强概率


@dataclass
class LoggingConfig:
    """日志配置"""
    # 日志级别
    log_level: str = 'INFO'             # 日志级别
    log_file: Optional[str] = None      # 日志文件路径
    
    # TensorBoard
    use_tensorboard: bool = True        # 是否使用 TensorBoard
    tensorboard_dir: str = 'runs'       # TensorBoard 目录
    
    # Weights & Biases
    use_wandb: bool = False             # 是否使用 W&B
    wandb_project: str = 'text-similarity'  # W&B 项目名
    wandb_entity: Optional[str] = None  # W&B 实体名
    
    # 实验跟踪
    experiment_name: Optional[str] = None  # 实验名称
    tags: List[str] = field(default_factory=list)  # 实验标签


@dataclass
class Config:
    """主配置类，包含所有子配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # 输出路径
    output_dir: str = 'output'          # 输出目录
    checkpoint_dir: str = 'checkpoints' # 检查点目录
    
    # 随机种子
    seed: int = 42                      # 随机种子
    
    # 配置元信息
    config_name: Optional[str] = None   # 配置名称
    config_file: Optional[str] = None   # 配置文件路径
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'Config':
        """从文件加载配置"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        # 根据文件扩展名选择加载方式
        if file_path.suffix in ['.yaml', '.yml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {file_path.suffix}")
        
        # 创建配置对象
        config = cls.from_dict(config_dict)
        config.config_file = str(file_path)
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置"""
        # 创建子配置
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        # 创建主配置
        main_config = {
            k: v for k, v in config_dict.items()
            if k not in ['model', 'training', 'data', 'logging']
        }
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            logging=logging_config,
            **main_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save(self, file_path: Union[str, Path]):
        """保存配置到文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        if file_path.suffix in ['.yaml', '.yml']:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        elif file_path.suffix == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的配置文件格式: {file_path.suffix}")
    
    def update(self, updates: Dict[str, Any]):
        """更新配置"""
        for key, value in updates.items():
            if '.' in key:
                # 处理嵌套键，如 'model.embed_dim'
                keys = key.split('.')
                obj = self
                for k in keys[:-1]:
                    obj = getattr(obj, k)
                setattr(obj, keys[-1], value)
            else:
                setattr(self, key, value)
    
    def validate(self):
        """验证配置的合法性"""
        # 检查模型配置
        if self.model.embed_dim % self.model.num_heads != 0:
            raise ValueError(
                f"嵌入维度 {self.model.embed_dim} 必须能被头数 {self.model.num_heads} 整除"
            )
        
        # 检查池化策略
        valid_pooling = ['mean', 'max', 'cls']
        if self.model.pooling_strategy not in valid_pooling:
            raise ValueError(
                f"无效的池化策略: {self.model.pooling_strategy}. "
                f"必须是 {valid_pooling} 之一"
            )
        
        # 检查优化器
        valid_optimizers = ['adam', 'adamw', 'sgd']
        if self.training.optimizer not in valid_optimizers:
            raise ValueError(
                f"无效的优化器: {self.training.optimizer}. "
                f"必须是 {valid_optimizers} 之一"
            )
        
        # 检查设备
        if self.training.device == 'cuda' and not torch.cuda.is_available():
            import torch
            self.training.device = 'cpu'
            print("警告: CUDA 不可用，切换到 CPU")
        
        # 检查数据路径
        if not os.path.exists(self.data.train_path):
            raise FileNotFoundError(f"训练数据文件不存在: {self.data.train_path}")
        
        if not os.path.exists(self.data.val_path):
            raise FileNotFoundError(f"验证数据文件不存在: {self.data.val_path}")
    
    def __str__(self) -> str:
        """字符串表示"""
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)


def create_default_config() -> Config:
    """创建默认配置"""
    return Config()


def load_config(
    config_file: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """
    加载配置
    
    参数:
        config_file: 配置文件路径
        overrides: 配置覆盖字典
    
    返回:
        config: 配置对象
    """
    # 加载配置
    if config_file:
        config = Config.from_file(config_file)
    else:
        config = create_default_config()
    
    # 应用覆盖
    if overrides:
        config.update(overrides)
    
    # 验证配置
    config.validate()
    
    return config


if __name__ == "__main__":
    # 测试代码
    # 创建默认配置
    config = create_default_config()
    print("默认配置:")
    print(config)
    
    # 保存配置
    config.save("test_config.yaml")
    print("\n配置已保存到 test_config.yaml")
    
    # 加载配置
    loaded_config = Config.from_file("test_config.yaml")
    print("\n加载的配置:")
    print(loaded_config.model)
    
    # 更新配置
    loaded_config.update({
        'model.embed_dim': 512,
        'training.batch_size': 64
    })
    print("\n更新后的配置:")
    print(f"嵌入维度: {loaded_config.model.embed_dim}")
    print(f"批次大小: {loaded_config.training.batch_size}")
    
    # 清理测试文件
    os.unlink("test_config.yaml")