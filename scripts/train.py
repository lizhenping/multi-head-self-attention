#!/usr/bin/env python
"""
训练脚本

提供命令行接口来训练文本相似度模型。
支持灵活的参数配置、实验管理和结果记录。

使用示例:
    # 使用默认配置训练
    python scripts/train.py
    
    # 使用自定义配置文件
    python scripts/train.py --config configs/my_config.yaml
    
    # 覆盖特定参数
    python scripts/train.py --batch-size 64 --learning-rate 0.001
    
    # 使用预训练嵌入
    python scripts/train.py --use-pretrained-embeddings --embeddings-name glove.6B.300d

作者: AI Assistant
日期: 2024
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe

from src.configs.config import Config, load_config
from src.models import TextSimilarityModel
from src.data.dataset import build_vocab_from_dataset, create_data_loaders
from src.utils.trainer import Trainer
from src.utils.utils import set_seed, setup_logging, save_experiment_info


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="训练文本相似度模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基础参数
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='配置文件路径 (YAML 或 JSON 格式)'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='实验名称，用于标识和组织结果'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子，用于结果复现'
    )
    
    # 数据参数
    parser.add_argument(
        '--train-path',
        type=str,
        default=None,
        help='训练数据文件路径'
    )
    
    parser.add_argument(
        '--val-path',
        type=str,
        default=None,
        help='验证数据文件路径'
    )
    
    parser.add_argument(
        '--test-path',
        type=str,
        default=None,
        help='测试数据文件路径'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=None,
        help='序列最大长度'
    )
    
    # 模型参数
    parser.add_argument(
        '--embed-dim',
        type=int,
        default=None,
        help='嵌入维度'
    )
    
    parser.add_argument(
        '--num-heads',
        type=int,
        default=None,
        help='注意力头数'
    )
    
    parser.add_argument(
        '--num-layers',
        type=int,
        default=None,
        help='编码器层数'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        default=None,
        help='Dropout 概率'
    )
    
    parser.add_argument(
        '--pooling-strategy',
        type=str,
        choices=['mean', 'max', 'cls'],
        default=None,
        help='池化策略'
    )
    
    parser.add_argument(
        '--use-pretrained-embeddings',
        action='store_true',
        help='是否使用预训练词嵌入'
    )
    
    parser.add_argument(
        '--embeddings-name',
        type=str,
        default='glove.6B.100d',
        help='GloVe 嵌入名称 (例如: glove.6B.100d, glove.6B.300d)'
    )
    
    # 训练参数
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='批次大小'
    )
    
    parser.add_argument(
        '--learning-rate', '--lr',
        type=float,
        default=None,
        help='学习率'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=None,
        help='训练轮数'
    )
    
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'adamw', 'sgd'],
        default=None,
        help='优化器类型'
    )
    
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['cosine', 'linear', 'constant'],
        default=None,
        help='学习率调度器'
    )
    
    parser.add_argument(
        '--gradient-clip',
        type=float,
        default=None,
        help='梯度裁剪值'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=None,
        help='权重衰减'
    )
    
    parser.add_argument(
        '--no-early-stopping',
        action='store_true',
        help='禁用早停机制'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=None,
        help='早停耐心值'
    )
    
    # 设备参数
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='训练设备 (auto 会自动选择)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='数据加载器工作进程数'
    )
    
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='使用混合精度训练'
    )
    
    # 输出参数
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='检查点保存目录'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别'
    )
    
    parser.add_argument(
        '--use-tensorboard',
        action='store_true',
        help='使用 TensorBoard 记录日志'
    )
    
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='使用 Weights & Biases 记录日志'
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='W&B 项目名称'
    )
    
    # 其他参数
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从检查点恢复训练'
    )
    
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='仅进行评估，不训练'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='试运行，仅显示配置不实际训练'
    )
    
    return parser.parse_args()


def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """根据命令行参数更新配置"""
    updates = {}
    
    # 基础参数
    if args.experiment_name:
        updates['logging.experiment_name'] = args.experiment_name
    if args.seed is not None:
        updates['seed'] = args.seed
    
    # 数据参数
    if args.train_path:
        updates['data.train_path'] = args.train_path
    if args.val_path:
        updates['data.val_path'] = args.val_path
    if args.test_path:
        updates['data.test_path'] = args.test_path
    if args.max_length is not None:
        updates['model.max_seq_len'] = args.max_length
    
    # 模型参数
    if args.embed_dim is not None:
        updates['model.embed_dim'] = args.embed_dim
    if args.num_heads is not None:
        updates['model.num_heads'] = args.num_heads
    if args.num_layers is not None:
        updates['model.num_layers'] = args.num_layers
    if args.dropout is not None:
        updates['model.dropout'] = args.dropout
    if args.pooling_strategy:
        updates['model.pooling_strategy'] = args.pooling_strategy
    if args.use_pretrained_embeddings:
        updates['model.use_pretrained_embeddings'] = True
        updates['model.embeddings_name'] = args.embeddings_name
    
    # 训练参数
    if args.batch_size is not None:
        updates['training.batch_size'] = args.batch_size
    if args.learning_rate is not None:
        updates['training.learning_rate'] = args.learning_rate
    if args.num_epochs is not None:
        updates['training.num_epochs'] = args.num_epochs
    if args.optimizer:
        updates['training.optimizer'] = args.optimizer
    if args.scheduler:
        updates['training.scheduler'] = args.scheduler
    if args.gradient_clip is not None:
        updates['training.gradient_clip'] = args.gradient_clip
    if args.weight_decay is not None:
        updates['training.weight_decay'] = args.weight_decay
    if args.no_early_stopping:
        updates['training.early_stopping'] = False
    if args.patience is not None:
        updates['training.patience'] = args.patience
    
    # 设备参数
    if args.device == 'auto':
        updates['training.device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device:
        updates['training.device'] = args.device
    if args.num_workers is not None:
        updates['training.num_workers'] = args.num_workers
    if args.fp16:
        updates['training.fp16'] = True
    
    # 输出参数
    if args.output_dir:
        updates['output_dir'] = args.output_dir
    if args.checkpoint_dir:
        updates['checkpoint_dir'] = args.checkpoint_dir
    if args.log_level:
        updates['logging.log_level'] = args.log_level
    if args.use_tensorboard:
        updates['logging.use_tensorboard'] = True
    if args.use_wandb:
        updates['logging.use_wandb'] = True
    if args.wandb_project:
        updates['logging.wandb_project'] = args.wandb_project
    
    # 应用更新
    config.update(updates)
    
    return config


def load_pretrained_embeddings(vocab, config: Config):
    """加载预训练词嵌入"""
    logging.info(f"加载预训练嵌入: {config.model.embeddings_name}")
    
    # 解析嵌入名称
    name_parts = config.model.embeddings_name.split('.')
    name = name_parts[0]
    dim = int(name_parts[-1][:-1])  # 提取维度，例如 '100d' -> 100
    
    # 加载 GloVe 嵌入
    glove = GloVe(name=name[6:], dim=dim)  # 'glove.6B' -> '6B'
    
    # 创建嵌入矩阵
    embeddings = torch.zeros(len(vocab), dim)
    
    # 填充嵌入
    found = 0
    for i, word in enumerate(vocab.get_itos()):
        if word in glove.stoi:
            embeddings[i] = glove[word]
            found += 1
        else:
            # 对未找到的词使用随机初始化
            embeddings[i] = torch.randn(dim) * 0.1
    
    logging.info(f"找到 {found}/{len(vocab)} 个词的预训练嵌入")
    
    return embeddings


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    config = update_config_from_args(config, args)
    
    # 设置实验名称
    if not config.logging.experiment_name:
        config.logging.experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建输出目录
    output_dir = Path(config.output_dir) / config.logging.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(output_dir)
    config.checkpoint_dir = str(output_dir / "checkpoints")
    
    # 设置日志
    setup_logging(
        log_file=output_dir / "train.log",
        log_level=config.logging.log_level
    )
    
    logging.info("="*50)
    logging.info(f"实验名称: {config.logging.experiment_name}")
    logging.info(f"输出目录: {output_dir}")
    logging.info("="*50)
    
    # 显示配置
    if args.dry_run:
        logging.info("配置内容:")
        print(config)
        logging.info("试运行模式，退出程序")
        return
    
    # 保存配置
    config.save(output_dir / "config.yaml")
    logging.info(f"配置已保存到: {output_dir / 'config.yaml'}")
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 准备数据
    logging.info("准备数据...")
    tokenizer = get_tokenizer(config.data.tokenizer)
    
    # 构建词汇表
    vocab = build_vocab_from_dataset(
        [config.data.train_path],
        tokenizer,
        min_freq=config.model.min_freq,
        max_vocab_size=config.model.max_vocab_size
    )
    config.model.vocab_size = len(vocab)
    logging.info(f"词汇表大小: {len(vocab)}")
    
    # 保存词汇表
    torch.save(vocab, output_dir / "vocab.pt")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path=config.data.train_path,
        val_path=config.data.val_path,
        test_path=config.data.test_path,
        tokenizer=tokenizer,
        vocab=vocab,
        batch_size=config.training.batch_size,
        max_length=config.model.max_seq_len,
        num_workers=config.training.num_workers,
        normalize_scores=config.data.normalize_scores,
        score_range=config.data.score_range,
        cache_dir=config.data.cache_dir
    )
    
    # 准备预训练嵌入
    pretrained_embeddings = None
    if config.model.use_pretrained_embeddings:
        pretrained_embeddings = load_pretrained_embeddings(vocab, config)
        # 更新嵌入维度
        config.model.embed_dim = pretrained_embeddings.size(1)
    
    # 创建模型
    logging.info("创建模型...")
    model = TextSimilarityModel(
        vocab_size=len(vocab),
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        max_seq_len=config.model.max_seq_len,
        pooling_strategy=config.model.pooling_strategy,
        dropout=config.model.dropout,
        pretrained_embeddings=pretrained_embeddings
    )
    
    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"总参数量: {total_params:,}")
    logging.info(f"可训练参数量: {trainable_params:,}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logging.info(f"从检查点恢复: {args.resume}")
    
    # 仅评估模式
    if args.eval_only:
        logging.info("仅评估模式")
        val_metrics = trainer._validate()
        logging.info(f"验证集指标: {val_metrics.to_dict()}")
        
        if test_loader:
            test_metrics = trainer._test()
            logging.info(f"测试集指标: {test_metrics.to_dict()}")
        
        return
    
    # 训练模型
    logging.info("开始训练...")
    results = trainer.train()
    
    # 保存结果
    save_experiment_info(
        output_dir=output_dir,
        config=config,
        results=results,
        model_info={
            'total_params': total_params,
            'trainable_params': trainable_params,
            'vocab_size': len(vocab)
        }
    )
    
    logging.info("训练完成！")
    logging.info(f"最佳 Pearson 相关系数: {results['best_metric']:.4f}")
    logging.info(f"结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()