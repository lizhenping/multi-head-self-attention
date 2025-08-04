"""
实用工具模块

提供各种实用函数，包括设置随机种子、日志配置、实验信息保存等。

作者: AI Assistant
日期: 2024
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """
    设置所有随机数生成器的种子，确保结果可重现
    
    参数:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 设置 cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logging.info(f"设置随机种子: {seed}")


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    log_level: str = "INFO",
    log_format: Optional[str] = None
):
    """
    配置日志系统
    
    参数:
        log_file: 日志文件路径
        log_level: 日志级别
        log_format: 日志格式
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 获取日志级别
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"无效的日志级别: {log_level}")
    
    # 配置处理器
    handlers = [logging.StreamHandler()]  # 控制台输出
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    # 配置日志
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers,
        force=True  # 强制重新配置
    )
    
    # 设置第三方库的日志级别
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torchtext").setLevel(logging.WARNING)


def save_experiment_info(
    output_dir: Union[str, Path],
    config: Any,
    results: Dict[str, Any],
    model_info: Optional[Dict[str, Any]] = None
):
    """
    保存实验信息
    
    参数:
        output_dir: 输出目录
        config: 配置对象
        results: 训练结果
        model_info: 模型信息
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存结果
    with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存训练历史
    if 'history' in results:
        history = results['history']
        
        # 保存为 JSON
        with open(output_dir / "history.json", 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        
        # 保存为 CSV (方便分析)
        import pandas as pd
        
        # 训练历史
        if 'train' in history:
            train_df = pd.DataFrame(history['train'])
            train_df.to_csv(output_dir / "train_history.csv", index=False)
        
        # 验证历史
        if 'val' in history:
            val_df = pd.DataFrame(history['val'])
            val_df.to_csv(output_dir / "val_history.csv", index=False)
    
    # 保存实验摘要
    summary = {
        'experiment_name': config.logging.experiment_name,
        'best_metric': results.get('best_metric'),
        'total_time': results.get('total_time'),
        'model_info': model_info,
        'config': config.to_dict() if hasattr(config, 'to_dict') else config
    }
    
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 创建简单的报告
    create_experiment_report(output_dir, summary)


def create_experiment_report(output_dir: Path, summary: Dict[str, Any]):
    """
    创建实验报告
    
    参数:
        output_dir: 输出目录
        summary: 实验摘要
    """
    report_lines = [
        "# 实验报告",
        "",
        f"**实验名称**: {summary.get('experiment_name', 'N/A')}",
        "",
        "## 最佳结果",
        f"- **最佳 Pearson 相关系数**: {summary.get('best_metric', 'N/A'):.4f}",
        f"- **总训练时间**: {format_time(summary.get('total_time', 0))}",
        "",
        "## 模型信息",
    ]
    
    if summary.get('model_info'):
        model_info = summary['model_info']
        report_lines.extend([
            f"- **总参数量**: {model_info.get('total_params', 'N/A'):,}",
            f"- **可训练参数量**: {model_info.get('trainable_params', 'N/A'):,}",
            f"- **词汇表大小**: {model_info.get('vocab_size', 'N/A'):,}",
        ])
    
    report_lines.extend([
        "",
        "## 配置摘要",
    ])
    
    if summary.get('config'):
        config = summary['config']
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        
        report_lines.extend([
            "### 模型配置",
            f"- **嵌入维度**: {model_config.get('embed_dim', 'N/A')}",
            f"- **注意力头数**: {model_config.get('num_heads', 'N/A')}",
            f"- **编码器层数**: {model_config.get('num_layers', 'N/A')}",
            f"- **池化策略**: {model_config.get('pooling_strategy', 'N/A')}",
            "",
            "### 训练配置",
            f"- **批次大小**: {training_config.get('batch_size', 'N/A')}",
            f"- **学习率**: {training_config.get('learning_rate', 'N/A')}",
            f"- **训练轮数**: {training_config.get('num_epochs', 'N/A')}",
            f"- **优化器**: {training_config.get('optimizer', 'N/A')}",
        ])
    
    # 写入报告
    with open(output_dir / "report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    参数:
        seconds: 秒数
        
    返回:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    统计模型参数
    
    参数:
        model: PyTorch 模型
        
    返回:
        包含总参数和可训练参数的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """
    加载检查点
    
    参数:
        checkpoint_path: 检查点文件路径
        model: 模型（如果提供，将加载模型参数）
        optimizer: 优化器（如果提供，将加载优化器状态）
        map_location: 设备映射位置
        
    返回:
        检查点字典
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"加载模型参数: {checkpoint_path}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"加载优化器状态: {checkpoint_path}")
    
    return checkpoint


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    **kwargs
):
    """
    保存检查点
    
    参数:
        checkpoint_path: 检查点保存路径
        model: 模型
        optimizer: 优化器
        epoch: 当前轮数
        **kwargs: 其他要保存的信息
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"保存检查点: {checkpoint_path}")


def plot_training_history(
    history: Dict[str, list],
    output_path: Union[str, Path],
    metrics: Optional[list] = None
):
    """
    绘制训练历史图表
    
    参数:
        history: 训练历史字典
        output_path: 输出路径
        metrics: 要绘制的指标列表
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        
        if metrics is None:
            metrics = ['loss', 'pearson', 'spearman']
        
        # 创建子图
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        # 绘制每个指标
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # 训练数据
            if 'train' in history and metric in history['train'][0]:
                train_values = [h[metric] for h in history['train']]
                ax.plot(train_values, label='训练集', marker='o', markersize=4)
            
            # 验证数据
            if 'val' in history and metric in history['val'][0]:
                val_values = [h[metric] for h in history['val']]
                ax.plot(val_values, label='验证集', marker='s', markersize=4)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} 变化曲线')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"保存训练历史图表: {output_path}")
        
    except ImportError:
        logging.warning("未安装 matplotlib，跳过绘图")


def create_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    output_path: Union[str, Path],
    bins: int = 5
):
    """
    创建混淆矩阵（用于回归任务的分箱版本）
    
    参数:
        predictions: 预测值
        labels: 真实值
        output_path: 输出路径
        bins: 分箱数量
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        # 将连续值分箱
        pred_bins = pd.cut(predictions, bins=bins, labels=False)
        label_bins = pd.cut(labels, bins=bins, labels=False)
        
        # 计算混淆矩阵
        cm = confusion_matrix(label_bins, pred_bins)
        
        # 绘制
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('预测分箱')
        plt.ylabel('真实分箱')
        plt.title('混淆矩阵（分箱）')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"保存混淆矩阵: {output_path}")
        
    except ImportError:
        logging.warning("未安装必要的绘图库，跳过混淆矩阵")


class AverageMeter:
    """
    计算和存储平均值和当前值的工具类
    """
    
    def __init__(self, name: str = "Meter"):
        self.name = name
        self.reset()
    
    def reset(self):
        """重置所有统计值"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        更新统计值
        
        参数:
            val: 当前值
            n: 样本数
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


if __name__ == "__main__":
    # 测试代码
    import tempfile
    
    # 测试设置种子
    set_seed(42)
    
    # 测试日志设置
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        setup_logging(log_file, "INFO")
        logging.info("测试日志消息")
        
        # 测试实验信息保存
        config = {'model': {'embed_dim': 256}, 'training': {'batch_size': 32}}
        results = {
            'best_metric': 0.85,
            'total_time': 3600,
            'history': {
                'train': [{'loss': 0.5, 'pearson': 0.7}],
                'val': [{'loss': 0.6, 'pearson': 0.8}]
            }
        }
        model_info = {'total_params': 1000000, 'trainable_params': 900000}
        
        save_experiment_info(tmpdir, config, results, model_info)
        
        print(f"测试文件已保存到: {tmpdir}")