"""
训练器模块

提供模型训练的完整流程，包括训练循环、验证、检查点保存、早停等功能。
支持 TensorBoard 和 Weights & Biases 日志记录。

作者: AI Assistant
日期: 2024
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr

from ..configs.config import Config


@dataclass
class TrainingMetrics:
    """训练指标"""
    loss: float
    pearson: float
    spearman: float
    mse: float
    mae: float
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'loss': self.loss,
            'pearson': self.pearson,
            'spearman': self.spearman,
            'mse': self.mse,
            'mae': self.mae
        }


class EarlyStopping:
    """
    早停机制
    
    当验证指标在指定轮数内没有改善时停止训练。
    
    参数:
        patience (int): 耐心值，即允许多少轮没有改善
        min_delta (float): 最小改善阈值
        mode (str): 'min' 或 'max'，指标优化方向
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        参数:
            score: 当前评估指标
            
        返回:
            should_stop: 是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class Trainer:
    """
    模型训练器
    
    提供完整的训练流程管理，包括:
    - 训练和验证循环
    - 指标计算和记录
    - 检查点保存和加载
    - 早停机制
    - 日志记录
    
    参数:
        model: 要训练的模型
        config: 配置对象
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器（可选）
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 设置设备
        self.device = torch.device(config.training.device)
        self.model.to(self.device)
        
        # 设置优化器
        self.optimizer = self._create_optimizer()
        
        # 设置学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 设置损失函数
        self.criterion = nn.MSELoss()
        
        # 混合精度训练
        self.scaler = GradScaler() if config.training.fp16 else None
        
        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            mode='max'  # 基于 Pearson 相关系数
        ) if config.training.early_stopping else None
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = -float('inf')
        self.training_history = defaultdict(list)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        config = self.config.training
        
        if config.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                betas=config.adam_betas,
                eps=config.adam_eps,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                betas=config.adam_betas,
                eps=config.adam_eps,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"未知的优化器: {config.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        config = self.config.training
        
        if config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs * len(self.train_loader),
                eta_min=1e-7
            )
        elif config.scheduler == 'linear':
            total_steps = config.num_epochs * len(self.train_loader)
            
            def linear_schedule(step):
                if step < config.warmup_steps:
                    return step / config.warmup_steps
                return max(0.0, (total_steps - step) / (total_steps - config.warmup_steps))
            
            return optim.lr_scheduler.LambdaLR(self.optimizer, linear_schedule)
        elif config.scheduler == 'constant':
            return None
        else:
            raise ValueError(f"未知的调度器: {config.scheduler}")
    
    def _setup_logging(self):
        """设置日志记录"""
        config = self.config.logging
        
        # TensorBoard
        if config.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(config.tensorboard_dir)
        else:
            self.writer = None
        
        # Weights & Biases
        if config.use_wandb:
            import wandb
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.experiment_name,
                config=self.config.to_dict(),
                tags=config.tags
            )
            self.wandb = wandb
        else:
            self.wandb = None
    
    def train(self) -> Dict[str, Any]:
        """
        执行完整的训练流程
        
        返回:
            results: 包含训练历史和最佳模型的字典
        """
        self.logger.info("开始训练...")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"训练样本数: {len(self.train_loader.dataset)}")
        self.logger.info(f"验证样本数: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config.training.num_epochs):
                self.current_epoch = epoch
                
                # 训练一个 epoch
                train_metrics = self._train_epoch()
                self.training_history['train'].append(train_metrics.to_dict())
                
                # 验证
                val_metrics = self._validate()
                self.training_history['val'].append(val_metrics.to_dict())
                
                # 记录日志
                self._log_metrics(train_metrics, val_metrics, epoch)
                
                # 保存检查点
                if self._should_save_checkpoint(val_metrics):
                    self._save_checkpoint(val_metrics)
                
                # 早停检查
                if self.early_stopping and self.early_stopping(val_metrics.pearson):
                    self.logger.info(f"早停触发，在 epoch {epoch + 1}")
                    break
            
            # 训练结束
            total_time = time.time() - start_time
            self.logger.info(f"训练完成，总用时: {total_time:.2f}秒")
            
            # 加载最佳模型并在测试集上评估
            if self.test_loader:
                self.load_checkpoint(self.checkpoint_dir / "best_model.pt")
                test_metrics = self._test()
                self.training_history['test'] = test_metrics.to_dict()
                self.logger.info(f"测试集结果: {test_metrics.to_dict()}")
            
        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
        except Exception as e:
            self.logger.error(f"训练出错: {e}")
            raise
        finally:
            # 清理资源
            if self.writer:
                self.writer.close()
            if self.wandb:
                self.wandb.finish()
        
        return {
            'history': self.training_history,
            'best_metric': self.best_metric,
            'total_time': total_time if 'total_time' in locals() else None
        }
    
    def _train_epoch(self) -> TrainingMetrics:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs}",
            leave=True
        )
        
        for batch in progress_bar:
            # 将数据移到设备上
            input_ids_a = batch['input_ids_a'].to(self.device)
            input_ids_b = batch['input_ids_b'].to(self.device)
            attention_mask_a = batch.get('attention_mask_a', None)
            attention_mask_b = batch.get('attention_mask_b', None)
            labels = batch['label'].to(self.device)
            
            if attention_mask_a is not None:
                attention_mask_a = attention_mask_a.to(self.device)
            if attention_mask_b is not None:
                attention_mask_b = attention_mask_b.to(self.device)
            
            # 前向传播
            if self.config.training.fp16 and self.scaler:
                with autocast():
                    similarity = self.model(
                        input_ids_a, input_ids_b,
                        attention_mask_a, attention_mask_b
                    )
                    loss = self.criterion(similarity, labels)
            else:
                similarity = self.model(
                    input_ids_a, input_ids_b,
                    attention_mask_a, attention_mask_b
                )
                loss = self.criterion(similarity, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.config.training.fp16 and self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                self.optimizer.step()
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 记录
            total_loss += loss.item()
            all_predictions.extend(similarity.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
            
            self.global_step += 1
        
        # 计算指标
        metrics = self._compute_metrics(all_predictions, all_labels, total_loss / len(self.train_loader))
        
        return metrics
    
    def _validate(self) -> TrainingMetrics:
        """验证模型"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证中", leave=False):
                # 将数据移到设备上
                input_ids_a = batch['input_ids_a'].to(self.device)
                input_ids_b = batch['input_ids_b'].to(self.device)
                attention_mask_a = batch.get('attention_mask_a', None)
                attention_mask_b = batch.get('attention_mask_b', None)
                labels = batch['label'].to(self.device)
                
                if attention_mask_a is not None:
                    attention_mask_a = attention_mask_a.to(self.device)
                if attention_mask_b is not None:
                    attention_mask_b = attention_mask_b.to(self.device)
                
                # 前向传播
                similarity = self.model(
                    input_ids_a, input_ids_b,
                    attention_mask_a, attention_mask_b
                )
                loss = self.criterion(similarity, labels)
                
                # 记录
                total_loss += loss.item()
                all_predictions.extend(similarity.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        metrics = self._compute_metrics(all_predictions, all_labels, total_loss / len(self.val_loader))
        
        return metrics
    
    def _test(self) -> TrainingMetrics:
        """在测试集上评估模型"""
        if not self.test_loader:
            raise ValueError("没有提供测试数据加载器")
        
        self.model.eval()
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="测试中"):
                # 将数据移到设备上
                input_ids_a = batch['input_ids_a'].to(self.device)
                input_ids_b = batch['input_ids_b'].to(self.device)
                attention_mask_a = batch.get('attention_mask_a', None)
                attention_mask_b = batch.get('attention_mask_b', None)
                
                if attention_mask_a is not None:
                    attention_mask_a = attention_mask_a.to(self.device)
                if attention_mask_b is not None:
                    attention_mask_b = attention_mask_b.to(self.device)
                
                # 前向传播
                similarity = self.model(
                    input_ids_a, input_ids_b,
                    attention_mask_a, attention_mask_b
                )
                
                all_predictions.extend(similarity.cpu().numpy())
        
        # 如果有标签，计算指标
        if 'label' in batch:
            labels = []
            for batch in self.test_loader:
                labels.extend(batch['label'].numpy())
            metrics = self._compute_metrics(all_predictions, labels, 0.0)
        else:
            # 没有标签，返回虚拟指标
            metrics = TrainingMetrics(
                loss=0.0,
                pearson=0.0,
                spearman=0.0,
                mse=0.0,
                mae=0.0
            )
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: List[float],
        labels: List[float],
        loss: float
    ) -> TrainingMetrics:
        """计算评估指标"""
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Pearson 相关系数
        pearson_corr, _ = pearsonr(predictions, labels)
        
        # Spearman 相关系数
        spearman_corr, _ = spearmanr(predictions, labels)
        
        # MSE 和 MAE
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        
        return TrainingMetrics(
            loss=loss,
            pearson=pearson_corr,
            spearman=spearman_corr,
            mse=mse,
            mae=mae
        )
    
    def _log_metrics(
        self,
        train_metrics: TrainingMetrics,
        val_metrics: TrainingMetrics,
        epoch: int
    ):
        """记录指标"""
        # 控制台日志
        self.logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {train_metrics.loss:.4f}, "
            f"Val Loss: {val_metrics.loss:.4f}, "
            f"Val Pearson: {val_metrics.pearson:.4f}, "
            f"Val Spearman: {val_metrics.spearman:.4f}"
        )
        
        # TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/train', train_metrics.loss, epoch)
            self.writer.add_scalar('Loss/val', val_metrics.loss, epoch)
            self.writer.add_scalar('Pearson/train', train_metrics.pearson, epoch)
            self.writer.add_scalar('Pearson/val', val_metrics.pearson, epoch)
            self.writer.add_scalar('Spearman/train', train_metrics.spearman, epoch)
            self.writer.add_scalar('Spearman/val', val_metrics.spearman, epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        # Weights & Biases
        if self.wandb:
            self.wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics.loss,
                'train/pearson': train_metrics.pearson,
                'train/spearman': train_metrics.spearman,
                'val/loss': val_metrics.loss,
                'val/pearson': val_metrics.pearson,
                'val/spearman': val_metrics.spearman,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
    
    def _should_save_checkpoint(self, val_metrics: TrainingMetrics) -> bool:
        """判断是否应该保存检查点"""
        if self.config.training.save_best_only:
            return val_metrics.pearson > self.best_metric
        else:
            return self.global_step % self.config.training.save_steps == 0
    
    def _save_checkpoint(self, val_metrics: TrainingMetrics):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'val_metrics': val_metrics.to_dict(),
            'config': self.config.to_dict()
        }
        
        # 保存检查点
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"保存检查点: {checkpoint_path}")
        
        # 如果是最佳模型，额外保存
        if val_metrics.pearson > self.best_metric:
            self.best_metric = val_metrics.pearson
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型: {best_path} (Pearson: {self.best_metric:.4f})")
        
        # 限制检查点数量
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """清理旧的检查点"""
        if self.config.training.save_total_limit:
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            
            while len(checkpoints) > self.config.training.save_total_limit:
                oldest = checkpoints.pop(0)
                oldest.unlink()
                self.logger.info(f"删除旧检查点: {oldest}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """加载检查点"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        self.logger.info(f"加载检查点: {checkpoint_path}")
        self.logger.info(f"恢复到 epoch {self.current_epoch}, step {self.global_step}")
    
    def predict(
        self,
        data_loader: DataLoader,
        return_embeddings: bool = False
    ) -> Union[List[float], Tuple[List[float], List[np.ndarray]]]:
        """
        对数据进行预测
        
        参数:
            data_loader: 数据加载器
            return_embeddings: 是否返回嵌入向量
        
        返回:
            predictions: 预测的相似度分数
            embeddings: 嵌入向量（如果 return_embeddings=True）
        """
        self.model.eval()
        
        all_predictions = []
        all_embeddings_a = []
        all_embeddings_b = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="预测中"):
                # 将数据移到设备上
                input_ids_a = batch['input_ids_a'].to(self.device)
                input_ids_b = batch['input_ids_b'].to(self.device)
                attention_mask_a = batch.get('attention_mask_a', None)
                attention_mask_b = batch.get('attention_mask_b', None)
                
                if attention_mask_a is not None:
                    attention_mask_a = attention_mask_a.to(self.device)
                if attention_mask_b is not None:
                    attention_mask_b = attention_mask_b.to(self.device)
                
                # 前向传播
                if return_embeddings:
                    similarity, (emb_a, emb_b) = self.model(
                        input_ids_a, input_ids_b,
                        attention_mask_a, attention_mask_b,
                        return_embeddings=True
                    )
                    all_embeddings_a.extend(emb_a.cpu().numpy())
                    all_embeddings_b.extend(emb_b.cpu().numpy())
                else:
                    similarity = self.model(
                        input_ids_a, input_ids_b,
                        attention_mask_a, attention_mask_b
                    )
                
                all_predictions.extend(similarity.cpu().numpy())
        
        if return_embeddings:
            return all_predictions, (all_embeddings_a, all_embeddings_b)
        else:
            return all_predictions


if __name__ == "__main__":
    # 测试代码
    from ..models import TextSimilarityModel
    from ..data.dataset import create_data_loaders, build_vocab_from_dataset
    from torchtext.data.utils import get_tokenizer
    
    # 创建配置
    config = Config()
    config.data.train_path = "tutorial/mha-lstm/data/sts-kaggle-train.csv"
    config.data.val_path = "tutorial/mha-lstm/data/sts-kaggle-test.csv"
    config.training.num_epochs = 2
    config.training.device = 'cpu'
    
    # 准备数据
    tokenizer = get_tokenizer(config.data.tokenizer)
    vocab = build_vocab_from_dataset([config.data.train_path], tokenizer)
    
    train_loader, val_loader, _ = create_data_loaders(
        config.data.train_path,
        config.data.val_path,
        None,
        tokenizer,
        vocab,
        batch_size=config.training.batch_size
    )
    
    # 创建模型
    model = TextSimilarityModel(
        vocab_size=len(vocab),
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers
    )
    
    # 创建训练器
    trainer = Trainer(model, config, train_loader, val_loader)
    
    # 训练
    results = trainer.train()
    print(f"训练完成，最佳 Pearson 相关系数: {results['best_metric']:.4f}")