"""
Training loop for RSNA Knee Abnormality Detection.
"""
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import yaml
import numpy as np

from src.models.model import create_model
from src.data.dataset import create_dataloaders
from src.training.losses_metrics import get_loss_fn, calculate_pos_weight, MetricTracker
from src.utils.logger import get_logger


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    config: dict,
    epoch: int,
    logger
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(loader)
    
    grad_accum_steps = config['training']['grad_accum_steps']
    use_amp = config['training']['use_amp']
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        
        if use_amp:
            with autocast():
                logits = model(images)
                loss = loss_fn(logits, targets)
                loss = loss / grad_accum_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss = loss / grad_accum_steps
            
            loss.backward()
            
            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps
        
        if i % config['logging']['log_interval'] == 0:
            logger.info(f"Epoch {epoch} [{i}/{num_batches}] Loss: {loss.item() * grad_accum_steps:.4f}")
    
    return {'train_loss': total_loss / num_batches}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    target_names: list,
    logger
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = len(loader)
    
    tracker = MetricTracker(len(target_names), target_names)
    
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        
        logits = model(images)
        loss = loss_fn(logits, targets)
        
        total_loss += loss.item()
        tracker.update(logits, targets)
    
    metrics = tracker.compute()
    metrics['val_loss'] = total_loss / num_batches
    
    logger.info(f"Validation: Loss={metrics['val_loss']:.4f}, AUC={metrics['auc_mean']:.4f}, AP={metrics['ap_mean']:.4f}")
    for name in target_names:
        logger.info(f"  {name}: AUC={metrics.get(f'auc_{name}', 0):.4f}, AP={metrics.get(f'ap_{name}', 0):.4f}")
    
    return metrics


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer."""
    lr = float(config['training']['lr'])
    weight_decay = config['training']['weight_decay']
    
    # Separate backbone and classifier parameters for different LR
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for backbone
        {'params': classifier_params, 'lr': lr}
    ]
    
    return AdamW(param_groups, weight_decay=weight_decay)


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict, num_epochs: int):
    """Create learning rate scheduler."""
    warmup_epochs = config['training']['warmup_epochs']
    min_lr = config['training']['min_lr']
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=min_lr
    )
    
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    metrics: Dict,
    config: dict,
    path: Path,
    is_best: bool = False
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'config': config
    }
    
    torch.save(checkpoint, path)
    
    if is_best:
        best_path = path.parent / 'best_model.pth'
        torch.save(checkpoint, best_path)


def load_checkpoint_for_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    path: Path
) -> tuple:
    """Load checkpoint for resuming training."""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']


def train(config: dict, fold: int = 0, resume_from: Optional[str] = None):
    """Main training function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger(__name__)
    
    # Paths
    processed_root = Path(config['paths']['processed_root'])
    model_dir = Path(config['paths']['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data splits
    train_df = pd.read_csv(processed_root / f"train_fold{fold}.csv")
    val_df = pd.read_csv(processed_root / f"val_fold{fold}.csv")
    
    target_cols = [c for c in train_df.columns if c not in ['id', 'series_id', 'study_id']]
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_df, val_df,
        config['paths']['data_root'],
        config,
        target_cols,
        fold
    )
    
    # Create model
    model = create_model(config).to(device)
    
    # Calculate pos_weight
    # train_targets = train_df[target_cols].values
    train_targets = (
        train_df[target_cols]
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
        .values
        .astype(np.float32)
    )
    pos_weight = calculate_pos_weight(train_targets).to(device)
    
    # Loss, optimizer, scheduler
    loss_fn = get_loss_fn(config, pos_weight)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, config['training']['epochs'])
    scaler = GradScaler() if config['training']['use_amp'] else None
    
    # Resume if specified
    start_epoch = 0
    best_metric = 0.0
    if resume_from:
        start_epoch, _ = load_checkpoint_for_resume(
            model, optimizer, scheduler, scaler, Path(resume_from)
        )
        start_epoch += 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info(f"Starting training for fold {fold} on {device}")
    logger.info(f"Model: {config['model']['name']}, Epochs: {config['training']['epochs']}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    patience_counter = 0
    monitor = config['training']['monitor']
    mode = config['training']['mode']
    
    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, config, epoch, logger
        )
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device, target_cols, logger)
        
        # Step scheduler
        scheduler.step()
        
        # Combine metrics
        all_metrics = {**train_metrics, **val_metrics}
        all_metrics['lr'] = optimizer.param_groups[0]['lr']
        all_metrics['epoch'] = epoch
        
        # Log
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")
        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}, AUC: {val_metrics['auc_mean']:.4f}")
        
        # Save checkpoint
        current_metric = val_metrics.get(monitor, 0)
        is_best = False
        
        if mode == 'max' and current_metric > best_metric:
            best_metric = current_metric
            is_best = True
            patience_counter = 0
        elif mode == 'min' and current_metric < best_metric:
            best_metric = current_metric
            is_best = True
            patience_counter = 0
        else:
            patience_counter += 1
        
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, all_metrics,
            config, model_dir / f"fold{fold}_epoch{epoch}.pth", is_best
        )
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    logger.info(f"Training completed. Best {monitor}: {best_metric:.4f}")
    return best_metric


if __name__ == "__main__":
    import pandas as pd
    
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Train fold 0 by default
    train(config, fold=0)
