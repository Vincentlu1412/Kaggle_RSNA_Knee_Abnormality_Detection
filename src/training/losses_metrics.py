"""
Loss functions and metrics for RSNA Knee Abnormality Detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


class BCEWithLogitsLoss(nn.Module):
    """Binary Cross Entropy with Logits Loss with optional pos_weight."""
    
    def __init__(self, pos_weight: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = focal_weight * (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification."""
    
    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 1, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Clipping
        if self.clip is not None and self.clip > 0:
            probs = probs.clamp(min=self.clip, max=1 - self.clip)
        
        # Calculate loss
        pos_loss = targets * torch.log(probs + self.eps) * (1 - probs) ** self.gamma_pos
        neg_loss = (1 - targets) * torch.log(1 - probs + self.eps) * probs ** self.gamma_neg
        
        loss = -(pos_loss + neg_loss)
        return loss.mean()


def get_loss_fn(config: dict, pos_weight: torch.Tensor = None) -> nn.Module:
    """Get loss function from config."""
    loss_name = config['training']['loss']
    
    if loss_name == 'bce_with_logits':
        return BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_name == 'focal':
        return FocalLoss()
    elif loss_name == 'asymmetric':
        return AsymmetricLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def calculate_pos_weight(targets: np.ndarray) -> torch.Tensor:
    """Calculate pos_weight for BCE loss from target distribution."""
    # targets shape: (N, num_classes)
    pos_counts = targets.sum(axis=0)
    neg_counts = len(targets) - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-8)
    return torch.tensor(pos_weight, dtype=torch.float32)


class MetricTracker:
    """Track metrics during training/validation."""
    
    def __init__(self, num_classes: int, target_names: list = None):
        self.num_classes = num_classes
        self.target_names = target_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        self.all_logits = []
        self.all_targets = []
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        self.all_logits.append(logits.detach().cpu())
        self.all_targets.append(targets.detach().cpu())
    
    def compute(self) -> dict:
        if not self.all_logits:
            return {}
        
        logits = torch.cat(self.all_logits, dim=0)
        targets = torch.cat(self.all_targets, dim=0)
        probs = torch.sigmoid(logits).numpy()
        targets_np = targets.numpy()
        
        metrics = {}
        
        # Per-class AUC
        for i, name in enumerate(self.target_names):
            try:
                auc = roc_auc_score(targets_np[:, i], probs[:, i])
                metrics[f'auc_{name}'] = auc
            except ValueError:
                metrics[f'auc_{name}'] = 0.0
        
        # Mean AUC
        auc_values = [v for k, v in metrics.items() if k.startswith('auc_')]
        metrics['auc_mean'] = np.mean(auc_values) if auc_values else 0.0
        
        # Per-class AP
        for i, name in enumerate(self.target_names):
            try:
                ap = average_precision_score(targets_np[:, i], probs[:, i])
                metrics[f'ap_{name}'] = ap
            except ValueError:
                metrics[f'ap_{name}'] = 0.0
        
        ap_values = [v for k, v in metrics.items() if k.startswith('ap_')]
        metrics['ap_mean'] = np.mean(ap_values) if ap_values else 0.0
        
        # Accuracy at threshold 0.5
        preds = (probs > 0.5).astype(int)
        metrics['accuracy'] = (preds == targets_np).mean()
        
        # Per-class accuracy
        for i, name in enumerate(self.target_names):
            metrics[f'acc_{name}'] = (preds[:, i] == targets_np[:, i]).mean()
        
        return metrics
    
    def get_best_metric(self, metric_name: str = 'auc_mean') -> float:
        metrics = self.compute()
        return metrics.get(metric_name, 0.0)


def compute_metrics(logits: np.ndarray, targets: np.ndarray, target_names: list = None) -> dict:
    """Compute all metrics from logits and targets."""
    tracker = MetricTracker(logits.shape[1], target_names)
    tracker.all_logits = [torch.tensor(logits)]
    tracker.all_targets = [torch.tensor(targets)]
    return tracker.compute()