"""
Model definitions for RSNA Knee Abnormality Detection.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class KneeClassifier(nn.Module):
    """Multi-label classifier for knee abnormality detection."""
    
    def __init__(
        self,
        model_name: str = "tf_efficientnet_b0_ns",
        num_classes: int = 3,
        pretrained: bool = True,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.1,
        in_chans: int = 3
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Create backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool='',  # Remove global pooling
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, in_chans, 224, 224)
            features = self.backbone(dummy)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            if len(features.shape) == 4:
                # Global average pooling
                feat_dim = features.shape[1]
            else:
                feat_dim = features.shape[-1]
        
        self.feature_dim = feat_dim
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        
        # Handle different backbone outputs
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        # Global pooling if needed
        if len(features.shape) == 4:
            features = self.global_pool(features)
            features = features.flatten(1)
        elif len(features.shape) == 3:
            features = features.mean(dim=1)
        
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if len(features.shape) == 4:
            features = self.global_pool(features)
            features = features.flatten(1)
        elif len(features.shape) == 3:
            features = features.mean(dim=1)
        return features


class KneeClassifierEnsemble(nn.Module):
    """Ensemble of multiple models."""
    
    def __init__(self, models: list):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = [model(x) for model in self.models]
        return torch.stack(logits).mean(dim=0)


def create_model(config: dict) -> nn.Module:
    """Factory function to create model from config."""
    model_cfg = config['model']
    return KneeClassifier(
        model_name=model_cfg['name'],
        num_classes=model_cfg['num_classes'],
        pretrained=model_cfg['pretrained'],
        drop_rate=model_cfg['drop_rate'],
        drop_path_rate=model_cfg['drop_path_rate']
    )


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return checkpoint


if __name__ == "__main__":
    # Test model creation
    model = KneeClassifier()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Model: {model.__class__.__name__}")
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Feature dim: {model.feature_dim}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")