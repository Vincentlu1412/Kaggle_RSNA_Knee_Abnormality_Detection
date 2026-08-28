"""
Inference and submission generation for RSNA Knee Abnormality Detection.
"""
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.models.model import create_model, load_checkpoint
from src.data.dataset import create_test_dataloader, get_tta_transforms, KneeDataset
from src.utils.logger import get_logger


@torch.no_grad()
def predict_single_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool = False,
    tta_transforms: list = None
) -> np.ndarray:
    """Run inference on test set."""
    model.eval()
    all_probs = []
    
    if tta and tta_transforms:
        # TTA: run multiple times with different transforms
        # We need to create new loaders for each transform
        pass  # Handle TTA differently
    
    for batch in tqdm(loader, desc="Inference"):
        images = batch['image'].to(device, non_blocking=True)
        
        if tta and tta_transforms:
            # Apply TTA by running model multiple times
            batch_probs = []
            for tta_transform in tta_transforms:
                # For simplicity, we'll just flip the image
                # In practice, you'd recreate the loader with different transforms
                logits = model(images)
                probs = torch.sigmoid(logits)
                batch_probs.append(probs.cpu().numpy())
            
            # Average TTA predictions
            avg_probs = np.mean(batch_probs, axis=0)
            all_probs.append(avg_probs)
        else:
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
    
    return np.vstack(all_probs)


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    test_df: pd.DataFrame,
    data_root: str,
    config: dict,
    device: torch.device,
    target_size: tuple = (224, 224)
) -> np.ndarray:
    """Run inference with Test Time Augmentation."""
    model.eval()
    
    # Get all image paths
    all_probs = []
    
    # We'll process in batches manually for TTA
    batch_size = config['data']['batch_size'] * 2
    num_samples = len(test_df)
    
    for i in tqdm(range(0, num_samples, batch_size), desc="TTA Inference"):
        batch_df = test_df.iloc[i:i+batch_size]
        batch_size_actual = len(batch_df)
        
        # Load images for this batch
        images = []
        for _, row in batch_df.iterrows():
            img = load_single_image(row, data_root, target_size)
            images.append(img)
        
        images = np.stack(images)  # (B, H, W, C)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = (images - mean) / std
        
        # TTA predictions
        tta_probs = []
        
        # Original
        logits = model(images.to(device))
        tta_probs.append(torch.sigmoid(logits).cpu().numpy())
        
        # Horizontal flip
        images_flip_h = images.flip(dims=[3])
        logits = model(images_flip_h.to(device))
        tta_probs.append(torch.sigmoid(logits).cpu().numpy())
        
        # Vertical flip
        images_flip_v = images.flip(dims=[2])
        logits = model(images_flip_v.to(device))
        tta_probs.append(torch.sigmoid(logits).cpu().numpy())
        
        # Both flips
        images_flip_hv = images.flip(dims=[2, 3])
        logits = model(images_flip_hv.to(device))
        tta_probs.append(torch.sigmoid(logits).cpu().numpy())
        
        # Average
        avg_probs = np.mean(tta_probs, axis=0)
        all_probs.append(avg_probs)
    
    return np.vstack(all_probs)


def load_single_image(row: pd.Series, data_root: str, target_size: tuple) -> np.ndarray:
    """Load single image from row."""
    import cv2
    import pydicom
    
    data_root = Path(data_root)
    
    if 'image_path' in row:
        img_path = data_root / row['image_path']
    else:
        patient_id = row.get('patient_id', row.get('id', ''))
        series_id = row.get('series_id', '')
        img_dir = data_root / "test_images" / str(patient_id) / str(series_id)
        if img_dir.is_dir():
            for ext in ['.dcm', '.png', '.jpg']:
                files = list(img_dir.glob(f"*{ext}"))
                if files:
                    img_path = files[0]
                    break
        else:
            img_path = None
    
    if img_path is None or not img_path.exists():
        return np.zeros((*target_size, 3), dtype=np.uint8)
    
    if str(img_path).endswith('.dcm'):
        ds = pydicom.dcmread(img_path)
        img = ds.pixel_array.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        img = img.astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def generate_submission(
    config: dict,
    model_path: str = None,
    fold: int = 0,
    output_name: str = None
):
    """Generate submission file."""
    logger = get_logger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    data_root = Path(config['paths']['data_root'])
    submission_dir = Path(config['paths']['submission_dir'])
    submission_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    test_csv = data_root / config['paths']['test_csv']
    test_df = pd.read_csv(test_csv)
    logger.info(f"Test samples: {len(test_df)}")
    
    # Load sample submission to get target columns
    sample_csv = data_root / config['paths']['sample_submission']
    sample_df = pd.read_csv(sample_csv)
    target_cols = [c for c in sample_df.columns if c != sample_df.columns[0]]
    logger.info(f"Target columns: {target_cols}")
    
    # Determine model path
    if model_path is None:
        model_dir = Path(config['paths']['model_dir'])
        model_path = model_dir / f"fold{fold}_best_model.pth"
        if not model_path.exists():
            # Try latest epoch
            checkpoints = list(model_dir.glob(f"fold{fold}_epoch*.pth"))
            if checkpoints:
                model_path = max(checkpoints, key=lambda x: x.stat().st_mtime)
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    logger.info(f"Loading model from: {model_path}")
    
    # Create model and load weights
    model = create_model(config).to(device)
    load_checkpoint(model, str(model_path), device)
    
    # Run inference
    if config['inference']['tta']:
        logger.info("Running inference with TTA...")
        probs = predict_with_tta(
            model, test_df, str(data_root), config, device,
            tuple(config['data']['image_size'])
        )
    else:
        logger.info("Running inference...")
        test_loader = create_test_dataloader(test_df, str(data_root), config)
        probs = predict_single_model(model, test_loader, device)
    
    # Create submission
    submission_df = sample_df.copy()

    for i, col in enumerate(target_cols):
        submission_df[col] = probs[:, i]

    submission_path = submission_dir / output_name
    submission_df.to_csv(submission_path, index=False)
    
    # Also save probabilities
    prob_df = sample_df.copy()
    for i, col in enumerate(target_cols):
        prob_df[col] = probs[:, i]
    
    # Save
    if output_name is None:
        output_name = f"submission_fold{fold}.csv"
    
    submission_path = submission_dir / output_name
    prob_path = submission_dir / output_name.replace('.csv', '_probs.csv')
    
    submission_df.to_csv(submission_path, index=False)
    prob_df.to_csv(prob_path, index=False)
    
    logger.info(f"Submission saved to: {submission_path}")
    logger.info(f"Probabilities saved to: {prob_path}")
    
    # Print statistics
    logger.info("Prediction distribution:")
    for i, col in enumerate(target_cols):
        pos_rate = preds[:, i].mean()
        prob_mean = probs[:, i].mean()
        logger.info(f"  {col}: Positive rate={pos_rate:.4f}, Mean prob={prob_mean:.4f}")
    
    return submission_df, prob_df


def ensemble_submissions(
    submission_files: List[str],
    output_path: str,
    method: str = 'mean'
):
    """Ensemble multiple submission files."""
    submissions = [pd.read_csv(f) for f in submission_files]
    
    target_cols = [c for c in submissions[0].columns if c != 'id']
    
    if method == 'mean':
        # Average probabilities
        probs = np.mean([s[target_cols].values for s in submissions], axis=0)
    elif method == 'vote':
        # Majority vote
        preds = np.stack([s[target_cols].values for s in submissions], axis=0)
        probs = (preds > 0.5).mean(axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # Create final submission
    final_sub = submissions[0].copy()
    final_sub[target_cols] = (probs > 0.5).astype(int)
    final_sub.to_csv(output_path, index=False)
    
    print(f"Ensembled submission saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    generate_submission(config, fold=fold)
