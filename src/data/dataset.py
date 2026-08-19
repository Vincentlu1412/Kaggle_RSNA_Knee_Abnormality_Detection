"""
Dataset and DataLoader for RSNA Knee Abnormality Detection.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


class KneeDataset(Dataset):
    """Dataset for RSNA Knee Abnormality Detection."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        image_size: Tuple[int, int] = (224, 224),
        target_cols: Optional[List[str]] = None,
        transform: Optional[A.Compose] = None,
        is_test: bool = False,
        use_png: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.target_cols = target_cols or []
        self.transform = transform
        self.is_test = is_test
        self.use_png = use_png
        
        # Default normalization (ImageNet)
        self.normalize = A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        )
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_image(self, row: pd.Series) -> np.ndarray:
        study_id = row['StudyInstanceUID']
    
        img_root = self.data_root / (
            "test_series" if self.is_test else "train_series"
        )
    
        study_dir = img_root / str(study_id)
    
        # 找下面所有dcm
        dcm_files = list(study_dir.rglob("*.dcm"))
    
        if len(dcm_files) == 0:
            print("No DICOM:", study_dir)
            img_path = study_dir
        else:
            # 先取第一张，保证跑通
            img_path = dcm_files[0]
    
        return self._load_dicom(img_path)
    
    def _load_dicom(self, path: Path) -> np.ndarray:
        """Load and preprocess DICOM image."""
        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            
            # Normalize
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
            img = img.astype(np.uint8)
            
            # Windowing
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                wc = ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter
                ww = ds.WindowWidth[0] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth
                img = np.clip(img, wc - ww//2, wc + ww//2)
                img = (img - (wc - ww//2)) / ww * 255
                img = img.astype(np.uint8)
            
            # To RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            img = np.zeros((*self.image_size, 3), dtype=np.uint8)
        
        # Resize
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        return img
    
    def _load_regular(self, path: Path) -> np.ndarray:
        """Load PNG/JPG image."""
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((*self.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        return img
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        # Load image
        img = self._load_image(row)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        # Normalize and convert to tensor
        img = self.normalize(image=img)['image']
        img = ToTensorV2()(image=img)['image']
        
        sample = {'image': img, 'id': row.get('id', idx)}
        
        if not self.is_test and self.target_cols:
            targets = row[self.target_cols].values.astype(np.float32)
            sample['targets'] = torch.tensor(targets, dtype=torch.float32)
        
        return sample


def get_train_transforms(image_size: Tuple[int, int], aug_prob: float = 0.5) -> A.Compose:
    """Training augmentations."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.RandomGamma(gamma_limit=(80, 120)),
            A.CLAHE(clip_limit=2),
        ], p=0.3),
        A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            min_holes=1,
            p=0.2
        ),
        A.Resize(image_size[0], image_size[1]),
    ], p=aug_prob)


def get_val_transforms(image_size: Tuple[int, int]) -> A.Compose:
    """Validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
    ])


def get_tta_transforms(image_size: Tuple[int, int]) -> List[A.Compose]:
    """Test Time Augmentation transforms."""
    return [
        A.Compose([A.Resize(image_size[0], image_size[1])]),  # Original
        A.Compose([A.HorizontalFlip(p=1.0), A.Resize(image_size[0], image_size[1])]),  # Flip H
        A.Compose([A.VerticalFlip(p=1.0), A.Resize(image_size[0], image_size[1])]),  # Flip V
        A.Compose([A.RandomRotate90(p=1.0), A.Resize(image_size[0], image_size[1])]),  # Rotate
    ]


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    data_root: str,
    config: dict,
    target_cols: List[str],
    fold: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    train_transform = get_train_transforms(
        tuple(config['data']['image_size']),
        config['data']['aug_prob']
    ) if config['data']['use_augmentation'] else get_val_transforms(tuple(config['data']['image_size']))
    
    val_transform = get_val_transforms(tuple(config['data']['image_size']))
    
    train_dataset = KneeDataset(
        train_df, data_root,
        image_size=tuple(config['data']['image_size']),
        target_cols=target_cols,
        transform=train_transform,
        is_test=False
    )
    
    val_dataset = KneeDataset(
        val_df, data_root,
        image_size=tuple(config['data']['image_size']),
        target_cols=target_cols,
        transform=val_transform,
        is_test=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'] * 2,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_test_dataloader(
    test_df: pd.DataFrame,
    data_root: str,
    config: dict,
    tta: bool = False
) -> DataLoader:
    """Create test dataloader."""
    transform = get_val_transforms(tuple(config['data']['image_size']))
    
    test_dataset = KneeDataset(
        test_df, data_root,
        image_size=tuple(config['data']['image_size']),
        target_cols=[],
        transform=transform,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'] * 2,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return test_loader
