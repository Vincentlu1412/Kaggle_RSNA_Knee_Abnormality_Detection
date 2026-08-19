#!/usr/bin/env python
"""
Preprocess RSNA Knee data: create train/val splits, convert DICOM to PNG if needed.
Run: python src/data/preprocess.py
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
import yaml


def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def read_dicom_image(path: Path, target_size=(224, 224)):
    """Read DICOM file and convert to RGB image."""
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        
        # Normalize to 0-255
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        img = img.astype(np.uint8)
        
        # Apply windowing if available
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            wc = ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter
            ww = ds.WindowWidth[0] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth
            img = np.clip(img, wc - ww//2, wc + ww//2)
            img = (img - (wc - ww//2)) / ww * 255
            img = img.astype(np.uint8)
        
        # Convert to RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Resize
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return np.zeros((*target_size, 3), dtype=np.uint8)


def find_images(data_root: Path, split: str):
    """Find all images in train_images or test_images directory."""
    img_dir = data_root / split
    records = []
    
    if not img_dir.exists():
        print(f"⚠️  {img_dir} not found")
        return records
    
    for patient_dir in tqdm(list(img_dir.iterdir()), desc=f"Scanning {split}"):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        for series_dir in patient_dir.iterdir():
            if not series_dir.is_dir():
                continue
            series_id = series_dir.name
            for img_path in series_dir.glob("*.dcm"):
                records.append({
                    'patient_id': patient_id,
                    'series_id': series_id,
                    'image_path': str(img_path.relative_to(data_root)),
                    'full_path': str(img_path)
                })
            # Also check for png/jpg
            for img_path in series_dir.glob("*.png"):
                records.append({
                    'patient_id': patient_id,
                    'series_id': series_id,
                    'image_path': str(img_path.relative_to(data_root)),
                    'full_path': str(img_path)
                })
            for img_path in series_dir.glob("*.jpg"):
                records.append({
                    'patient_id': patient_id,
                    'series_id': series_id,
                    'image_path': str(img_path.relative_to(data_root)),
                    'full_path': str(img_path)
                })
    return records


def create_splits(config):
    """Create train/val splits with stratification."""
    data_root = Path(config['paths']['data_root'])
    processed_root = Path(config['paths']['processed_root'])
    processed_root.mkdir(parents=True, exist_ok=True)
    
    # Load train labels
    train_csv = data_root / config['paths']['train_csv']
    df = pd.read_csv(train_csv)
    
    # Target columns (exclude id columns)
    target_cols = [c for c in df.columns if c not in ['StudyInstanceUID', 'Report']]
    print(f"Target columns: {target_cols}")
    
    # Create stratification key (combination of all targets)
    # df['stratify_key'] = df[target_cols].astype(str).agg('_'.join, axis=1)
    # fill missing labels
    df[target_cols] = df[target_cols].fillna(0)

    # abnormal / normal stratification
    df['abnormal'] = (df[target_cols].sum(axis=1) > 0).astype(int)
    
    # Split
    train_df, val_df = train_test_split(
        df, 
        test_size=config['data']['val_split'],
        stratify=df['abnormal'],
        random_state=config['project']['seed']
    )
    
    train_df = train_df.drop('abnormal', axis=1).reset_index(drop=True)
    val_df = val_df.drop('abnormal', axis=1).reset_index(drop=True)
    
    # Save splits
    train_df.to_csv(processed_root / "train_split.csv", index=False)
    val_df.to_csv(processed_root / "val_split.csv", index=False)
    
    print(f"✅ Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Train target dist:\n{train_df[target_cols].sum()}")
    print(f"Val target dist:\n{val_df[target_cols].sum()}")
    
    return train_df, val_df, target_cols


def create_kfold_splits(config, n_splits=5):
    """Create K-fold splits for cross-validation."""
    data_root = Path(config['paths']['data_root'])
    processed_root = Path(config['paths']['processed_root'])
    
    train_csv = data_root / config['paths']['train_csv']
    df = pd.read_csv(train_csv)
    
    target_cols = [c for c in df.columns if c not in ['StudyInstanceUID', 'Report']
    
    # fill missing labels
    for c in target_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # abnormal / normal stratification
    df['abnormal'] = (df[target_cols].sum(axis=1) > 0).astype(int)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config['project']['seed'])
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['stratify_key'])):
        fold_train = df.iloc[train_idx].drop('stratify_key', axis=1).reset_index(drop=True)
        fold_val = df.iloc[val_idx].drop('stratify_key', axis=1).reset_index(drop=True)
        
        fold_train.to_csv(processed_root / f"train_fold{fold}.csv", index=False)
        fold_val.to_csv(processed_root / f"val_fold{fold}.csv", index=False)
    
    print(f"✅ Created {n_splits}-fold splits")


def convert_dicom_to_png(config):
    """Convert DICOM to PNG for faster loading (optional)."""
    data_root = Path(config['paths']['data_root'])
    processed_root = Path(config['paths']['processed_root'])
    target_size = tuple(config['data']['image_size'])
    
    for split in ["train_images", "test_images"]:
        src_dir = data_root / split
        dst_dir = processed_root / split
        if not src_dir.exists():
            continue
        
        dst_dir.mkdir(parents=True, exist_ok=True)
        records = find_images(data_root, split)
        
        print(f"Converting {split} ({len(records)} images)...")
        for rec in tqdm(records):
            src_path = Path(rec['full_path'])
            rel_path = Path(rec['image_path']).with_suffix('.png')
            dst_path = dst_dir / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not dst_path.exists():
                img = read_dicom_image(src_path, target_size)
                cv2.imwrite(str(dst_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print("✅ DICOM to PNG conversion complete")


def find_images(data_root: Path, split: str):
    """Find all images in train_images or test_images directory."""
    img_dir = data_root / split
    records = []
    
    if not img_dir.exists():
        return records
    
    for patient_dir in img_dir.iterdir():
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        for series_dir in patient_dir.iterdir():
            if not series_dir.is_dir():
                continue
            series_id = series_dir.name
            for img_path in series_dir.glob("*.dcm"):
                records.append({
                    'patient_id': patient_id,
                    'series_id': series_id,
                    'image_path': str(img_path.relative_to(data_root)),
                    'full_path': str(img_path)
                })
            for img_path in series_dir.glob("*.png"):
                records.append({
                    'patient_id': patient_id,
                    'series_id': series_id,
                    'image_path': str(img_path.relative_to(data_root)),
                    'full_path': str(img_path)
                })
            for img_path in series_dir.glob("*.jpg"):
                records.append({
                    'patient_id': patient_id,
                    'series_id': series_id,
                    'image_path': str(img_path.relative_to(data_root)),
                    'full_path': str(img_path)
                })
    return records


def main():
    config = load_config()
    
    print("🔄 Creating train/val splits...")
    create_splits(config)
    
    print("\n🔄 Creating 5-fold CV splits...")
    create_kfold_splits(config, n_splits=5)
    
    # Optional: convert DICOM to PNG (uncomment if needed)
    # print("\n🔄 Converting DICOM to PNG...")
    # convert_dicom_to_png(config)
    
    print("\n✅ Preprocessing complete!")
    print("Next: python src/training/train.py")


if __name__ == "__main__":
    main()
