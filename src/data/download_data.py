#!/usr/bin/env python
"""
Download and explore RSNA Knee Abnormality Detection data.
Run: python src/data/download_data.py
"""
import os
import sys
import zipfile
from pathlib import Path

import kaggle
import pandas as pd
from tqdm import tqdm


def setup_kaggle():
    """Check Kaggle API credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ Kaggle API credentials not found!")
        print("Please create ~/.kaggle/kaggle.json with your credentials:")
        print('{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}')
        print("Get your key from: https://www.kaggle.com/settings/account")
        sys.exit(1)
    os.chmod(kaggle_json, 0o600)


def download_competition_data(competition: str, path: str):
    """Download competition data using Kaggle API."""
    print(f"📥 Downloading {competition} data to {path}...")
    os.makedirs(path, exist_ok=True)
    kaggle.api.competition_download_files(competition, path=path, quiet=False)
    
    # Extract zip files
    for zip_file in Path(path).glob("*.zip"):
        print(f"📦 Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(path)
        zip_file.unlink()  # Remove zip after extraction
    print("✅ Download and extraction complete!")


def explore_data(data_root: str):
    """Explore and print data statistics."""
    data_root = Path(data_root)
    
    # Check train.csv
    train_csv = data_root / "train.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        print("\n📊 Train CSV Info:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample rows:\n{df.head()}")
        print(f"\n  Target distribution:")
        for col in df.columns:
            if col not in ['id', 'series_id', 'study_id']:
                print(f"    {col}: {df[col].value_counts().to_dict()}")
    
    # Check test.csv
    test_csv = data_root / "test.csv"
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        print(f"\n📊 Test CSV Info:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample rows:\n{df.head()}")
    
    # Check sample submission
    sample_csv = data_root / "sample_submission.csv"
    if sample_csv.exists():
        df = pd.read_csv(sample_csv)
        print(f"\n📊 Sample Submission Info:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample rows:\n{df.head()}")
    
    # Check image directories
    for split in ["train_images", "test_images"]:
        img_dir = data_root / split
        if img_dir.exists():
            patients = list(img_dir.iterdir())
            print(f"\n📁 {split}: {len(patients)} patients")
            if patients:
                sample_patient = patients[0]
                series = list(sample_patient.iterdir())
                print(f"  Example patient {sample_patient.name}: {len(series)} series")
                if series:
                    sample_series = series[0]
                    images = list(sample_series.glob("*.dcm")) + list(sample_series.glob("*.png")) + list(sample_series.glob("*.jpg"))
                    print(f"  Example series {sample_series.name}: {len(images)} images")


def main():
    import yaml
    
    config_path = "configs/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    competition = config['project']['competition']
    data_root = config['paths']['data_root']
    
    setup_kaggle()
    download_competition_data(competition, data_root)
    explore_data(data_root)
    
    print("\n✅ Data ready! Next: python src/data/preprocess.py")


if __name__ == "__main__":
    main()