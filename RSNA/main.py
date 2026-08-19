#!/usr/bin/env python
"""
Main entry point for RSNA Knee Abnormality Detection.
Usage:
  python main.py download       # Download data
  python main.py preprocess     # Preprocess data
  python main.py train [fold]   # Train model (default fold=0)
  python main.py predict [fold] # Generate submission (default fold=0)
  python main.py ensemble       # Ensemble submissions
"""
import sys
import subprocess
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"✅ {description} completed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "download":
        run_command([sys.executable, "src/data/download_data.py"], "Downloading data")
    
    elif command == "preprocess":
        run_command([sys.executable, "src/data/preprocess.py"], "Preprocessing data")
    
    elif command == "train":
        fold = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        run_command([sys.executable, "src/training/train.py", str(fold)], f"Training fold {fold}")
    
    elif command == "predict":
        fold = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        run_command([sys.executable, "src/inference/predict.py", str(fold)], f"Generating submission for fold {fold}")
    
    elif command == "ensemble":
        # Ensemble all fold submissions
        from src.inference.predict import ensemble_submissions
        sub_dir = Path("outputs/submissions")
        files = list(sub_dir.glob("submission_fold*.csv"))
        if files:
            ensemble_submissions([str(f) for f in files], str(sub_dir / "submission_ensemble.csv"))
            print("✅ Ensemble complete!")
        else:
            print("❌ No submission files found to ensemble")
    
    elif command == "all":
        # Run full pipeline
        run_command([sys.executable, "src/data/download_data.py"], "Downloading data")
        run_command([sys.executable, "src/data/preprocess.py"], "Preprocessing data")
        for fold in range(5):
            run_command([sys.executable, "src/training/train.py", str(fold)], f"Training fold {fold}")
            run_command([sys.executable, "src/inference/predict.py", str(fold)], f"Predicting fold {fold}")
        run_command([sys.executable, "main.py", "ensemble"], "Ensembling submissions")
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()