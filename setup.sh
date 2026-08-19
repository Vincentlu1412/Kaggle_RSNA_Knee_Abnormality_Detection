#!/usr/bin/env bash
# setup.sh - One-click environment setup for RSNA Knee Abnormality Detection

set -e

echo "🚀 Setting up RSNA Knee Abnormality Detection environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "📦 Creating conda environment..."
    conda env create -f environment.yml
    echo "✅ Conda environment created. Activate with: conda activate rsna-knee"
elif command -v mamba &> /dev/null; then
    echo "📦 Creating mamba environment (faster)..."
    mamba env create -f environment.yml
    echo "✅ Mamba environment created. Activate with: conda activate rsna-knee"
else
    echo "⚠️  Conda/Mamba not found. Installing via pip..."
    pip install -r requirements.txt
    echo "✅ Pip packages installed."
fi

# Create necessary directories
mkdir -p data/raw data/processed outputs/models outputs/logs outputs/submissions

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate environment: conda activate rsna-knee"
echo "2. Configure Kaggle API: kaggle config set -n username -v YOUR_USERNAME"
echo "3. Configure Kaggle API: kaggle config set -n key -v YOUR_KEY"
echo "4. Download data: python src/data/download_data.py"
echo "5. Run training: python src/training/train.py"
echo "6. Generate submission: python src/inference/predict.py"