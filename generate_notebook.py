import json
import os

files_to_write = [
    "config.yaml",
    "src/train.py",
    "src/model.py",
    "src/dataset.py",
    "src/datamodule.py",
    "src/data_utils.py",
    "src/lightning_module.py",
    "src/losses.py",
    "src/logging_utils.py",
    "scripts/fetch_data.py",
    "scripts/prepare_dataset.py",
    "inference.py"
]

cells = []

# Header
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🌊 VaayuChakshu — Flood Monitoring (No-ZIP Colab Version)\n", 
        "**Instructions:** Just click **Run All** at the top!\n\n",
        "This notebook will automatically:\n",
        "1. Recreate your entire project codebase on the Colab server.\n",
        "2. Download the satellite dataset.\n",
        "3. Install dependencies.\n",
        "4. Train the model on the T4 GPU."
    ]
})

# Setup folders
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n", 
        "os.makedirs('src', exist_ok=True)\n", 
        "os.makedirs('scripts', exist_ok=True)\n", 
        "os.makedirs('data/raw', exist_ok=True)\n", 
        "os.makedirs('data/processed', exist_ok=True)\n",
        "print('Directories created on Colab.')"
    ]
})

# Write files
for fpath in files_to_write:
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # We need to use %%writefile
        source = [f"%%writefile {fpath}\n"] + [line + "\n" for line in content.split("\n")]
        # Remove trailing newline from last element if needed
        if source[-1].endswith("\n\n"):
            source[-1] = source[-1][:-1]
            
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source
        })

# Install deps
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Install Dependencies"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!pip install -q pytorch-lightning==2.5.2 lightning==2.5.2 wandb==0.20.1 rasterio==1.4.3 albumentations==2.0.8 torchmetrics==1.7.3 onnx==1.18.0 python-dotenv==1.1.0 scikit-image==0.25.2 tifffile==2025.6.11 boto3\n",
        "print('Dependencies installed!')"
    ]
})

# Extract Data
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Extract Dataset\n",
               "**IMPORTANT**: Make sure you have uploaded `data.zip` to Colab before running this cell!"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "if not os.path.exists('data.zip'):\n",
        "    raise FileNotFoundError('Please upload data.zip to Colab first!')\n",
        "!unzip -q -o data.zip\n",
        "print('Dataset extracted successfully!')"
    ]
})

# Prepare Dataset
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Prepare Dataset Manifest\n",
               "This generates the `data_manifest.csv` required for training."]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!python scripts/prepare_dataset.py"
    ]
})

# Train
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 🚀 Start Training!"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import torch\n",
        "import os\n",
        "os.environ['WANDB_MODE'] = 'offline'\n",
        "torch.set_float32_matmul_precision('medium')\n",
        "!python -m src.train --config config.yaml"
    ]
})

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"name": "VaayuChakshu_Colab_NoZip.ipynb", "gpuType": "T4"}
    },
    "cells": cells
}

with open("VaayuChakshu_Colab_NoZip.ipynb", "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated: VaayuChakshu_Colab_NoZip.ipynb")
