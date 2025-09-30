#!/bin/bash

# Local installation script for ARC25 dependencies
# Uses wheels from making-wheels-of-necessary-packages-for-vllm dataset

WHEELS_DIR="kaggle/input/making-wheels-of-necessary-packages-for-vllm"

echo "🔧 Installing libraries from wheels..."

# Check if vllm is already installed
if ! python -c "import vllm" &> /dev/null; then
    echo "�� Installing vllm and dependencies..."
    pip uninstall -q -y torch
    # Install core packages from wheels
    pip install --no-index --find-links="$WHEELS_DIR" vllm
    pip install --no-index --find-links="$WHEELS_DIR" peft
    pip install --no-index --find-links="$WHEELS_DIR" trl
    pip install --no-index --find-links="$WHEELS_DIR" bitsandbytes
    pip install --no-index --find-links="$WHEELS_DIR" GPUtil
    pip install --no-index --find-links="$WHEELS_DIR" transformers
    pip install --no-index --find-links="$WHEELS_DIR" termcolor
    pip install --no-index --find-links="$WHEELS_DIR" accelerate --upgrade
    
    echo "✅ Core packages installed successfully"
else
    echo "✅ vllm is already installed"
fi

# Install additional useful packages
echo "📦 Installing additional packages..."
pip install --no-index --find-links="$WHEELS_DIR" psutil matplotlib tqdm

# Install packages needed for fine-tuning
echo "📦 Installing fine-tuning dependencies..."
pip install wandb tensorboard

echo "🎉 Installation complete!"
