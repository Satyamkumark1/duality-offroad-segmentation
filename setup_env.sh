#!/bin/bash

# Duality Offroad Segmentation - Environment Setup Script for Mac/Linux
echo "Setting up Python environment for Duality Offroad Segmentation Challenge..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv EDU

# Activate virtual environment
echo "Activating virtual environment..."
source EDU/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for compatibility, can be changed to CUDA if GPU available)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install computer vision libraries
echo "Installing computer vision libraries..."
pip install opencv-python
pip install Pillow
pip install scikit-image

# Install data science libraries
echo "Installing data science libraries..."
pip install numpy
pip install matplotlib
pip install seaborn
pip install pandas

# Install deep learning utilities
echo "Installing deep learning utilities..."
pip install tqdm
pip install tensorboard

# Install image segmentation specific libraries
echo "Installing segmentation libraries..."
pip install segmentation-models-pytorch
pip install albumentations

# Install utility libraries
echo "Installing utility libraries..."
pip install scikit-learn
pip install scipy

echo "Environment setup complete!"
echo "To activate the environment, run: source EDU/bin/activate"
