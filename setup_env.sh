#!/bin/bash

# Exit on error
set -e

echo "Setting up virtual environment for LLM-Finetuning-Recipe-Assistant..."

# Check if Python 3.12 is available
python3 --version | grep "3.12" > /dev/null || {
    echo "Python 3.12.x is required. Please install it before continuing."
    exit 1
}

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw data/processed data/processed/generation
mkdir -p models/recipe_assistant
mkdir -p logs/recipe_assistant

# Add .gitkeep files to empty directories to ensure they're tracked by git
touch data/raw/.gitkeep data/processed/.gitkeep 
touch models/recipe_assistant/.gitkeep

echo ""
echo "Setup complete! Virtual environment is now active."
echo "To activate this environment in the future, run: source venv/bin/activate"
echo "To deactivate, run: deactivate"
echo ""
echo "Next steps:"
echo "1. For recipe generation demo:"
echo "   - Prepare dataset: python -m src.data.recipe_prepare_dataset"
echo "   - Train the model: python -m src.model.recipe_train"
echo "2. Run tests: pytest tests/ -v"
echo ""