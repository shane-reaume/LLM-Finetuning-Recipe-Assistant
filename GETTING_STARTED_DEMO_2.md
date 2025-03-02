# Getting Started with LLM-Finetuning-Playground - Recipe Generation

**Recipe Generation** - Using a decoder model (TinyLlama) for text generation

## 📋 Prerequisites

- **Python 3.12.3 or later**
- **Git** for version control
- **GPU with 8GB+ VRAM** recommended for training (CPU can be used but will be very slow)
- **Basic Python knowledge** (No ML experience required)
- **Ollama** (optional, for local deployment) - [Install Ollama](https://ollama.ai/download)

## 🔧 Initial Setup

### Step 1: Clone the repository and set up environment

```bash
git clone https://github.com/your-repo/LLM-Finetuning-Playground.git
cd LLM-Finetuning-Playground
chmod +x setup_env.sh  # Only needs to be done once
./setup_env.sh
```

This will:

- Create a virtual environment in the `venv` directory
- Install all dependencies
- Create necessary project directories

### Step 2: Activate the virtual environment
(if not already activated by the script)

```bash
source venv/bin/activate
```

## Recipe Generation Project

This project fine-tunes a small language model (TinyLlama) to generate recipes from ingredient lists. Here's how to use it:

### Step 1: Download the Recipe Dataset

Unlike the sentiment analysis project, the recipe generation model requires a dataset that must be manually downloaded:

```bash
# Create download directory
mkdir -p ~/Downloads/recipe_data
cd ~/Downloads

# Download the Recipe NLG dataset
wget https://recipenlg.blob.core.windows.net/recipenlg-misc/dataset.zip
unzip dataset.zip -d ~/Downloads/recipe_data

# Create directory for the dataset and copy the file
mkdir -p ~/manual_data
cp ~/Downloads/recipe_data/full_dataset.csv ~/manual_data/
```

### Step 2: Prepare the Dataset

Generate a dataset of recipes for training:

```bash
# Specify the data directory when running the command
python -m src.data.recipe_prepare_dataset --data_dir ~/manual_data
```

Alternatively, if you've set up the Makefile:

```bash
# Use the make command with the DATA_DIR variable
make recipe-data DATA_DIR=~/manual_data
```

This processes recipe data and saves it to `data/processed/recipe_dataset.json` and creates test examples at `data/processed/recipe_test_examples.json`.

### Step 3: Train the Model

Start the training process with a LoRA adapter:

```bash
# Basic training command
python -m src.model.recipe_train

# Or with the Makefile
make recipe-train
```

This will:

- Download the TinyLlama base model
- Set up the LoRA training configuration
- Fine-tune the model on recipe generation
- Save the model to `models/recipe`

Training typically takes 2-4 hours on a good GPU.

### Step 4: Evaluate the Model

Assess the model's performance:

```bash
# Basic evaluation command
python -m src.model.recipe_evaluate

# Or with the Makefile
make recipe-evaluate
```

This will generate sample recipes and evaluate:

- Generation quality
- Recipe formatting
- Ingredient usage
- Runtime performance

### Step 5: Try the Interactive Demos

Once your model is trained, you can use two different demo interfaces:

#### CLI Demo
```bash
# Run the command-line interface demo
python -m src.recipe_demo --model_dir="models/recipe"
```

This demo provides a command-line interface where you can enter ingredients and get formatted recipes.

#### Web UI Demo
```bash
# Run the web interface demo
python -m src.recipe_web_demo --model_dir="models/recipe"
```

The web UI demo launches a Gradio interface in your browser, allowing for:

- Entering ingredients to generate recipes
- Adjusting generation parameters
- Seeing formatted recipe outputs

### Step 6 (Optional): Deploy to Ollama

If you have Ollama installed, you can export your model for easy local deployment:

```bash
python -m src.recipe_export_to_ollama_utils --model_dir="models/recipe" --model_name="recipe-gen" --type="template"
```

Then run your model with:

```bash
ollama run recipe-gen
```

## Project Structure for Recipe Generation

- `config/text_generation_improved.yaml`: Configuration file
- `src/model/recipe_train.py`: Training script
- `src/model/recipe_evaluate.py`: Evaluation script
- `src/recipe_demo.py`: CLI demo interface
- `src/recipe_web_demo.py`: Web UI demo interface
- `src/recipe_export_to_ollama_utils.py`: Ollama deployment utilities
- `tests/test_recipe_model.py`: Test suite

## Customization

You can modify the configuration files to change:

- `config/text_generation_improved.yaml`: Training parameters
- `config/recipe_prompt.txt`: Template for recipe generation
- Generation parameters in the demo scripts

Editing these files allows you to experiment with:

- Different base models
- LoRA adapter settings
- Training parameters
- Generation parameters (temperature, top-p, etc.)

## Advanced Usage

### Merging LoRA Adapter with Base Model

For better performance or offline use, you can merge the LoRA adapter with the base model:

```bash
python -m src.model.recipe_merge_and_export_lora --base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lora_model="models/recipe" --output_dir="models/recipe_merged"
```

### Performance Optimization

If you're experiencing slow generation speeds, try:

1. Reducing the context length in the configuration file
2. Using a smaller base model
3. Enabling 8-bit quantization during inference
4. Deploying to Ollama with optimized parameters

## Next Steps

Once you're comfortable with the recipe generation project, consider:

1. **Enhancing the prompts** - Create better system prompts and templates
2. **Fine-tuning on different domains** - Try other cooking styles or cuisines
3. **Improving evaluation metrics** - Develop better ways to assess generation quality
4. **Exploring model merging** - Learn more about model merging techniques
5. **Trying our sentiment analysis demo** - Check GETTING_STARTED_DEMO_1.md

