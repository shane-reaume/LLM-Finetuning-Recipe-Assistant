# Getting Started with LLM-Finetuning-Playground - Recipe Generation

**Recipe Generation** - Using a decoder model (TinyLlama) for text generation

## 🌟 Overview

This demo shows how to fine-tune a small language model to generate cooking recipes from ingredient lists. The process includes:

1. **Dataset preparation** - Download and process the RecipeNLG dataset
2. **Model fine-tuning** - Train TinyLlama with LoRA adapters
3. **Evaluation** - Test the model's performance
4. **Deployment** - Export to Ollama for local use

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
# Create directory for the dataset
mkdir -p ~/recipe_manual_data

# Visit the official website and download the dataset
# https://recipenlg.cs.put.poznan.pl/dataset

# After downloading, place full_dataset.csv in your recipe_manual_data directory
```

The dataset can be found at the official project website: [https://recipenlg.cs.put.poznan.pl/dataset](https://recipenlg.cs.put.poznan.pl/dataset)

> **Note:** The dataset is approximately 600MB in size and contains over 2 million recipes.

### Step 2: Prepare the Dataset

After downloading the dataset, prepare it for training:

```bash
# Specify the data directory when running the command
python -m src.data.recipe_prepare_dataset --data_dir ~/recipe_manual_data
```

Alternatively, use the Makefile:

```bash
# For the manually downloaded dataset
make recipe-data CONFIG=config/text_generation.yaml DATA_DIR=~/recipe_manual_data
```

This processes recipe data and creates test examples at `data/processed/recipe_test_examples.json`. The configuration file (`config/text_generation.yaml`) controls the dataset processing parameters and output locations.

### Step 3: Train the Model

Before running the full training process (which can take 2-4 hours), you may want to perform a quick sanity test to ensure everything is configured correctly:

#### Option A: Run a Quick Sanity Test (Recommended)

```bash
# Run a minimal training session for testing purposes
make recipe-train-test DATA_DIR=~/recipe_manual_data
```

This will:
- Process only 50 training examples
- Train for just 1% of an epoch
- Use reduced sequence lengths
- Complete in a few minutes rather than hours

If the sanity test runs successfully, you can proceed with the full training.

> **Troubleshooting Memory Issues**: If you encounter a "CUDA out of memory" error, your GPU may not have enough VRAM. Try these solutions:
> 1. Edit `config/text_generation_sanity_test.yaml` and reduce `batch_size` to 1
> 2. Further reduce `max_length` to 32
> 3. Add `"offload_modules": true` to the LoRA configuration
> 4. If errors persist, run the CPU version of the sanity test (very slow but works on any machine):
>    ```bash
>    make recipe-train-test-cpu DATA_DIR=~/recipe_manual_data
>    ```

#### Option B: Run Full Training

Start the training process with a LoRA adapter:

```bash
# Run the full training process
make recipe-train DATA_DIR=~/recipe_manual_data
```

This will:

- Download the TinyLlama base model
- Set up the LoRA training configuration
- Fine-tune the model on recipe generation
- Save the model to `models/recipe_assistant`

Training typically takes 2-4 hours on a good GPU.

### Step 4: Evaluate the Model

Assess the model's performance:

```bash
# Basic evaluation command
python -m src.model.recipe_evaluate --config config/text_generation.yaml

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
python -m src.recipe_demo --model_dir="models/recipe_assistant"
```

This demo provides a command-line interface where you can enter ingredients and get formatted recipes.

#### Web UI Demo
```bash
# Run the web interface demo
python -m src.recipe_web_demo --model_dir="models/recipe_assistant"
```

The web UI demo launches a Gradio interface in your browser, allowing for:

- Entering ingredients to generate recipes
- Adjusting generation parameters
- Seeing formatted recipe outputs

### Step 6 (Optional): Deploy to Ollama

If you have Ollama installed, you can export your model for easy local deployment:

```bash
python -m src.model.recipe_export_to_ollama --model_dir="models/recipe_assistant" --model_name="recipe-gen"
```

Alternatively, use the Makefile target:

```bash
make recipe-export
```

Then run your model with:

```bash
ollama run recipe-gen
```

## Project Structure for Recipe Generation

- `config/text_generation.yaml`: Configuration file for training and evaluation
- `config/text_generation_sanity_test.yaml`: Configuration for quick sanity testing
- `config/recipe_prompt.txt`: Template for recipe formatting
- `src/data/recipe_prepare_dataset.py`: Dataset preparation script
- `src/model/recipe_train.py`: Training script
- `src/model/recipe_evaluate.py`: Evaluation script
- `src/recipe_demo.py`: CLI demo interface
- `src/recipe_web_demo.py`: Web UI demo interface
- `src/model/recipe_export_to_ollama.py`: Ollama deployment utilities
- `tests/test_recipe_model.py`: Test suite
- `data/processed/recipe_test_examples.json`: Test examples for evaluation
- `models/recipe_assistant/`: Directory for the trained model checkpoints

## Customization

You can modify the configuration files to change how the recipe generation model works:

### Main Configuration File (`config/text_generation.yaml`):

- **Data Settings**:
  - `dataset_name`: The dataset to use (default: "recipe_nlg")
  - `cache_dir`: Where processed data is stored (`./data/processed/generation`)
  - `max_length`: Maximum sequence length (default: 512)
  - `max_train_samples`: Optional limit on training examples (for testing)

- **Model Settings**:
  - `name`: Base model to fine-tune (default: "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
  - `save_dir`: Where to save the trained model (`./models/recipe_assistant`)

- **Training Settings**:
  - Learning parameters (batch size, learning rate, etc.)
  - LoRA configuration (rank, alpha, target modules)
  - `gradient_checkpointing`: Reduces memory usage but slows training
  - `fp16`: Uses half-precision for better performance

- **Testing Settings**:
  - `test_examples_file`: Path to test examples (`data/processed/recipe_test_examples.json`)
  - Generation parameters (temperature, top_p, etc.)

### Sanity Test Configuration (`config/text_generation_sanity_test.yaml`):

This is a minimal version of the main configuration designed for quick testing:
  - Uses only 50 training examples
  - Trains for just 1% of an epoch
  - Uses much shorter sequence lengths (64 tokens)
  - Employs memory optimizations like gradient checkpointing
  - Uses a smaller batch size to reduce memory requirements
  - Saves output to a different directory (`./models/recipe_assistant_test`)

### Recipe Prompt Template (`config/recipe_prompt.txt`):

You can edit this file to change how recipes are formatted during generation.

Editing these files allows you to experiment with:

- Different base models
- LoRA adapter settings
- Training parameters
- Generation parameters (temperature, top-p, etc.)

## Advanced Usage

### Merging LoRA Adapter with Base Model

For better performance or offline use, you can merge the LoRA adapter with the base model:

```bash
python -m src.model.recipe_merge_and_export_lora --base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lora_model="models/recipe_assistant" --output_dir="models/recipe_merged"
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

## Testing

Testing is a critical aspect of model development. The recipe generation model can be tested using:

```bash
# Run the recipe model tests
pytest tests/test_recipe_model.py -v

# Generate coverage report
pytest --cov=src --cov-report=html
```

The HTML coverage report will be generated in the `htmlcov/` directory and can be viewed in a browser.

For a comprehensive guide to our testing philosophy, methodology, and approach to ML model testing, please refer to the detailed testing documentation in [GETTING_STARTED_DEMO_1.md](GETTING_STARTED_DEMO_1.md#testing-philosophy-and-methodology).