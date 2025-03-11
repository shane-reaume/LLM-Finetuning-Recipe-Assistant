# Getting Started with LLM-Finetuning-Recipe-Assistant

This guide provides detailed instructions for training, testing, and deploying the recipe generation model. It's designed to highlight quality assurance practices throughout the ML lifecycle.

## Table of Contents
- [Project Overview](#project-overview)
- [The ML Development and QA Lifecycle](#the-ml-development-and-qa-lifecycle)
- [Setup and Installation](#setup-and-installation)
- [Step 1: Dataset Preparation (with QA)](#step-1-dataset-preparation-with-qa)
- [Step 2: Training the Model (with QA)](#step-2-training-the-model-with-qa)
- [Step 3: Model Evaluation (with QA)](#step-3-model-evaluation-with-qa)
- [Step 4: Interactive Testing](#step-4-interactive-testing)
- [Step 5: Deployment and Monitoring](#step-5-deployment-and-monitoring)
- [Hardware Optimization Guide](#hardware-optimization-guide)
- [Testing Framework](#testing-framework)
- [Customization Options](#customization-options)
- [Advanced Usage](#advanced-usage)

## Project Overview

This project fine-tunes a small language model (TinyLlama) to generate cooking recipes from ingredient lists, with an emphasis on quality assurance practices at each step.

| Component | Description | QA Relevance |
|-----------|-------------|--------------|
| **Model** | TinyLlama (1.1B parameters) with LoRA adapters | Smaller model for faster testing iterations |
| **Training Method** | Instruction fine-tuning on RecipeNLG dataset | Controlled training for result validation |
| **Memory Optimization** | Parameter-efficient methods for limited hardware | Ensures accessibility for testing |
| **Testing Approach** | Functional + Performance + Quality Assessment | Comprehensive validation strategy |

## The ML Development and QA Lifecycle

Unlike traditional software where tests can definitively pass or fail, ML systems require different QA approaches:

1. **Data Preparation QA**
   - Data validation (format, completeness)
   - Distribution analysis
   - Bias detection

2. **Training Process QA**
   - Loss curve monitoring
   - Convergence verification
   - Regularization validation

3. **Model Evaluation QA**
   - Performance metrics
   - Generalization testing
   - Boundary case analysis

4. **Deployment QA**
   - Memory usage validation
   - Latency testing
   - Production monitoring

This guide will highlight QA practices at each stage of the process.

## Setup and Installation

### Prerequisites

- **GPU with 8GB+ VRAM** recommended (16GB+ ideal, but accommodations for smaller GPUs provided)
- **Python 3.12.3+** installed
- **Ollama** (optional, for deployment) - [Install Ollama](https://ollama.ai/download)

If you haven't already, run the setup script:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

## Step 1: Dataset Preparation (with QA)

The recipe generation model uses the RecipeNLG dataset, which must be manually downloaded:

```bash
# Create directory for the dataset
mkdir -p ~/recipe_manual_data

# Download dataset from https://recipenlg.cs.put.poznan.pl/dataset
# After downloading, place full_dataset.csv in your recipe_manual_data directory
```

### Prepare Dataset with QA Checks

```bash
# Process dataset with validation
python -m src.data.recipe_prepare_dataset --data_dir ~/recipe_manual_data

# Or using Makefile
make recipe-data DATA_DIR=~/recipe_manual_data
```

**QA Integration**: The dataset preparation:
- Validates data format and structure
- Removes malformed entries
- Creates test examples at `data/processed/recipe_test_examples.json`
- Reports statistics on processed data

This critical QA step ensures you begin with high-quality training data, preventing the "garbage in, garbage out" problem common in ML.

## Step 2: Training the Model (with QA)

### Hardware Analysis (Pre-Training QA)

Before training, analyze your hardware to select the right configuration:

```bash
# Check hardware and get recommendations
python scripts/config_optimizer.py
```

### Training Options

Select the appropriate training configuration based on your hardware:

#### A. Sanity Test (Recommended First Step)

This quick test validates your entire ML pipeline before committing to longer training:

```bash
# Run minimal training for test validation
make recipe-train-test DATA_DIR=~/recipe_manual_data
```

```bash
# Run more agressive training for test validation
make test-recipe-train-geforce-rtx-3060 DATA_DIR=~/recipe_manual_data
```

**QA Value**: This step:
- Validates the entire pipeline quickly
- Confirms hardware compatibility
- Identifies potential issues early
- Creates a baseline for comparison

#### B. Full Training Options

Choose based on your GPU's VRAM:

| Configuration | VRAM Requirement | Command |
|---------------|-----------------|---------|
| Standard | 16GB+ | `make recipe-train DATA_DIR=~/recipe_manual_data` |
| Optimized | 10GB+ | `make recipe-train-optimized DATA_DIR=~/recipe_manual_data` |
| Medium-Memory | 8GB+ | `make recipe-train-medium-memory DATA_DIR=~/recipe_manual_data` |
| Low-Memory | 6GB+ | `make recipe-train-low-memory DATA_DIR=~/recipe_manual_data` |

```bash
# I've optimized this for my GeForce RTX 3060 setup, so worth glossing over for ideas
make recipe-train-geforce-rtx-3060 DATA_DIR=~/recipe_manual_data
```

**QA During Training**: The training process includes:
- Progress tracking with WandB
- Regular evaluation on validation data
- Checkpoint saving for model comparison
- Early stopping to prevent overfitting

## Step 3: Model Evaluation (with QA)

After training, thoroughly evaluate your model:

```bash
# Run minimal evaluation suite for test validations
python -m src.model.recipe_evaluate --config config/text_generation_sanity_test.yaml

# Run more agressive evaluation suite for test validations
python -m src.model.recipe_evaluate --config config/test_text_generation_3060_geforce_rtx.yaml

# Run comprehensive evaluation suite for production model
python -m src.model.recipe_evaluate --config config/text_generation.yaml

# Or with Makefile
make recipe-evaluate
```

**QA Metrics Collected**:
- **Generation quality**: Text coherence and recipe completeness
- **Ingredient adherence**: Whether all ingredients are used correctly
- **Format compliance**: Proper recipe structure
- **Performance**: Inference speed and memory usage

This provides objective measurements of model quality beyond simple "it works."

## Step 4: Interactive Testing

Interactive testing allows human-in-the-loop evaluation, crucial for subjective quality assessment:

### CLI Testing Interface

```bash
# Test with command-line interface
python -m src.recipe_demo --model_dir="models/recipe_assistant"

# Run in interactive mode
python -m src.recipe_demo --model_dir="models/recipe_assistant" --interactive
```

### Important: Prompt Format Requirements

When interacting with the recipe model, using the **exact prompt format** is critical for proper functioning. The model was trained with specific special tokens and expects this format during inference:

```
<|system|>You are a recipe assistant. Create detailed recipes with exact measurements and clear instructions.<|endoftext|>
<|user|>Write a complete recipe using these ingredients: {ingredients}. Include a title, ingredients list with measurements, and numbered cooking steps.<|endoftext|>
<|assistant|>
```

#### Why This Format Matters

- **Special Tokens**: The `<|system|>`, `<|user|>`, and `<|assistant|>` markers are recognized by the model
- **Format Consistency**: The model generates recipes based on patterns it learned during training
- **Content Structure**: The prompt specifies exactly what the output should contain (title, ingredients with measurements, numbered steps)

#### Common Issues When Using Incorrect Formats

- Incomplete or malformed recipes
- Model repeating the prompt instead of generating content
- Missing sections (title, ingredients, or instructions)
- Poor quality scores on evaluation metrics

If you're creating custom tools or integrations with this model, always maintain this exact prompt structure.

For more detailed information about prompt formatting, see the [Prompt Format Guide](docs/PROMPT_FORMAT.md).

### Web UI Testing Interface

```bash
# Launch browser-based testing interface
python -m src.recipe_web_demo --model_dir="models/recipe_assistant"
```

**QA Approach**: Interactive testing enables:
- Subjective quality assessment
- Exploration of edge cases
- Parameter tuning (temperature, sampling)
- Identification of failure modes

## Step 5: Deployment and Monitoring

Deploy your model to Ollama for real-world testing:

```bash
# Export model to Ollama
python -m src.model.recipe_export_to_ollama --model_dir="models/recipe_assistant" --model_name="recipe-gen"

# Or with Makefile
make recipe-export
```

Run your model with:
```bash
ollama run recipe-gen "Create a recipe with chicken, rice, and bell peppers"
```

**QA for Deployment**:
- Verify model size and memory requirements
- Measure cold-start and response latency
- Compare deployed outputs with development results
- Test integration with other systems

## Hardware Optimization Guide

### Memory Management Tools

Use these tools for hardware-specific optimizations:

```bash
# Display memory statistics and recommendations
python scripts/optimize_gpu_memory.py

# Run memory test to find maximum allocation
python scripts/optimize_gpu_memory.py --test

# Monitor memory during other operations
python scripts/optimize_gpu_memory.py --monitor
```

### Memory Monitoring During Training

You can add memory monitoring to any process:

```python
from src.utils.memory_monitor import GPUMemoryMonitor

# Create and start the monitor
memory_monitor = GPUMemoryMonitor(interval_seconds=30)
memory_monitor.start()

# Run your operations here

# Stop monitoring when done
memory_monitor.stop()
```

**QA Value**: Memory monitoring helps:
- Identify memory leaks
- Optimize batch sizes
- Prevent OOM (out of memory) errors
- Ensure hardware compatibility

## Testing Framework

The project includes a dedicated testing framework for model validation:

```bash
# Run all tests
pytest

# Run recipe model tests specifically
pytest tests/test_recipe_model.py -v

# Generate code coverage report
pytest --cov=src --cov-report=html
```

**Test Types Implemented**:

1. **Functional Testing**: Verifies model generates recipes from ingredients
2. **Format Testing**: Ensures output adheres to recipe structure
3. **Ingredient Testing**: Confirms all ingredients are used
4. **Performance Testing**: Measures generation speed and memory usage

The testing framework provides objective, repeatable validation that complements subjective human evaluation.

## Customization Options

### Configuration Files

The project includes several configuration files for different scenarios:

1. **Standard Configuration** (`config/text_generation.yaml`)
   - Default settings for 16GB+ VRAM
   - Full sequence length (512 tokens)

2. **Optimized Configuration** (`config/text_generation_optimized.yaml`)
   - Balanced settings for 10GB+ VRAM
   - Limited dataset size (100,000 samples)

3. **Low-Memory Configuration** (`config/text_generation_low_memory.yaml`)
   - For GPUs with limited VRAM (6-8GB)
   - CPU offloading and aggressive optimizations

### Customization Parameters

Key parameters you can modify:

#### Data Settings
- `max_train_samples`: Limit training examples
- `max_length`: Maximum sequence length 

#### Model Settings
- `lora_r`: Rank of LoRA adapters (higher = more parameters)
- `lora_alpha`: Scale of LoRA updates
- `target_modules`: Which model layers to fine-tune

#### Training Settings
- `learning_rate`: Speed of model updates
- `batch_size` and `gradient_accumulation_steps`: Memory vs. speed tradeoff
- `num_train_epochs`: Training duration

#### Generation Settings
- `temperature`: Randomness in generation (higher = more creative)
- `top_p` and `top_k`: Control sampling strategy
- `max_new_tokens`: Maximum response length

## Advanced Usage

### Merging LoRA Adapter with Base Model

For better performance or offline use:

```bash
python -m src.model.recipe_merge_and_export_lora \
  --base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --lora_model="models/recipe_assistant" \
  --output_dir="models/recipe_merged"
```

### Performance Optimization Tips

If generation seems slow:

1. Reduce context length in configuration
2. Use a merged model instead of adapter
3. Enable quantization (8-bit) during inference
4. Deploy to Ollama with optimized parameters

## Troubleshooting

### Common Issues

**"CUDA out of memory" during training**
- Try a lower-memory configuration
- Reduce batch size or sequence length
- Use gradient checkpointing and CPU offloading

**Slow generation time**
- Merge the LoRA adapter with base model
- Use a smaller context length
- Try quantized inference

**Poor recipe quality**
- Increase training data size
- Adjust prompt format in configuration
- Fine-tune generation parameters (temperature, top_p)

## 📚 Additional Documentation

For more detailed information, check out these documentation files:

- [Prompt Format Guide](docs/PROMPT_FORMAT.md) - Detailed guidance on prompt formatting
- [Test Examples Guide](docs/TEST_EXAMPLES.md) - Information about test examples
- [Testing Guide](docs/TESTING_GUIDE.md) - Best practices for testing the model
- [Documentation Index](docs/README.md) - Overview of all available documentation
