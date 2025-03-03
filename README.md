# LLM-Finetuning-Playground

A beginner-friendly project for fine-tuning, testing, and deploying language models with a strong emphasis on quality assurance and testing methodologies.

## 🎯 What This Project Does

This project demonstrates how to:

- Fine-tune language models for different types of tasks (classification and generation)
- Implement proper testing and evaluation methodologies for LLMs
- Create evaluation metrics and test sets for consistent model testing
- Deploy models to different platforms (Hugging Face Hub, Ollama)
- Create interactive demos for testing models

## 📋 System Requirements

- **Python 3.12.3 or higher**
- **Git** for version control
- **GPU with VRAM**:
  - 8GB+ recommended for sentiment analysis
  - 8GB+ required for recipe generation (16GB+ preferred)
  - CPU-only training is possible but very slow
- **Basic Python knowledge** (No ML experience required)
- **Platform Compatibility**:
  - ✅ **Windows 11**: Using WSL with Ubuntu
  - ✅ **Linux**: Confirmed on Debian-based distributions like Linux Mint and Ubuntu
  - ❌ **macOS**: Not currently compatible due to PyTorch version requirements
    - Project uses PyTorch 2.6.0, while macOS is limited to PyTorch 2.2.0
    - May work with modifications to `requirements.txt` but not officially supported

## 🔧 Initial Setup

```bash
git clone https://github.com/your-repo/LLM-Finetuning-Playground.git
cd LLM-Finetuning-Playground
chmod +x setup_env.sh
./setup_env.sh
```

This script will:

- Create a virtual environment
- Install all dependencies
- Create necessary project directories

## 🤖 Demo Projects

### 1. Sentiment Analysis (Classification)

![Sentiment Analysis Demo](data/img/sentiment-analysis-demo.png)

- **Model Architecture**: Fine-tuning a **DistilBERT** encoder model for binary text classification
- **Training Methodology**: Binary classification on the IMDB movie reviews dataset
- **Key Techniques**: Transfer learning, mixed-precision training, supervised fine-tuning
- **Evaluation Metrics**: Accuracy, F1 score, precision, recall
- **Deployment Target**: Published to **Hugging Face Hub**

→ [**Get Started with Sentiment Analysis Demo**](GETTING_STARTED_DEMO_1.md)

### 2. Recipe Assistant (Text Generation)

- **Model Architecture**: Fine-tuning a **TinyLlama** (1.1B parameters) with LoRA adapters
- **Training Methodology**: Instruction fine-tuning on the RecipeNLG dataset
- **Key Techniques**: Parameter-efficient training, gradient checkpointing, memory optimizations
- **Memory-Optimized Options**: Choose from standard, medium, or low-memory training configurations
- **Deployment Target**: Exported to **Ollama** for local inference

→ [**Get Started with Recipe Generation Demo**](GETTING_STARTED_DEMO_2.md)

## 🧪 Testing & Quality Assurance Focus

This project places special emphasis on testing methodologies for ML models. For a comprehensive guide to our testing approach, see [GETTING_STARTED_DEMO_1.md#testing-philosophy-and-methodology](GETTING_STARTED_DEMO_1.md#testing-philosophy-and-methodology).

### Test Types Implemented

- **Unit tests**: Testing individual components like data loaders
- **Functional tests**: Testing model predictions with known inputs
- **Performance tests**: Ensuring the model meets accuracy and speed requirements
- **Balanced test sets**: Creating test data with equal class distribution
- **High-confidence evaluations**: Analyzing model confidence in predictions
- **Memory tests**: Ensuring models can run on consumer hardware

### Testing Principles

- **Reproducibility**: Tests use fixed test sets to ensure consistent evaluation
- **Isolation**: Components are tested independently
- **Metrics tracking**: F1 score, precision, recall, and accuracy are tracked
- **Performance benchmarking**: Measuring inference speed and memory usage

## 📊 Example Results

### Sentiment Analysis

After training the sentiment analysis model, you'll be able to classify text sentiment:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="your-username/imdb-sentiment-analysis")
result = classifier("This movie was absolutely amazing, I loved every minute of it!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

### Recipe Generation

With the recipe generation model, you can create recipes from ingredients:

```bash
# Using Ollama after export
ollama run recipe-assistant "Create a recipe with these ingredients: chicken, rice, bell peppers, onions"
```

## Next Steps

1. Follow the [Sentiment Analysis Demo Guide](GETTING_STARTED_DEMO_1.md)
2. Try the [Recipe Generation Demo Guide](GETTING_STARTED_DEMO_2.md)
3. Experiment with your own datasets and models
4. Contribute to the project by adding new test types or model architectures

---

## Project Developer Notes

### Project Structure

The project is organized into two main applications:

1. **Sentiment Analysis (DistilBERT)**: A classification task that analyzes movie reviews
   - Training: `src/model/sentiment_train.py`
   - Inference: `src/model/sentiment_inference.py`  
   - Demo: `src/sentiment_demo.py`
   - Tests: `tests/test_sentiment_model.py`

2. **Recipe Generation (TinyLlama)**: A text generation task that creates recipes
   - Training: `src/model/recipe_train.py`
   - Evaluation: `src/model/recipe_evaluate.py`
   - Demos: 
     - CLI: `src/recipe_demo.py`
     - Web UI: `src/recipe_web_demo.py`
   - Tests: `tests/test_recipe_model.py`
   - Deployment: Various export utilities for Ollama

The project uses:

- **YAML configurations** in the `config/` directory for model parameters
- **Weights & Biases** for experiment tracking
- **Pytest** for automated testing
- **Hugging Face** and **Ollama** for model deployment

## File Structure

```yaml
LLM-Finetuning-Playground/
├── config/                           # Configuration files
│   ├── sentiment_analysis.yaml       # Sentiment analysis training configuration
│   ├── text_generation.yaml          # Recipe generation base configuration
│   ├── text_generation_medium_memory.yaml # Medium memory configuration
│   └── text_generation_low_memory.yaml # Low memory configuration
├── data/                             # Data directories
├── models/                           # Saved model checkpoints
├── src/                              # Source code
│   ├── data/                         # Data processing
│   ├── model/                        # Model training and inference
│   └── utils/                        # Utility functions
├── tests/                            # Automated tests
├── GETTING_STARTED_DEMO_1.md         # Sentiment analysis guide
├── GETTING_STARTED_DEMO_2.md         # Recipe generation guide
├── requirements.txt                  # Project dependencies
├── setup_env.sh                      # Environment setup script
└── Makefile                          # Automation of common tasks
```

## Integrations

- **Hugging Face Transformers & Datasets**: For models, tokenizers, and data loading
- **PEFT**: Parameter-Efficient Fine-Tuning with LoRA
- **Pytest**: For unit and integration testing
- **Weights & Biases**: For experiment tracking
- **Ollama**: For local deployment of recipe generation models