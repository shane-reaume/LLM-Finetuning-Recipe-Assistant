# LLM-Finetuning-Playground

A beginner-friendly project for fine-tuning, testing, and deploying language models with a strong emphasis on quality assurance and testing methodologies.

## 🎯 What This Project Does

This project demonstrates how to:

- Fine-tune language models for different types of tasks
- Implement proper testing and evaluation methodologies 
- Create evaluation metrics and test sets for consistent model testing
- Deploy models to different platforms (Hugging Face Hub, Ollama)
- Create interactive demos for testing models

## 🤖 Demo Projects

### 1. Sentiment Analysis (Classification)

![Sentiment Analysis Demo](data/img/sentiment-analysis-demo.png)

- Fine-tunes a **DistilBERT** model to classify movie reviews as positive or negative
- Emphasizes classification metrics (accuracy, F1 score, precision, recall)
- Deployed to **Hugging Face Hub**

### 2. Recipe Assistant (Text Generation)

- Fine-tunes a **TinyLlama** model to generate recipes from ingredient lists
- Focuses on generation quality and coherence
- Deployed to **Ollama**

## 🧪 Testing & Quality Assurance Focus

This project places special emphasis on testing methodologies for ML models. For a comprehensive guide to our testing approach, see [GETTING_STARTED_DEMO_1.md#testing-philosophy-and-methodology](GETTING_STARTED_DEMO_1.md#testing-philosophy-and-methodology).

### Test Types Implemented

- **Unit tests**: Testing individual components like data loaders
- **Functional tests**: Testing model predictions with known inputs
- **Performance tests**: Ensuring the model meets accuracy and speed requirements
- **Balanced test sets**: Creating test data with equal class distribution
- **High-confidence evaluations**: Analyzing model confidence in predictions

### Testing Principles

- **Reproducibility**: Tests use fixed test sets to ensure consistent evaluation
- **Isolation**: Components are tested independently
- **Metrics tracking**: F1 score, precision, recall, and accuracy are tracked
- **Performance benchmarking**: Measuring inference speed

## 🚀 Getting Started

See the [GETTING_STARTED.md](GETTING_STARTED.md) file for detailed instructions on:

- Setting up your environment
- Training your first model
- Running the test suite
- Using the interactive demo

## 📊 Example Results

After training, you'll be able to analyze sentiment in text:

# Project WIP Notes
here we will keep notes on the project to assist with the direction of the project for myself and for any AI agent assistance so they have a high level of the project without crawling the file directory.

## Project Structure

```yaml
LLM-Finetuning-Playground/
├── config/
│   ├── default.yaml                  # Base configuration file
│   ├── sentiment_analysis.yaml       # Sentiment analysis training configuration
│   ├── text_generation.yaml          # Recipe generation base configuration
│   ├── text_generation_small.yaml    # Recipe generation with smaller model settings
│   ├── text_generation_improved.yaml # Improved recipe generation configuration
│   └── recipe_prompt.txt             # Text prompt template for recipe generation
├── data/
│   ├── raw/                          # Original datasets for both projects
│   ├── processed/                    # Preprocessed data files
│   └── img/                          # Images for documentation
├── docs/                             # Documentation files
├── logs/                             # Training and evaluation logs
├── models/                           # Saved model checkpoints
│   ├── sentiment/                    # Sentiment analysis model
│   └── recipe/                       # Recipe generation model
├── notebooks/                        # Jupyter notebooks for exploration
├── scripts/                          # Utility scripts for automation
├── src/
│   ├── data/                         # Data loading and preprocessing code
│   │   ├── dataset.py                # Base dataset functions
│   │   ├── sentiment_create_test_set.py       # Test set creation for sentiment analysis
│   │   ├── sentiment_create_balanced_test_set.py  # Balanced test set for sentiment
│   │   ├── prepare_gen_dataset.py    # Data preparation for text generation
│   │   └── recipe_prepare_dataset.py # Recipe dataset preparation
│   ├── model/                        # Model-related code
│   │   ├── generation/               # Additional generation model components
│   │   ├── sentiment_model_loader.py # Code to load sentiment models and tokenizers
│   │   ├── sentiment_train.py        # Sentiment model training
│   │   ├── sentiment_inference.py    # Sentiment model inference
│   │   ├── sentiment_evaluate.py     # Sentiment model evaluation
│   │   ├── sentiment_publish.py      # Publish sentiment model to HF Hub
│   │   ├── recipe_train.py           # Recipe model training
│   │   ├── recipe_evaluate.py        # Recipe model evaluation
│   │   ├── recipe_export_to_ollama.py # Export recipe model to Ollama
│   │   ├── recipe_merge_and_export_lora.py  # Merge LoRA adapters and export
│   │   └── update_model_card.py      # Generate/update model cards (general utility)
│   ├── utils/                        # Utility functions
│   │   ├── config_utils.py           # Configuration utilities
│   │   ├── recipe_formatter.py       # Format recipe outputs
│   │   ├── recipe_generator.py       # Recipe generation utilities
│   │   └── recipe_prompts.py         # Recipe prompt templates
│   ├── sentiment_demo.py             # Sentiment analysis demo
│   ├── recipe_demo.py                # Recipe generation CLI demo
│   ├── recipe_web_demo.py            # Recipe generation web UI demo
│   ├── direct_recipe_test.py         # Direct testing of recipe generation
│   └── recipe_export_to_ollama_utils.py # Consolidated Ollama export utilities
├── tests/                            # Automated tests
│   ├── conftest.py                   # Pytest configurations
│   ├── test_sentiment_model.py       # Tests for sentiment analysis
│   └── test_recipe_model.py          # Tests for recipe generation
├── wandb/                            # Weights & Biases logging data
├── GETTING_STARTED.md                # Getting started guide
├── GETTING_STARTED_DEMO_1.md         # Demo 1 guide with testing documentation
├── GETTING_STARTED_DEMO_2.md         # Demo 2 guide
├── AI_TESTING_IDEAS.md               # Ideas for AI model testing
├── DATASET_INSTRUCTIONS.md           # Dataset preparation instructions
├── model_card.md                     # Model card template
├── requirements.txt                  # Project dependencies
├── setup_env.sh                      # Environment setup script
├── Makefile                          # Automation of common tasks
└── LICENSE                           # Project license
```

## Explanation:

This project is organized into two main applications:

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

## Integrations
Here are some integrations and tools that can be used for free or on low-cost/free tiers:

a. Model & Data Libraries
Hugging Face Transformers:
Use this library to load pretrained models and tokenizers (e.g., AutoModelForCausalLM and AutoTokenizer). It also provides the Trainer API which simplifies training loops.

Hugging Face Datasets:
Efficiently load, preprocess, and manage datasets.

b. Testing and QA
Pytest:
A robust testing framework for Python. Write unit tests for data processing, model loading, and training routines.

Coverage Tools:
Use tools like pytest-cov to track test coverage.

Mocking Libraries:
For parts of your code that require heavy resources (like the full fine-tuning loop), you can use mocking (with libraries like unittest.mock) to simulate behavior on a small scale.

c. Configuration and Experiment Management
Configuration Management:
Use YAML/JSON configuration files. Libraries like PyYAML can help you read configurations into your Python scripts.

Logging and Monitoring:

TensorBoard: Integrated with PyTorch or TensorFlow (if you use them indirectly via Hugging Face) to track metrics.
Weights & Biases (wandb): The free tier is often sufficient for logging experiments and tracking hyperparameters.
d. Version Control and CI/CD
Git:
Use Git for version control. GitHub or GitLab offer free private repositories.

GitHub Actions:
Set up a simple CI pipeline to run your tests automatically on every commit or pull request.

e. Compute Resources
Local CPU/Low-end GPU:
Fine-tuning a 14B parameter model is resource intensive, so for testing and learning:

Use a smaller model version or a subset of your data for rapid iteration.
Leverage gradient accumulation or mixed precision training to simulate training on limited hardware.
Cloud Platforms:
Consider free tiers from Google Colab, Kaggle Kernels, or free trial credits from cloud providers. These can be useful for initial experiments without incurring high costs.

Docker (Optional):
Containerize your environment to ensure reproducibility and ease deployment, though this is optional if your budget is very low.

## 🔄 Version Control & Collaboration

### Getting Started with this Repository
