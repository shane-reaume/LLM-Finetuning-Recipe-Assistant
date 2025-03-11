# LLM-Finetuning-Recipe-Assistant

A beginner-friendly project for fine-tuning, testing, and deploying language models that emphasizes quality assurance methodologies for ML/AI systems.

## 🎯 Project Overview

This project demonstrates a complete ML lifecycle for recipe generation with QA practices integrated at every stage:

- **Model Development**: Fine-tune TinyLlama (1.1B parameters) to generate cooking recipes from ingredient lists
- **QA Integration**: Implement testing strategies throughout the ML development pipeline
- **Memory Optimization**: Configure training for different hardware capabilities (6GB-16GB+ VRAM)
- **Deployment & Validation**: Export trained models to Ollama for practical usage testing

Perfect for QA engineers looking to understand how quality assurance fits into ML projects.

### 📚 Documentation

- [Getting Started Guide](GETTING_STARTED.md) - Detailed setup and usage instructions
- [Prompt Format Guide](docs/PROMPT_FORMAT.md) - How to correctly format prompts
- [Test Examples Guide](docs/TEST_EXAMPLES.md) - Understanding the test examples
- [Testing Guide](docs/TESTING_GUIDE.md) - Best practices for model testing
- [Documentation Index](docs/README.md) - Overview of all documentation

## 🔍 Why This Project?

Traditional software QA focuses on deterministic systems with clear right/wrong outcomes. Machine learning brings unique challenges:

- **ML systems are probabilistic** - outputs vary even with identical inputs
- **Quality is subjective** - "correctness" often depends on human judgment
- **Failure modes are complex** - issues might stem from data, model architecture, or training
- **Testing requires specialized metrics** - beyond simple pass/fail

This project bridges the gap between traditional QA and ML testing by demonstrating practical approaches to each challenge.

## 📋 System Requirements

- **Python 3.12.3 or higher**
- **Git** for version control
- **GPU with VRAM**:
  - 8GB+ required for recipe generation (12GB+ preferred)
  - CPU-only training possible but very slow
- **Basic Python knowledge** (No ML experience required)
- **Platform Compatibility**:
  - ✅ **Windows 11**: Using WSL with Ubuntu
  - ✅ **Linux**: Confirmed on Debian-based distributions
  - ❌ **macOS**: Not currently compatible due to PyTorch version requirements

## 🔧 Quick Start

```bash
git clone https://github.com/your-repo/LLM-Finetuning-Recipe-Assistant.git
cd LLM-Finetuning-Recipe-Assistant
chmod +x setup_env.sh
./setup_env.sh
```

Then follow the [detailed guide](GETTING_STARTED.md) for complete instructions.

## 🧪 QA & Testing Methodology

This project demonstrates how ML testing differs from traditional software testing:

### Testing Throughout the ML Lifecycle

| Stage | Traditional QA | ML QA | Implementation |
|-------|----------------|-------|----------------|
| **Data Preparation** | Input validation | Data quality checks, bias detection | Data preprocessing scripts with validation |
| **Training** | N/A | Convergence monitoring, loss analysis | Automated metric logging, early stopping |
| **Model Validation** | Functional testing | Accuracy metrics, robustness testing | Test set evaluation, performance benchmarks |
| **Deployment** | Integration testing | Latency testing, memory usage | Ollama export with performance metrics |

### Key Testing Types Implemented

- **Functional Tests**: Verify the model generates recipes from ingredients
- **Quality Assessment**: Measure coherence and ingredient adherence  
- **Memory Tests**: Ensure models work on consumer hardware
- **Performance Tests**: Validate inference speed and throughput
- **Format Tests**: Verify recipe structure follows expected pattern

### Testing Principles

- **Reproducibility**: Fixed test sets and seeds ensure consistent evaluation
- **Isolation**: Components are tested independently
- **Observability**: Metrics are tracked and logged for analysis
- **Benchmarking**: Performance characteristics are measured systematically

## 📊 Example Results

After training, you can generate recipes from ingredient lists:

```bash
# Using Ollama after export
ollama run recipe-assistant "Create a recipe with these ingredients: chicken, rice, bell peppers, onions"
```

For detailed information on training, evaluation, and deployment, see the [Getting Started Guide](GETTING_STARTED.md).

## 📋 Prompt Format Requirements

The recipe model requires a specific prompt format to generate high-quality recipes:

```
<|system|>You are a recipe assistant. Create detailed recipes with exact measurements and clear instructions.<|endoftext|>
<|user|>Write a complete recipe using these ingredients: {ingredients}. Include a title, ingredients list with measurements, and numbered cooking steps.<|endoftext|>
<|assistant|>
```

This specific format with special tokens (`<|system|>`, `<|user|>`, `<|assistant|>`) is critical for proper functioning, as the model was trained with this exact pattern. Incorrect prompt formats will result in poor quality outputs.

See the following documentation for more details:
- [Getting Started Guide](GETTING_STARTED.md) - Basic usage instructions
- [Prompt Format Guide](docs/PROMPT_FORMAT.md) - Detailed prompt formatting guidance
- [Test Examples Guide](docs/TEST_EXAMPLES.md) - Information about test examples
- [Testing Guide](docs/TESTING_GUIDE.md) - Best practices for testing the model

## 🚀 Project Organization

```
LLM-Finetuning-Recipe-Assistant/
├── config/                  # Training configurations for different hardware
├── data/                    # Data processing and storage
├── models/                  # Saved model checkpoints
├── src/                     # Source code
│   ├── data/                # Dataset preparation and processing
│   ├── model/               # Model training and evaluation
│   └── utils/               # Utility functions
├── tests/                   # Test suite for model verification
└── scripts/                 # Helper scripts for optimization
```

## 🔧 Key Components

- **Training Pipeline**: `src/model/recipe_train.py` - Trains the recipe generation model
- **Evaluation Tools**: `src/model/recipe_evaluate.py` - Tests model quality
- **Demo Interfaces**: 
  - CLI: `src/recipe_demo.py` - Command-line testing
  - Web UI: `src/recipe_web_demo.py` - Browser-based interface
- **Test Suite**: `tests/test_recipe_model.py` - Automated testing

## 🔌 Integrations

- **Hugging Face**: Model architecture and training utilities
- **PEFT**: Parameter-Efficient Fine-Tuning with LoRA
- **Pytest**: Automated testing framework
- **Weights & Biases**: Experiment tracking
- **Ollama**: Local model deployment

## 📚 Next Steps

- See the [Getting Started Guide](GETTING_STARTED.md) for detailed instructions
- Explore the test suite to understand ML testing approaches
- Experiment with different configurations and model parameters
- Try creating your own test cases for the model