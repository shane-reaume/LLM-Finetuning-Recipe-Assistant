# Getting Started with LLM-Finetuning-Playground - Sentiment Analysis

**Sentiment Analysis** - Using an encoder model (DistilBERT) for classification

## 📋 Common Prerequisites

- **Python 3.12.3 or later**
- **Git** for version control
- **GPU with 8GB+ VRAM** recommended for training (CPU can be used but will be very slow)
- **Basic Python knowledge** (No ML experience required)

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

## Sentiment Analysis Project

This project is a sentiment analysis model using movie reviews. Here's how to use it:

### Step 1: Create Test Examples

Generate a set of test examples that will be used for consistent evaluation:

```bash
python -m src.data.sentiment_create_test_set
```

This creates a JSON file with test examples at `data/processed/sentiment_test_examples.json`.

### Step 2: Train the Model

Start the training process:

```bash
python -m src.model.sentiment_train
```

This will:

- Download the IMDB dataset
- Load the DistilBERT model
- Fine-tune it for sentiment analysis
- Save the model to `models/sentiment`

Training should take approximately 1-2 hours on an 8GB GPU.

### Step 3: Run Tests

Evaluate the model's performance:

```bash
pytest tests/test_sentiment_model.py -v
```

This will run a suite of tests to check that:

- The model can be loaded correctly
- Predictions work as expected
- The model meets performance requirements

### Step 4: Try the Interactive Demo

Once your model is trained, you can use the demo script to try it out:

```bash
# Run with pre-defined examples
python -m src.sentiment_demo

# Run in interactive mode to type your own text
python -m src.sentiment_demo --interactive
```

The demo provides:

- Color-coded sentiment predictions (green for positive, red for negative)
- Confidence scores for each prediction
- Performance metrics for batch processing

## Project Structure for Sentiment Analysis

- `config/sentiment_analysis.yaml`: Configuration file
- `src/model/sentiment_train.py`: Training script
- `src/model/sentiment_inference.py`: Inference code
- `src/model/sentiment_evaluate.py`: Evaluation code  
- `src/sentiment_demo.py`: Interactive demo
- `tests/test_sentiment_model.py`: Test suite

## Customization

You can modify the `config/sentiment_analysis.yaml` file to change:

- Model parameters
- Training settings
- Performance thresholds
- Dataset options

## Next Steps

Once you're comfortable with the sentiment analysis project, consider:

1. **Adding your own test cases** - Create challenging examples to test model robustness
2. **Experimenting with different models** - Try other small models like Phi-2 or Gemma 2B
3. **Implementing advanced testing** - Test for bias or concept drift
4. **Extending to other tasks** - Try our recipe generation demo in GETTING_STARTED_DEMO_2.md
