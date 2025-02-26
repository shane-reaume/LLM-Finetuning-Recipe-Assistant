# Getting Started with LLM-Finetuning-Playground

This guide will help you set up and start using the LLM-Finetuning-Playground project for sentiment analysis.

## Prerequisites

- Python 3.12.3 or later
- Git
- 8GB+ GPU for training (CPU can be used but will be much slower)

## Setup

1. **Set up the environment**

   ```bash
   # Make the setup script executable
   chmod +x setup_env.sh
   
   # Run the setup script
   ./setup_env.sh
   ```

   This will:
   - Create a virtual environment in the `venv` directory
   - Install all dependencies
   - Create necessary project directories

2. **Activate the virtual environment** (if not already activated by the script)

   ```bash
   source venv/bin/activate
   ```

## Initial Project: Sentiment Analysis

The initial project is a sentiment analysis model using movie reviews. Here's how to use it:

### Step 1: Create Test Examples

Generate a set of test examples that will be used for consistent evaluation:

```bash
python -m src.data.create_test_set
```

This creates a JSON file with test examples at `data/test_examples.json`.

### Step 2: Train the Model

Start the training process:

```bash
python -m src.model.train
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
python -m src.demo

# Run in interactive mode to type your own text
python -m src.demo --interactive
```

The demo provides:
- Color-coded sentiment predictions (green for positive, red for negative)
- Confidence scores for each prediction
- Performance metrics for batch processing

## Project Structure

- `config/`: Configuration files
- `data/`: Dataset storage
- `src/`: Source code
  - `data/`: Data processing
  - `model/`: Model training and inference
  - `utils/`: Utility functions
- `tests/`: Test scripts

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
4. **Extending to other tasks** - Try text generation or summarization tasks 