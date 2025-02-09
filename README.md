# LLM-Finetuning-Playground
Project for LLM transformer finetuning, model deployment and to establish a robust testing workflow.

# Project WIP Notes
here we will keep notes on the project to assist with the direction of the project for myself and for any AI agent assistance so they have a high level of the project without crawling the file directory.

## Project Structure

my_finetune_project/
├── config/
│   └── default.yaml            # Configuration file (training parameters, paths, etc.)
├── data/
│   ├── raw/                    # Original/raw datasets
│   └── processed/              # Preprocessed/tokenized data files
├── notebooks/                  # Jupyter notebooks for exploration and quick experiments
├── src/
│   ├── __init__.py
│   ├── data/                   # Data loading and preprocessing code
│   │   ├── __init__.py
│   │   └── dataset.py          # Functions/classes to load and process your dataset
│   ├── model/                  # Model-related code
│   │   ├── __init__.py
│   │   ├── model_loader.py     # Code to load pretrained models and tokenizers
│   │   ├── train.py            # Training script (fine-tuning loop, Trainer setup, etc.)
│   │   └── inference.py        # Inference code to load and run your fine-tuned model
│   └── utils/                  # Utility functions (logging, configuration parsing, etc.)
│       ├── __init__.py
│       └── config_utils.py     # Functions to read configuration files (e.g., YAML parser)
├── tests/                      # Automated tests for QA
│   ├── __init__.py
│   ├── test_dataset.py         # Unit tests for data processing functions
│   ├── test_model_loading.py   # Tests for loading models/tokenizers correctly
│   └── test_training.py        # Tests for parts of the training loop (e.g., with dummy data)
├── requirements.txt            # List of required Python packages
├── setup.py                    # (Optional) Packaging script if you plan to distribute your code
└── README.md                   # Project overview, installation instructions, etc.

## Explanation:

`config/`
Store configuration files (using YAML, JSON, or INI formats) so that you can change hyperparameters, file paths, and other settings without modifying the code. Tools like Hydra or OmegaConf can help manage configurations.

`data/`
Keep your raw data separate from processed data. This makes it easier to rerun preprocessing and avoid accidental data corruption.

`src/`
Divide the core functionality:

- `data/`: Handle data loading, preprocessing, and any dataset manipulation.
- `model/`: Code for loading the Hugging Face model, setting up the training loop (using Trainer), and handling inference.
- `utils/`: General helper functions (e.g., reading config files, logging utilities).
- `tests/`
Write tests (using frameworks like pytest or Python’s built-in unittest) for each module. For instance, test that your data preprocessing functions work correctly with sample inputs, or that your model can load a checkpoint properly.
- `notebooks/`
Use these for rapid prototyping or visual analysis of training runs (e.g., to explore sample outputs, loss curves, etc.).

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

