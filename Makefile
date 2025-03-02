.PHONY: setup test train evaluate demo clean coverage

# Setup environment
setup:
	chmod +x setup_env.sh
	./setup_env.sh

# Create test sets
test-set:
	python -m src.data.sentiment_create_test_set

# Create balanced test set
balanced-test-set:
	python -m src.data.sentiment_create_balanced_test_set --output data/processed/balanced_test_examples.json

# Train the model
train:
	python -m src.model.sentiment_train

# Evaluate the model
evaluate:
	python -m src.model.evaluate --test_file data/processed/balanced_test_examples.json

# Run the demo
demo:
	python -m src.demo

# Interactive demo
demo-interactive:
	python -m src.demo --interactive

# Recipe model targets
recipe-data:
	python -m src.data.recipe_prepare_dataset --config $(CONFIG) $(if $(DATA_DIR),--data_dir $(DATA_DIR),)

recipe-train:
	python -m src.model.recipe_train --config $(CONFIG) $(if $(DATA_DIR),--data_dir $(DATA_DIR),)

recipe-evaluate:
	python -m src.model.recipe_evaluate --config config/text_generation.yaml

recipe-demo:
	python -m src.recipe_demo

recipe-demo-interactive:
	python -m src.recipe_demo --interactive

# Export recipe model to Ollama
recipe-export:
	python -m src.model.recipe_export_to_ollama --name recipe-assistant

# Export with version
recipe-export-versioned:
	python -m src.model.recipe_export_to_ollama --name recipe-assistant --version $(VERSION)

# Run tests
test:
	pytest

# Run tests with coverage
coverage:
	pytest --cov=src --cov-report=html

# Clean build artifacts
clean:
	rm -rf __pycache__
	rm -rf **/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
