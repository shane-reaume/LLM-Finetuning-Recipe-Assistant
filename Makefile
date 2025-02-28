.PHONY: setup test train evaluate demo clean coverage

# Setup environment
setup:
	chmod +x setup_env.sh
	./setup_env.sh

# Create test sets
test-set:
	python -m src.data.create_test_set

# Create balanced test set
balanced-test-set:
	python -m src.data.create_balanced_test_set --output data/processed/balanced_test_examples.json

# Train the model
train:
	python -m src.model.train

# Evaluate the model
evaluate:
	python -m src.model.evaluate --test_file data/processed/balanced_test_examples.json

# Run the demo
demo:
	python -m src.demo

# Interactive demo
demo-interactive:
	python -m src.demo --interactive

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
