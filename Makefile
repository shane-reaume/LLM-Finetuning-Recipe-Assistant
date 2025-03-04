.PHONY: setup test train evaluate demo clean coverage publish-sentiment update-model-card recipe-data recipe-train recipe-train-low-memory recipe-train-medium-memory recipe-evaluate recipe-demo recipe-export recipe-export-versioned recipe-train-test recipe-train-test-cpu

# Setup environment
setup:
	chmod +x setup_env.sh
	./setup_env.sh

# Create test sets
test-set:
	python -m src.data.sentiment_create_test_set

# Train the model
train:
	python -m src.model.sentiment_train

# Evaluate the model
evaluate:
	python -m src.model.evaluate --test_file data/processed/sentiment_test_examples.json

# Run the demo
demo:
	python -m src.demo

# Interactive demo
demo-interactive:
	python -m src.demo --interactive

# Publish sentiment model to Hugging Face
publish-sentiment:
	@if [ -z "$(REPO_NAME)" ]; then \
		echo "Error: REPO_NAME is required (e.g., make publish-sentiment REPO_NAME=username/model-name)"; \
		exit 1; \
	fi
	python -m src.model.sentiment_publish --repo_name="$(REPO_NAME)"

# Update model card on Hugging Face
update-model-card:
	@if [ -z "$(REPO_NAME)" ]; then \
		echo "Error: REPO_NAME is required (e.g., make update-model-card REPO_NAME=username/model-name)"; \
		exit 1; \
	fi
	@if [ -z "$(MODEL_CARD)" ]; then \
		MODEL_CARD="model_card.md"; \
	fi
	python -m src.model.update_model_card --repo_name="$(REPO_NAME)" --model_card="$(MODEL_CARD)"

# Recipe model targets
recipe-data:
	@if [ -z "$(CONFIG)" ]; then \
		CONFIG_PATH="config/text_generation.yaml"; \
		echo "No CONFIG specified, using default: $$CONFIG_PATH"; \
	else \
		CONFIG_PATH="$(CONFIG)"; \
	fi; \
	python -m src.data.recipe_prepare_dataset --config $$CONFIG_PATH $(if $(DATA_DIR),--data_dir $(DATA_DIR),)

recipe-train:
	@if [ -z "$(CONFIG)" ]; then \
		CONFIG_PATH="config/text_generation.yaml"; \
		echo "No CONFIG specified, using default: $$CONFIG_PATH"; \
	else \
		CONFIG_PATH="$(CONFIG)"; \
	fi; \
	if [ -z "$(DATA_DIR)" ]; then \
		echo "Warning: DATA_DIR not specified. The script may fail if dataset preparation hasn't been completed."; \
		PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8 python -m src.model.recipe_train --config $$CONFIG_PATH; \
	else \
		PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8 python -m src.model.recipe_train --config $$CONFIG_PATH --data_dir $(DATA_DIR); \
	fi

recipe-train-low-memory:
	@if [ -z "$(CONFIG)" ]; then \
		CONFIG_PATH="config/text_generation_low_memory.yaml"; \
		echo "Using low-memory configuration: $$CONFIG_PATH"; \
	else \
		CONFIG_PATH="$(CONFIG)"; \
	fi; \
	if [ -z "$(DATA_DIR)" ]; then \
		echo "Warning: DATA_DIR not specified. The script may fail if dataset preparation hasn't been completed."; \
		PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8 python -m src.model.recipe_train --config $$CONFIG_PATH; \
	else \
		PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8 python -m src.model.recipe_train --config $$CONFIG_PATH --data_dir $(DATA_DIR); \
	fi

recipe-train-optimized:
	@if [ -z "$(DATA_DIR)" ]; then \
		echo "Warning: DATA_DIR not specified. The script may fail if dataset preparation hasn't been completed."; \
		PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8 python -m src.model.recipe_train --config config/text_generation_optimized.yaml; \
	else \
		PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8 python -m src.model.recipe_train --config config/text_generation_optimized.yaml --data_dir $(DATA_DIR); \
	fi

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
	@echo "HTML coverage report generated in htmlcov/ directory"
	@echo "View the report by opening htmlcov/index.html in your browser"
	@echo "On Linux: xdg-open htmlcov/index.html"
	@echo "On macOS: open htmlcov/index.html"
	@echo "On Windows: start htmlcov/index.html"

# Clean build artifacts
clean:
	rm -rf __pycache__
	rm -rf **/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

# Quick sanity test for recipe training
recipe-train-test:
ifdef DATA_DIR
	python -m src.data.recipe_prepare_dataset --config config/text_generation_sanity_test.yaml --data_dir $(DATA_DIR)
	python -m src.model.recipe_train --config config/text_generation_sanity_test.yaml --data_dir $(DATA_DIR)
else
	$(error DATA_DIR is not set. Please specify DATA_DIR=path/to/data)
endif

# Quick sanity test for recipe training on CPU (for users with limited GPU memory)
recipe-train-test-cpu:
	@if [ -z "$(DATA_DIR)" ]; then \
		echo "Error: DATA_DIR is required for sanity test (e.g., make recipe-train-test-cpu DATA_DIR=~/recipe_manual_data)"; \
		exit 1; \
	fi; \
	echo "Running a quick sanity test of the recipe training process on CPU (this will be slow)..."; \
	CUDA_VISIBLE_DEVICES=-1 python -m src.model.recipe_train --config config/text_generation_sanity_test.yaml --data_dir $(DATA_DIR)
