#!/bin/bash

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Please install Ollama first."
    exit 1
fi

# Check if model already exists
if ollama list | grep -q "recipe-assistant"; then
    echo "Model recipe-assistant already exists. Remove it first with 'ollama rm recipe-assistant' if you want to reinstall."
    echo "Running existing model..."
    ollama run recipe-assistant
    exit 0
fi

# Pull/create the model
echo "Creating recipe-assistant model in Ollama..."

# Create a simple Modelfile in a temp directory
TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR

# Write the Modelfile
cat > Modelfile << EOF
FROM ${PWD}/models/recipe_assistant_merged

# Basic parameters
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER stop "<|endoftext|>"

# System prompt
SYSTEM """
You are a helpful chef assistant that creates recipes based on available ingredients.
"""
EOF

# Create the model
ollama create recipe-assistant -f Modelfile

# Clean up
cd -
rm -rf $TEMP_DIR

echo "Model created successfully!"
echo "Try it out with: ollama run recipe-assistant"
