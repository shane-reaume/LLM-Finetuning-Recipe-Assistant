#!/bin/bash

# Colors for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Ollama is installed and running
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Ollama is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if recipe-assistant model exists
if ! ollama list | grep -q "recipe-assistant"; then
    echo -e "${RED}recipe-assistant model not found in Ollama. Did you create it?${NC}"
    exit 1
fi

# Try to find where Ollama stores its models
echo -e "${BLUE}Searching for Ollama models directory...${NC}"

# List of possible directories
possible_dirs=(
    "/home/shane/.ollama"
    "/home/shane/.local/share/ollama"
    "/var/lib/ollama"
    "/usr/share/ollama"
)

# Check for running Ollama process to get hints
echo -e "${BLUE}Checking Ollama process...${NC}"
ollama_process=$(ps aux | grep ollama | grep -v grep)
echo "$ollama_process"

# Try to find the models directory
found=false
for dir in "${possible_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}Found Ollama directory: $dir${NC}"
        if [ -d "$dir/models" ]; then
            echo -e "${GREEN}Found models directory: $dir/models${NC}"
            echo -e "${BLUE}Contents:${NC}"
            ls -la "$dir/models"
            found=true
        fi
    fi
done

# If not found, try a system-wide search (this might take a while)
if [ "$found" = false ]; then
    echo -e "${BLUE}Could not find models in expected locations. Trying system search...${NC}"
    echo -e "${BLUE}This may take a few minutes...${NC}"
    
    # Look for Ollama configuration or model files
    model_files=$(sudo find / -type f -name "recipe-assistant*" -o -name "ollama.json" 2>/dev/null)
    
    if [ -n "$model_files" ]; then
        echo -e "${GREEN}Found Ollama-related files:${NC}"
        echo "$model_files"
    else
        echo -e "${RED}Could not locate Ollama model files.${NC}"
    fi
fi

# Provide usage instructions regardless
echo ""
echo -e "${GREEN}=== RECIPE ASSISTANT MODEL USAGE ===${NC}"
echo -e "${BLUE}Even if we couldn't find the model files, you can still use your model with:${NC}"
echo ""
echo "ollama run recipe-assistant"
echo ""
echo -e "${BLUE}Example prompt:${NC}"
echo 'Create a recipe with these ingredients: chicken, rice, bell peppers, onions'
echo ""
echo -e "${BLUE}For a one-shot command:${NC}"
echo 'echo "Create a recipe with these ingredients: cheese, pasta, tomatoes" | ollama run recipe-assistant'
