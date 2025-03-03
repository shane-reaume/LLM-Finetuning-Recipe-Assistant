import os
import sys
import argparse
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

from src.utils.config_utils import load_config

def export_model_to_ollama(model_dir, name, version=None, replace=False):
    """
    Export a fine-tuned model to Ollama
    
    Args:
        model_dir (str): Path to the model directory
        name (str): Name to use for the Ollama model
        version (str, optional): Version string (e.g., "v1", "v2")
        replace (bool): Whether to replace an existing model
    """
    # Validate model directory
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory {model_dir} not found")
    
    # Generate versioned name if specified
    if version:
        full_name = f"{name}-{version}"
    else:
        full_name = name
        
    # Check if model already exists in Ollama
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    
    if full_name in result.stdout and not replace:
        raise ValueError(f"Model {full_name} already exists in Ollama. Use --replace to overwrite.")
    
    # Create a temporary Modelfile with proper syntax
    modelfile_content = (
        f"FROM {model_dir}\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER top_k 40\n"
        "PARAMETER repetition_penalty 1.2\n"
        "TEMPLATE <<EOT\n"
        "Create a recipe using these ingredients: {{ingredients}}\n"
        "EOT\n"
    )
    
    # Write Modelfile
    with open("Modelfile", "w") as f:
        f.write(modelfile_content)
    
    try:
        # Build the Ollama model
        print(f"Building Ollama model {full_name}...")
        subprocess.run(["ollama", "create", full_name, "-f", "Modelfile"], check=True)
        print(f"Successfully exported model to Ollama as '{full_name}'")
        
        # Save metadata for tracking
        metadata = {
            "original_model": model_dir,
            "ollama_name": full_name,
            "exported_at": datetime.now().isoformat(),
            "version": version
        }
        
        metadata_dir = os.path.join(os.path.dirname(model_dir), "ollama_exports")
        os.makedirs(metadata_dir, exist_ok=True)
        
        with open(os.path.join(metadata_dir, f"{full_name}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"To use the model, run: ollama run {full_name}")
        
    finally:
        # Clean up temporary Modelfile
        if os.path.exists("Modelfile"):
            os.remove("Modelfile")

def main():
    parser = argparse.ArgumentParser(description="Export a fine-tuned model to Ollama")
    parser.add_argument("--model_dir", type=str, default="./models/recipe_assistant",
                       help="Path to the model directory")
    parser.add_argument("--name", type=str, default="recipe-assistant",
                       help="Name to use for the Ollama model")
    parser.add_argument("--version", type=str, 
                       help="Optional version string (e.g., 'v1', 'v2')")
    parser.add_argument("--replace", action="store_true",
                       help="Replace existing model if it exists")
    args = parser.parse_args()
    
    export_model_to_ollama(args.model_dir, args.name, args.version, args.replace)

if __name__ == "__main__":
    main()
