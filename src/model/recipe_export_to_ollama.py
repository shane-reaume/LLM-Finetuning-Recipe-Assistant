import os
import sys
import argparse
import json
import subprocess
import shutil

from src.utils.config_utils import load_config

def export_model_to_ollama(model_dir, model_name, description=None):
    """
    Export a fine-tuned text generation model to Ollama
    
    Args:
        model_dir (str): Path to the fine-tuned model directory
        model_name (str): Name to give the model in Ollama
        description (str, optional): Model description
    """
    print(f"Preparing to export model from {model_dir} to Ollama as '{model_name}'")
    
    # Load model info
    model_info_path = os.path.join(model_dir, "model_info.json")
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
    else:
        model_info = {}
    
    # Create temporary directory for Ollama model files
    ollama_dir = os.path.join(os.getcwd(), "ollama_export")
    os.makedirs(ollama_dir, exist_ok=True)
    
    # Create Modelfile for Ollama
    base_model = model_info.get("model_name", "tinyllama:latest").split('/')[-1]
    if ":" not in base_model:
        base_model += ":latest"
    
    modelfile_path = os.path.join(ollama_dir, "Modelfile")
    with open(modelfile_path, 'w') as f:
        f.write(f"FROM {base_model}\n\n")
        
        # Add description
        if description:
            f.write(f"DESCRIPTION {description}\n\n")
        elif model_info.get("prompt_template"):
            f.write(f"DESCRIPTION Fine-tuned {base_model} with a custom prompt template\n\n")
        
        # Add system prompt based on the original prompt template
        if model_info.get("prompt_template"):
            system_prompt = f"You are a helpful assistant that specializes in generating recipes. " \
                          f"When provided with ingredients, create delicious recipes using them."
            f.write(f'SYSTEM """{system_prompt}"""\n\n')
        
        # Add parameter settings
        f.write("PARAMETER temperature 0.7\n")
        f.write("PARAMETER top_p 0.9\n")
        f.write("PARAMETER stop \"\\n\\n\"\n\n")
        
        # Add the model path (for LoRA adapters)
        if model_info.get("lora_enabled", False):
            adapter_path = os.path.join(model_dir, "adapter_model.bin")
            if os.path.exists(adapter_path):
                # Copy adapter to the Ollama directory
                shutil.copy(adapter_path, os.path.join(ollama_dir, "adapter_model.bin"))
                f.write("ADAPTER adapter_model.bin\n\n")
            else:
                print(f"Warning: LoRA adapter file not found at {adapter_path}")
    
    print(f"Created Modelfile at {modelfile_path}")
    print("\nTo import the model into Ollama:")
    print(f"1. Navigate to: {ollama_dir}")
    print(f"2. Run: ollama create {model_name} -f Modelfile")
    print(f"3. Then use: ollama run {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model to Ollama")
    parser.add_argument("--model_dir", type=str, default=None, 
                        help="Path to the fine-tuned model directory")
    parser.add_argument("--config", type=str, default="config/text_generation.yaml",
                        help="Path to configuration file")
    parser.add_argument("--name", type=str, default="recipe-assistant",
                        help="Name to give the model in Ollama")
    parser.add_argument("--description", type=str, default=None,
                        help="Optional description for the model")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get model directory from args or config
    model_dir = args.model_dir if args.model_dir else config["model"]["save_dir"]
    
    # Export model
    export_model_to_ollama(model_dir, args.name, args.description)

if __name__ == "__main__":
    main()
