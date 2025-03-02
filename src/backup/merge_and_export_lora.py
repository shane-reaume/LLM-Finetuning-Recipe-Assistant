import os
import argparse
import subprocess
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig

def merge_lora_to_base_model(base_model_name, lora_model_path, output_dir):
    """
    Merge a LoRA model with its base model and export a complete model
    
    Args:
        base_model_name (str): Name or path of the base model
        lora_model_path (str): Path to the LoRA model directory
        output_dir (str): Directory to save the merged model
    """
    print(f"Loading base model: {base_model_name}")
    
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=False  # Allow downloading if needed
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        local_files_only=False
    )
    
    # Load the LoRA model config to check if it's actually a LoRA model
    try:
        peft_config = PeftConfig.from_pretrained(lora_model_path)
        is_lora = True
        print(f"Detected LoRA model with config: {peft_config}")
    except Exception as e:
        print(f"Not a LoRA model or error loading config: {e}")
        is_lora = False
    
    # If it's a LoRA model, merge it with the base model
    if (is_lora):
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        print("Merging weights...")
        model = model.merge_and_unload()
        print("Successfully merged LoRA weights with base model")
    else:
        # If not a LoRA model, just use it directly
        print("Using model as-is (not a LoRA model)")
        model = base_model
    
    # Copy model info if it exists
    model_info_path = os.path.join(lora_model_path, "model_info.json")
    if os.path.exists(model_info_path):
        print(f"Copying model_info.json from {model_info_path}")
        try:
            import json
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                
            # Add merged info
            model_info["merged_with_base"] = base_model_name
            model_info["is_lora_merged"] = is_lora
        except Exception as e:
            print(f"Error reading model_info.json: {e}")
            model_info = {
                "base_model": base_model_name,
                "lora_path": lora_model_path,
                "is_lora_merged": is_lora
            }
    else:
        model_info = {
            "base_model": base_model_name,
            "lora_path": lora_model_path,
            "is_lora_merged": is_lora
        }
    
    # Save the merged model to the output directory
    print(f"Saving merged model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save model info
    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        import json
        json.dump(model_info, f, indent=2)
    
    print(f"Model successfully saved to {output_dir}")
    return True

def export_merged_model_to_ollama(model_path, ollama_name):
    """Export the merged model to Ollama"""
    # Using regular quotes and escaping as needed
    modelfile_content = f"""FROM {model_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repetition_penalty 1.2
TEMPLATE <<EOT
Create a recipe using these ingredients: {{ingredients}}

EOT
"""
    
    # Debug: Show exactly what we're writing to the file
    print("\nWriting Modelfile with content:")
    print("-" * 40)
    print(modelfile_content)
    print("-" * 40)
    
    with open("Modelfile", "w") as f:
        f.write(modelfile_content)
    
    try:
        print(f"Creating Ollama model {ollama_name}...")
        result = subprocess.run(
            ["ollama", "create", ollama_name, "-f", "Modelfile"],
            check=False,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"Successfully created Ollama model: {ollama_name}")
            print(f"Try it with: ollama run {ollama_name}")
            return True
        else:
            print(f"Error creating Ollama model: {result.stderr}")
            print("Command output:", result.stdout)
            return False
    finally:
        if os.path.exists("Modelfile"):
            os.remove("Modelfile")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA model with base model and export to Ollama")
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                      help="Name or path of the base model")
    parser.add_argument("--lora_model", type=str, default="./models/recipe_assistant",
                      help="Path to the LoRA model directory")
    parser.add_argument("--output_dir", type=str, default="./models/recipe_assistant_merged",
                      help="Directory to save the merged model")
    parser.add_argument("--ollama_name", type=str, default="recipe-assistant",
                      help="Name for the Ollama model")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Merge models
    success = merge_lora_to_base_model(args.base_model, args.lora_model, args.output_dir)
    
    if success:
        # Export to Ollama
        export_merged_model_to_ollama(args.output_dir, args.ollama_name)
    else:
        print("Model merging failed, cannot export to Ollama")

if __name__ == "__main__":
    main()
