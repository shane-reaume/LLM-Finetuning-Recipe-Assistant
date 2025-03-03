import os
import argparse
import subprocess

def export_to_ollama(model_path, model_name):
    """
    Export a model to Ollama using a simplified approach
    
    Args:
        model_path (str): Path to the model directory
        model_name (str): Name for the Ollama model
    """
    # Create a simple Modelfile
    with open("Modelfile", "w") as f:
        f.write(f"FROM {model_path}\n")
    
    try:
        # Create the Ollama model
        print(f"Creating Ollama model '{model_name}' from {model_path}...")
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", "Modelfile"],
            check=False,
            capture_output=True,
            text=True
        )
        
        # Check if successful
        if result.returncode == 0:
            print(f"Successfully created Ollama model: {model_name}")
            print(f"Try it with: ollama run {model_name}")
        else:
            print(f"Error creating Ollama model: {result.stderr}")
            print("Command output:", result.stdout)
            
    finally:
        # Clean up
        if os.path.exists("Modelfile"):
            os.remove("Modelfile")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a model to Ollama")
    parser.add_argument("--model_path", type=str, default="./models/recipe_assistant",
                       help="Path to the model directory")
    parser.add_argument("--model_name", type=str, default="recipe-assistant",
                       help="Name for the Ollama model")
    args = parser.parse_args()
    
    export_to_ollama(args.model_path, args.model_name)
