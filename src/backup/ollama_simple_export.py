import os
import argparse
import subprocess
import shutil

def export_to_ollama(model_dir, model_name):
    """
    Simple function to export a model to Ollama
    """
    # First, check if the model directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} does not exist.")
        return False
    
    # Create a minimal Modelfile
    modelfile = f"""FROM {model_dir}

# Set parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
    
    # Write the Modelfile
    with open("Modelfile", "w") as f:
        f.write(modelfile)
    
    print(f"Created Modelfile with content:\n{modelfile}")
    
    # Run ollama create command
    try:
        print(f"Creating Ollama model {model_name}...")
        cmd = ["ollama", "create", model_name, "-f", "Modelfile"]
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check result
        if result.returncode == 0:
            print(f"Success! Model {model_name} created in Ollama.")
            print(f"Try it with: ollama run {model_name}")
            return True
        else:
            print(f"Error: {result.stderr}")
            print(f"Output: {result.stdout}")
            return False
            
    finally:
        # Clean up
        if os.path.exists("Modelfile"):
            os.remove("Modelfile")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple tool to export a model to Ollama")
    parser.add_argument("--model_dir", type=str, default="./models/recipe_assistant_merged",
                      help="Path to the model directory")
    parser.add_argument("--model_name", type=str, default="recipe-assistant",
                      help="Name to use for the Ollama model")
    args = parser.parse_args()
    
    export_to_ollama(args.model_dir, args.model_name)
