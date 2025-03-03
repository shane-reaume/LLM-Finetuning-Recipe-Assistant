import os
import argparse
import subprocess

def export_minimal(model_path, model_name):
    """
    Create the simplest possible Ollama model using only the FROM directive
    
    Args:
        model_path (str): Path to the model directory
        model_name (str): Name for the Ollama model
    """
    # Create the absolute simplest Modelfile
    modelfile = f"FROM {model_path}"
    
    with open("Modelfile", "w") as f:
        f.write(modelfile)
    
    print(f"Created minimal Modelfile: {modelfile}")
    
    # Run the ollama command
    try:
        cmd = ["ollama", "create", model_name, "-f", "Modelfile"]
        print(f"Running: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Get the final return code
        return_code = process.poll()
        
        # Print any stderr
        if return_code != 0:
            stderr = process.stderr.read()
            print(f"Error (code {return_code}):\n{stderr}")
            return False
        
        print(f"Success! Model {model_name} created in Ollama.")
        print(f"Try running: ollama run {model_name}")
        return True
        
    finally:
        # Clean up
        if os.path.exists("Modelfile"):
            os.remove("Modelfile")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create minimal Ollama model")
    parser.add_argument("--model_path", type=str, default="./models/recipe_assistant_merged",
                      help="Path to the model directory")
    parser.add_argument("--model_name", type=str, default="recipe-assistant",
                      help="Name for the Ollama model")
    args = parser.parse_args()
    
    export_minimal(args.model_path, args.model_name)
