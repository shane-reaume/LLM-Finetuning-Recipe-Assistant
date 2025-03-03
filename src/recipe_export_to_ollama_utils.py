import os
import argparse
import subprocess
import shutil
from pathlib import Path

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
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print(f"Successfully created Ollama model: {model_name}")
            print(f"Try it with: ollama run {model_name}")
            return True
        else:
            print(f"Error creating Ollama model: {stderr}")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def export_with_parameters(model_dir, model_name, temp=0.7, top_p=0.9):
    """
    Export a model to Ollama with temperature and top_p parameters
    
    Args:
        model_dir (str): Path to the model directory
        model_name (str): Name for the Ollama model
        temp (float): Temperature parameter for generation
        top_p (float): Top-p parameter for generation
    """
    # First, check if the model directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} does not exist.")
        return False
    
    # Create a Modelfile with parameters
    modelfile = f"""FROM {model_dir}

# Set parameters
PARAMETER temperature {temp}
PARAMETER top_p {top_p}
"""
    
    # Write the Modelfile
    with open("Modelfile", "w") as f:
        f.write(modelfile)
    
    print(f"Created Modelfile with content:\n{modelfile}")
    
    # Run ollama create command
    try:
        cmd = ["ollama", "create", model_name, "-f", "Modelfile"]
        print(f"Running: {' '.join(cmd)}")
        
        process = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True
        )
        
        if process.returncode == 0:
            print(f"Successfully created Ollama model: {model_name}")
            print(f"Try it with: ollama run {model_name}")
            
            # Clean up the Modelfile
            if os.path.exists("Modelfile"):
                os.remove("Modelfile")
                
            return True
        else:
            print(f"Error creating Ollama model: {process.stderr}")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def export_with_prompt_template(model_dir, model_name, system_prompt=None, prompt_template=None):
    """
    Export a model to Ollama with a system prompt and prompt template
    
    Args:
        model_dir (str): Path to the model directory
        model_name (str): Name for the Ollama model
        system_prompt (str, optional): System prompt to set context
        prompt_template (str, optional): Template for user prompts
    """
    # Default system prompt for recipe generation
    if system_prompt is None:
        system_prompt = """You are RecipeBot, an AI assistant that specializes in creating delicious recipes.
Given a list of ingredients, you will create a complete recipe with a title, ingredients list,
and step-by-step instructions. Be creative and helpful!"""
    
    # Default prompt template for recipe generation
    if prompt_template is None:
        prompt_template = """<|im_start|>system
{{.System}}
<|im_start|>user
Here are my ingredients:
{{.Prompt}}
<|im_start|>assistant
"""
    
    # Create a Modelfile with the prompt template
    modelfile = f"""FROM {model_dir}

# Set the system message that provides context for the model
SYSTEM "{system_prompt}"

# Define the template format for inputs
TEMPLATE "{prompt_template}"

# Set parameters for generation
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
"""
    
    # Write the Modelfile
    with open("Modelfile", "w") as f:
        f.write(modelfile)
    
    print(f"Created Modelfile with prompt template")
    
    # Run ollama create command
    try:
        cmd = ["ollama", "create", model_name, "-f", "Modelfile"]
        print(f"Running: {' '.join(cmd)}")
        
        process = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True
        )
        
        if process.returncode == 0:
            print(f"Successfully created Ollama model: {model_name}")
            print(f"Try it with: ollama run {model_name}")
            
            # Clean up the Modelfile
            if os.path.exists("Modelfile"):
                os.remove("Modelfile")
                
            return True
        else:
            print(f"Error creating Ollama model: {process.stderr}")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    """Main function to parse arguments and export model to Ollama"""
    parser = argparse.ArgumentParser(description="Export model to Ollama with various options")
    parser.add_argument("--model_dir", required=True, help="Path to the model directory")
    parser.add_argument("--model_name", required=True, help="Name for the Ollama model")
    parser.add_argument("--type", choices=["minimal", "parameters", "template"], 
                        default="parameters", help="Type of export to perform")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p parameter")
    parser.add_argument("--system_prompt", help="System prompt for template export")
    
    args = parser.parse_args()
    
    if args.type == "minimal":
        export_minimal(args.model_dir, args.model_name)
    elif args.type == "parameters":
        export_with_parameters(args.model_dir, args.model_name, args.temperature, args.top_p)
    elif args.type == "template":
        export_with_prompt_template(args.model_dir, args.model_name, args.system_prompt)

if __name__ == "__main__":
    main() 