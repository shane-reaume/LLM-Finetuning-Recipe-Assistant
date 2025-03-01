import os
import argparse
import time
from colorama import Fore, Style, init

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.config_utils import load_config

# Initialize colorama for colored output
init()

class RecipeGenerator:
    """Class for generating recipes from ingredients"""
    
    def __init__(self, model_dir):
        """Initialize with the model directory"""
        # Load model info
        model_info_path = os.path.join(model_dir, "model_info.json")
        self.model_info = {}
        if os.path.exists(model_info_path):
            import json
            with open(model_info_path, 'r') as f:
                self.model_info = json.load(f)
        
        # Set up model and tokenizer
        print(f"Loading recipe model from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Get prompt template
        self.prompt_template = self.model_info.get("prompt_template", 
                                                "Create a recipe with these ingredients: {ingredients}\n\n")
    
    def generate_recipe(self, ingredients, **generation_kwargs):
        """Generate a recipe given a list of ingredients"""
        # Format the prompt
        prompt = self.prompt_template.format(ingredients=ingredients)
        
        # Set default generation parameters
        params = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True
        }
        
        # Update with any provided parameters
        params.update(generation_kwargs)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        start_time = time.time()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **params,
                pad_token_id=self.tokenizer.eos_token_id
            )
        gen_time = time.time() - start_time
        
        # Decode the output
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the recipe part (removing the prompt)
        if prompt in generated_text:
            recipe = generated_text[len(prompt):]
        else:
            recipe = generated_text
            
        return recipe.strip(), gen_time

def format_recipe(recipe):
    """Apply formatting to make the recipe more readable"""
    # Add color
    title_pattern = recipe.split("\n")[0] if "\n" in recipe else recipe
    colored_recipe = recipe.replace(title_pattern, f"{Fore.CYAN}{title_pattern}{Style.RESET_ALL}", 1)
    
    # Look for sections and highlight them
    sections = ["Ingredients:", "Instructions:", "Directions:", "Steps:", "Preparation:"]
    for section in sections:
        if section in colored_recipe:
            colored_recipe = colored_recipe.replace(
                section, f"{Fore.YELLOW}{section}{Style.RESET_ALL}"
            )
    
    return colored_recipe

def main():
    parser = argparse.ArgumentParser(description="Generate recipes from ingredients")
    parser.add_argument("--model_dir", type=str, default="./models/recipe_assistant",
                        help="Path to the recipe model directory")
    parser.add_argument("--ingredients", type=str, 
                        help="Comma-separated list of ingredients")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    args = parser.parse_args()
    
    # Load the recipe generator
    generator = RecipeGenerator(args.model_dir)
    
    if args.interactive:
        print(f"\n{Fore.GREEN}===== INTERACTIVE RECIPE GENERATOR ====={Style.RESET_ALL}")
        print("Enter ingredients (separated by commas) and get recipe suggestions.")
        print("Type 'quit', 'exit', or press Ctrl+C to exit.\n")
        
        while True:
            try:
                ingredients = input(f"{Fore.GREEN}Enter ingredients:{Style.RESET_ALL} ")
                if ingredients.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not ingredients.strip():
                    continue
                
                print(f"\n{Fore.GREEN}Generating recipe...{Style.RESET_ALL}")
                recipe, gen_time = generator.generate_recipe(ingredients)
                
                print(f"\n{format_recipe(recipe)}")
                print(f"\n{Fore.BLUE}Generation time: {gen_time:.2f} seconds{Style.RESET_ALL}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}Thank you for using the Recipe Generator!{Style.RESET_ALL}")
    
    else:
        # Use provided ingredients or default examples
        ingredients = args.ingredients if args.ingredients else "chicken, rice, onion, garlic"
        
        print(f"{Fore.GREEN}Generating recipe with: {ingredients}{Style.RESET_ALL}\n")
        recipe, gen_time = generator.generate_recipe(ingredients)
        
        print(format_recipe(recipe))
        print(f"\n{Fore.BLUE}Generation time: {gen_time:.2f} seconds{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
