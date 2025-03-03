import gradio as gr
import torch
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class WebRecipeGenerator:
    """Recipe generator for web interface"""
    
    def __init__(self, model_dir="./models/recipe_assistant"):
        """Initialize generator with model"""
        print(f"Loading model from {model_dir}...")
        
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model with appropriate settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print(f"Model loaded successfully on {self.model.device}")
    
    def format_recipe(self, raw_recipe):
        """Format the raw recipe text into a structured format"""
        # Try to identify and structure the recipe
        sections = {}
        current_section = "title"
        sections[current_section] = []
        
        for line in raw_recipe.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if re.match(r'^ingredients:?$', line.lower()):
                current_section = "ingredients"
                continue
            elif re.match(r'^(instructions|directions|steps|preparation):?$', line.lower()):
                current_section = "instructions"
                continue
            
            # Add line to current section
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(line)
        
        # Format the recipe nicely
        formatted_recipe = ""
        
        if "title" in sections and sections["title"]:
            formatted_recipe += f"# {sections['title'][0]}\n\n"
        
        if "ingredients" in sections and sections["ingredients"]:
            formatted_recipe += "## Ingredients\n\n"
            for item in sections["ingredients"]:
                formatted_recipe += f"- {item}\n"
            formatted_recipe += "\n"
        
        if "instructions" in sections and sections["instructions"]:
            formatted_recipe += "## Instructions\n\n"
            for i, step in enumerate(sections["instructions"]):
                formatted_recipe += f"{i+1}. {step}\n"
        
        return formatted_recipe
        
    def generate_recipe(self, ingredients, prompt_style, temperature, max_tokens):
        """Generate a recipe with various prompt styles"""
        
        # Different prompt templates
        prompt_templates = {
            "basic": f"Create a recipe with these ingredients: {ingredients}\n\n",
            "structured": f"Create a recipe using: {ingredients}\n\nRecipe Name:",
            "chef": f"As a professional chef, create a detailed recipe with: {ingredients}\n\nRecipe:",
            "step_by_step": f"Write a step-by-step recipe using these ingredients: {ingredients}\n\nRecipe Name:"
        }
        
        prompt = prompt_templates.get(prompt_style, prompt_templates["basic"])
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generation parameters
        gen_params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.92,
            "top_k": 50,
            "do_sample": True,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)
            
        # Decode
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        recipe_text = full_text[len(prompt):] if prompt in full_text else full_text
        
        # Format the recipe for better presentation
        formatted_recipe = self.format_recipe(recipe_text.strip())
        
        # Return both raw and formatted versions
        return recipe_text.strip(), formatted_recipe

# Set up the Gradio interface
def create_recipe_interface():
    generator = WebRecipeGenerator()
    
    with gr.Blocks(title="Recipe Generator") as app:
        gr.Markdown("# 🍳 AI Recipe Generator")
        gr.Markdown("Generate recipes from your ingredients using a fine-tuned language model.")
        
        with gr.Row():
            with gr.Column():
                ingredients = gr.Textbox(
                    label="Ingredients (comma separated)", 
                    placeholder="chicken, rice, garlic, onions, bell peppers",
                    lines=3
                )
                
                with gr.Row():
                    prompt_style = gr.Dropdown(
                        label="Prompt Style",
                        choices=["basic", "structured", "chef", "step_by_step"],
                        value="structured"
                    )
                    temperature = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                        label="Temperature (higher = more creative)"
                    )
                
                max_tokens = gr.Slider(
                    minimum=100, maximum=1024, value=512, step=32,
                    label="Maximum Tokens"
                )
                
                generate_btn = gr.Button("Generate Recipe", variant="primary")
            
            with gr.Column():
                raw_output = gr.Textbox(label="Raw Output", lines=10, visible=False)
                formatted_output = gr.Markdown(label="Recipe")
        
        generate_btn.click(
            fn=generator.generate_recipe,
            inputs=[ingredients, prompt_style, temperature, max_tokens],
            outputs=[raw_output, formatted_output]
        )
    
    return app

if __name__ == "__main__":
    # Add requirements installation check
    requirements = ["gradio"]
    missing_requirements = []
    
    for req in requirements:
        try:
            __import__(req)
        except ImportError:
            missing_requirements.append(req)
    
    if missing_requirements:
        print(f"Missing requirements: {', '.join(missing_requirements)}")
        print("Installing required packages...")
        os.system(f"pip install {' '.join(missing_requirements)}")
        print("Installation complete. Restarting script...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    
    # Launch the app
    app = create_recipe_interface()
    app.launch(share=True)
