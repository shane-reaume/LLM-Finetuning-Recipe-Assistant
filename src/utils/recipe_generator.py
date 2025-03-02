import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleRecipeGenerator:
    """A simplified recipe generator that loads the model once and can be reused"""
    
    def __init__(self, model_path="./models/recipe_assistant"):
        """Initialize with model path"""
        print(f"Loading model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print("Model loaded successfully!")
    
    def generate_recipe(self, ingredients, template="basic", max_tokens=512, temperature=0.7):
        """Generate a recipe based on ingredients"""
        
        # Different prompt templates to try
        templates = {
            "basic": f"Create a recipe with these ingredients: {ingredients}\n\n",
            "structured": f"Create a recipe using the following ingredients: {ingredients}\n\nRecipe Name:",
            "detailed": f"Write a recipe that uses these ingredients: {ingredients}\n\nRecipe Name:",
            "chef": f"As a professional chef, create a recipe with: {ingredients}\n\nRecipe:"
        }
        
        prompt = templates.get(template, templates["basic"])
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.92,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        recipe = full_text[len(prompt):] if prompt in full_text else full_text
        
        return recipe.strip()

# Simple usage example
if __name__ == "__main__":
    generator = SimpleRecipeGenerator()
    ingredients = "chicken, rice, onions, bell peppers, garlic"
    
    print("\nTesting different prompt templates:\n")
    
    for template in ["basic", "structured", "detailed", "chef"]:
        print(f"\n=== {template.upper()} TEMPLATE ===")
        recipe = generator.generate_recipe(ingredients, template=template, max_tokens=300)
        print(recipe)
        print("="*40)
