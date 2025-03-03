import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def test_direct_recipe_generation(ingredients=None, temperature=0.8, max_tokens=512):
    """Test direct recipe generation with a simple prompt"""
    
    # Load model and tokenizer
    model_path = "./models/recipe_assistant"
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # Use provided ingredients or default
    if not ingredients:
        ingredients = "ground beef, tortillas, cheese, onions, peppers"
    
    # Create a more detailed prompt
    prompt = f"""Create a complete recipe using: {ingredients}

Recipe Name:"""
    
    print(f"Testing with prompt:\n{prompt}\n")
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.92,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.3,  # Increased to reduce repetition
            no_repeat_ngram_size=3,  # Avoid repeating 3-grams
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print the result
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the output
    recipe = full_text[len(prompt):] if prompt in full_text else full_text
    
    print("\nGENERATED RECIPE:")
    print("=" * 60)
    print(recipe.strip())
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test recipe generation directly")
    parser.add_argument("--ingredients", type=str, help="Comma-separated list of ingredients")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    args = parser.parse_args()
    
    test_direct_recipe_generation(args.ingredients, args.temperature, args.max_tokens)
