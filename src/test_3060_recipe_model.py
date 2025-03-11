#!/usr/bin/env python
"""
Testing script specifically designed for the 3060 RTX model with debugging capabilities.
This uses the exact same prompt format as in training to ensure consistency.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import re
from peft import PeftModel, PeftConfig

def format_response(text):
    """Format the response for better readability"""
    # Bold the title
    text = re.sub(r'(Title:.*?)(\n|$)', r'**\1**\2', text, flags=re.IGNORECASE)
    
    # Format ingredients section
    text = re.sub(r'(Ingredients:.*?)(\n|$)', r'**\1**\2', text, flags=re.IGNORECASE)
    
    # Format instructions section
    text = re.sub(r'(Instructions:.*?)(\n|$)', r'**\1**\2', text, flags=re.IGNORECASE)
    
    return text

def load_model(model_dir):
    """Load the model with detailed logging for debugging"""
    print(f"Loading model from {model_dir}...")
    
    try:
        # Check if this is a PEFT/LoRA model
        if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
            print("LoRA adapter detected, loading with PEFT...")
            
            # Load config to get base model name
            config = PeftConfig.from_pretrained(model_dir)
            base_model_name = config.base_model_name_or_path
            print(f"Base model: {base_model_name}")
            
            # Load base model and tokenizer
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, model_dir)
            print(f"Successfully loaded LoRA adapter from {model_dir}")
        else:
            # Load as standard model
            print("Loading as standard model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            print(f"Successfully loaded standard model from {model_dir}")
        
        # Check if tokenizer has special tokens
        special_tokens = []
        for token in ["<|system|>", "<|user|>", "<|assistant|>", "<|endoftext|>"]:
            if token in tokenizer.get_vocab():
                special_tokens.append(token)
        
        if special_tokens:
            print(f"Tokenizer has the following special tokens: {', '.join(special_tokens)}")
        else:
            print("Warning: Tokenizer does not have chat-specific special tokens")
            
        # Get model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def generate_recipe(model, tokenizer, ingredients, temperature=0.5, max_tokens=512):
    """Generate a recipe with detailed debugging"""
    # Create the exact prompt format used during training
    prompt = f"<|system|>You are a recipe assistant. Create detailed recipes with exact measurements and clear instructions.<|endoftext|>\n<|user|>Write a complete recipe using these ingredients: {ingredients}. Include a title, ingredients list with measurements, and numbered cooking steps.<|endoftext|>\n<|assistant|>"
    
    print(f"\nPrompt: {prompt}\n")
    
    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    
    # Print token count
    print(f"Input token count: {input_ids.shape[1]}")
    
    # Generation parameters
    gen_params = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.92,
        "top_k": 50,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    # Generate
    print(f"Generating with temperature={temperature}, max_tokens={max_tokens}...")
    start_time = time.time()
    
    with torch.no_grad():
        output = model.generate(input_ids, **gen_params)
    
    gen_time = time.time() - start_time
    
    # Decode
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract just the assistant's response (remove the prompt)
    try:
        assistant_start = prompt.index("<|assistant|>") + len("<|assistant|>")
        recipe_text = full_output[assistant_start:].strip()
    except:
        recipe_text = full_output.replace(prompt, "").strip()
    
    # Token statistics
    output_tokens = output.shape[1] - input_ids.shape[1]
    tokens_per_sec = output_tokens / gen_time if gen_time > 0 else 0
    
    print(f"\nGeneration stats:")
    print(f"- Generation time: {gen_time:.2f} seconds")
    print(f"- Output tokens: {output_tokens}")
    print(f"- Speed: {tokens_per_sec:.2f} tokens/sec")
    
    return recipe_text

def main():
    parser = argparse.ArgumentParser(description="Test the 3060 RTX recipe model")
    parser.add_argument("--model_dir", type=str, default="models/test_recipe_assistant_3060_geforce_rtx", help="Path to the model directory")
    parser.add_argument("--ingredients", type=str, required=True, help="Comma-separated list of ingredients")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_dir)
    if model is None or tokenizer is None:
        return
    
    # Generate recipe
    recipe = generate_recipe(
        model, 
        tokenizer, 
        args.ingredients,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Format and print recipe
    print("\n" + "="*40)
    print("GENERATED RECIPE")
    print("="*40 + "\n")
    print(recipe)
    print("\n" + "="*40)
    
    # Basic quality assessment
    quality_check = {}
    quality_check["length"] = len(recipe.split())
    quality_check["has_title"] = "title:" in recipe.lower() or any(line.strip().endswith(":") for line in recipe.splitlines()[:3])
    quality_check["has_ingredients"] = "ingredients:" in recipe.lower() or "-" in recipe
    quality_check["has_instructions"] = "instructions:" in recipe.lower() or any(line.strip().startswith(str(i)) for i in range(1, 10) for line in recipe.splitlines())
    
    print("\nQuality Check:")
    print(f"- Word count: {quality_check['length']} words")
    print(f"- Has title: {'✓' if quality_check['has_title'] else '✗'}")
    print(f"- Has ingredients list: {'✓' if quality_check['has_ingredients'] else '✗'}")
    print(f"- Has instructions: {'✓' if quality_check['has_instructions'] else '✗'}")
    
    # Check if all ingredients are used
    ingredients_list = [ing.strip().lower() for ing in args.ingredients.split(",")]
    recipe_lower = recipe.lower()
    used_ingredients = sum(1 for ing in ingredients_list if ing in recipe_lower)
    print(f"- Used {used_ingredients}/{len(ingredients_list)} ingredients")
    
    # Overall assessment
    overall_score = (
        (quality_check["length"] >= 100) + 
        quality_check["has_title"] + 
        quality_check["has_ingredients"] + 
        quality_check["has_instructions"] + 
        (used_ingredients / max(1, len(ingredients_list)) >= 0.7)
    ) / 5.0
    
    print(f"- Overall score: {overall_score:.1f}/1.0")
    
    if overall_score >= 0.8:
        print("Assessment: Excellent recipe! ✨")
    elif overall_score >= 0.6:
        print("Assessment: Good recipe with minor issues")
    elif overall_score >= 0.4:
        print("Assessment: Average recipe, needs improvement")
    else:
        print("Assessment: Poor quality, model needs significant improvement")

if __name__ == "__main__":
    main() 