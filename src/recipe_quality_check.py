#!/usr/bin/env python
"""
Recipe Quality Checker

This script evaluates a single recipe generation using recipe-specific metrics.
It's useful for quick testing of recipe quality without running the full evaluation pipeline.
"""

import argparse
import sys
import os
from src.model.recipe_evaluate import evaluate_recipe_quality, load_model_and_tokenizer, generate_text
import time
import torch

def generate_recipe(model, tokenizer, prompt, generation_params):
    """Generate a recipe with the given model and prompt"""
    # Create a properly formatted prompt with the chat format
    prompt = f"<|system|>You are a recipe assistant. Create detailed recipes with exact measurements and clear instructions.<|endoftext|>\n<|user|>Write a complete recipe using these ingredients: {prompt}. Include a title, ingredients list with measurements, and numbered cooking steps.<|endoftext|>\n<|assistant|>"
    
    print(f"\nUsing prompt template that matches training format\n")
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_params)
    
    generation_time = time.time() - start_time
    
    # Decode and extract recipe text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt part to get just the recipe
    try:
        recipe_text = full_output.replace(prompt, "").strip()
    except:
        recipe_text = full_output
    
    print(f"Generation time: {generation_time:.2f} seconds")
    
    return recipe_text

def main():
    parser = argparse.ArgumentParser(description="Evaluate a single recipe generation")
    parser.add_argument("--model_dir", type=str, default="models/recipe_assistant",
                        help="Path to model directory")
    parser.add_argument("--ingredients", type=str, required=True,
                        help="Comma-separated list of ingredients")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    args = parser.parse_args()
    
    # Load model
    try:
        print(f"Loading recipe model from {args.model_dir}...")
        model, tokenizer, _ = load_model_and_tokenizer(args.model_dir)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return 1
    
    # Create prompt
    prompt = f"Create a recipe using these ingredients: {args.ingredients}"
    print(f"Generating recipe with: {args.ingredients}")
    
    # Set generation parameters
    generation_params = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": True
    }
    
    # Generate recipe
    try:
        print("\nGenerating recipe...")
        recipe = generate_recipe(model, tokenizer, args.ingredients, generation_params)
        
        # Print the generated recipe
        print("\n===== GENERATED RECIPE =====")
        print(recipe)
        print("===========================\n")
        
        # Evaluate recipe quality
        metrics = evaluate_recipe_quality(recipe, args.ingredients)
        
        # Print metrics
        print("\n===== RECIPE QUALITY METRICS =====")
        print(f"Ingredient Coverage: {metrics['ingredient_coverage']:.2f} - How well required ingredients are used")
        print(f"Structure Score: {metrics['structure_score']:.2f} - Presence of title, ingredients list, instructions")
        print(f"Step Clarity: {metrics['step_clarity']:.2f} - Clear, numbered recipe steps")
        print(f"Overall Quality: {metrics['overall_quality']:.2f} - Combined recipe quality score")
        print("==================================\n")
        
        # Provide interpretation
        print("Interpretation:")
        if metrics['overall_quality'] >= 0.8:
            print("✅ Excellent recipe! Well-structured with good use of ingredients.")
        elif metrics['overall_quality'] >= 0.6:
            print("✓ Good recipe with room for improvement.")
        elif metrics['overall_quality'] >= 0.4:
            print("⚠️ Average recipe quality. Consider retraining with better examples.")
        else:
            print("❌ Poor recipe quality. The model needs significant improvement.")
        
        # Specific feedback
        if metrics['ingredient_coverage'] < 0.7:
            print("- The recipe doesn't use all the requested ingredients effectively.")
        if metrics['structure_score'] < 0.7:
            print("- The recipe structure could be improved (title, ingredients list, instructions).")
        if metrics['step_clarity'] < 0.7:
            print("- The cooking steps could be clearer or more detailed.")
            
    except Exception as e:
        print(f"Error generating or evaluating recipe: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 