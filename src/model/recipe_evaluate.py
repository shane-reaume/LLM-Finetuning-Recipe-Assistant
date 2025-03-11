import os
import json
import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer

from src.utils.config_utils import load_config

def load_model_and_tokenizer(model_path):
    """
    Load the fine-tuned model and tokenizer
    
    Args:
        model_path (str): Path to the model directory
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    
    # Load model info
    model_info_path = os.path.join(model_path, "model_info.json")
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
    else:
        model_info = {}
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    return model, tokenizer, model_info

def generate_text(model, tokenizer, prompt, generation_params):
    """
    Generate text from a prompt
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt (str): The input prompt
        generation_params (dict): Parameters for text generation
        
    Returns:
        str: Generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate text
    outputs = model.generate(
        **inputs,
        max_new_tokens=generation_params.get("max_new_tokens", 256),
        temperature=generation_params.get("temperature", 0.7),
        top_p=generation_params.get("top_p", 0.9),
        top_k=generation_params.get("top_k", 50),
        do_sample=generation_params.get("do_sample", True),
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode and clean up the text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the output
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]
    
    return generated_text.strip()

def load_test_examples(test_file_path):
    """
    Load test examples from file
    
    Args:
        test_file_path (str): Path to test examples JSON file
        
    Returns:
        list: Test examples
    """
    with open(test_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract examples and convert to the format expected by evaluate_generation
    if "examples" in data:
        examples = data["examples"]
        # Transform the examples to the expected format
        formatted_examples = []
        for example in examples:
            formatted_example = {
                "prompt": f"Create a recipe using these ingredients: {example['ingredients']}",
                "response": example["recipe"]
            }
            formatted_examples.append(formatted_example)
        examples = formatted_examples
    else:
        # Handle as if data itself is the list of examples
        examples = data
        
        # Verify the examples have the expected format
        if examples and isinstance(examples, list):
            if "prompt" not in examples[0] or "response" not in examples[0]:
                print("Warning: Test examples don't have the expected format (prompt/response fields). Attempting conversion...")
                formatted_examples = []
                for example in examples:
                    # Try to handle common alternative formats
                    if "ingredients" in example and "recipe" in example:
                        formatted_example = {
                            "prompt": f"Create a recipe using these ingredients: {example['ingredients']}",
                            "response": example["recipe"]
                        }
                        formatted_examples.append(formatted_example)
                examples = formatted_examples
    
    print(f"Loaded {len(examples)} test examples from {test_file_path}")
    return examples

def evaluate_recipe_quality(generated_text, ingredients_list, reference_text=None):
    """
    Evaluate recipe quality based on recipe-specific metrics
    
    Args:
        generated_text (str): The generated recipe text
        ingredients_list (str): Comma-separated list of ingredients that should be in the recipe
        reference_text (str, optional): Reference recipe text for comparison
        
    Returns:
        dict: Recipe quality metrics
    """
    metrics = {
        "ingredient_coverage": 0.0,  # Percentage of required ingredients mentioned
        "structure_score": 0.0,      # Does it have title, ingredients list, and instructions
        "step_clarity": 0.0,         # Are steps clearly numbered/separated
        "overall_quality": 0.0       # Combined score
    }
    
    # Clean and normalize text for analysis
    generated_lower = generated_text.lower()
    
    # 1. Ingredient Coverage
    ingredients = [ing.strip().lower() for ing in ingredients_list.split(',')]
    found_ingredients = 0
    
    for ingredient in ingredients:
        if ingredient in generated_lower:
            found_ingredients += 1
    
    if ingredients:
        metrics["ingredient_coverage"] = found_ingredients / len(ingredients)
    
    # 2. Structure Score - Check for key recipe components
    structure_points = 0
    
    # Check for title
    title_patterns = [
        r"title:.*\n",
        r"^#.*\n",
        r"^.*recipe.*\n",
        r"^.*dish.*\n"
    ]
    
    for pattern in title_patterns:
        if re.search(pattern, generated_lower, re.MULTILINE):
            structure_points += 1
            break
    
    # Check for ingredients list
    ingredient_list_patterns = [
        r"ingredients:.*\n",
        r"ingredients.*\n-.*\n",
        r"\n-.*\n-.*\n"
    ]
    
    for pattern in ingredient_list_patterns:
        if re.search(pattern, generated_lower, re.MULTILINE):
            structure_points += 1
            break
    
    # Check for instructions
    instruction_patterns = [
        r"instructions:.*\n",
        r"directions:.*\n",
        r"steps:.*\n",
        r"\n\d+\..*\n\d+\.",
        r"step \d+:.*\nstep \d+:"
    ]
    
    for pattern in instruction_patterns:
        if re.search(pattern, generated_lower, re.MULTILINE):
            structure_points += 1
            break
    
    metrics["structure_score"] = structure_points / 3.0
    
    # 3. Step Clarity - Check for numbered steps
    step_patterns = [
        r"\n\d+\.",
        r"\nstep \d+:",
        r"\n-\s+\w+",  # Bullet points also count
    ]
    
    steps_found = 0
    for pattern in step_patterns:
        steps = re.findall(pattern, generated_lower)
        if len(steps) >= 3:  # At least 3 steps for a reasonable recipe
            steps_found = len(steps)
            break
    
    # Score based on number of steps (3-10 steps is reasonable)
    if steps_found >= 3:
        metrics["step_clarity"] = min(1.0, steps_found / 10.0)
    
    # 4. Overall Quality - Weighted combination of other metrics
    metrics["overall_quality"] = (
        0.4 * metrics["ingredient_coverage"] +
        0.3 * metrics["structure_score"] +
        0.3 * metrics["step_clarity"]
    )
    
    return metrics

def evaluate_generation(model, tokenizer, test_examples, generation_params):
    """
    Evaluate the generative model on test examples
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        test_examples (list): List of test examples
        generation_params (dict): Parameters for text generation
        
    Returns:
        dict: Evaluation metrics
    """
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Initialize metrics
    metrics = {
        "rouge1_precision": 0.0,
        "rouge1_recall": 0.0,
        "rouge1_fmeasure": 0.0,
        "rouge2_precision": 0.0,
        "rouge2_recall": 0.0,
        "rouge2_fmeasure": 0.0,
        "rougeL_precision": 0.0,
        "rougeL_recall": 0.0,
        "rougeL_fmeasure": 0.0,
        # Recipe-specific metrics
        "ingredient_coverage": 0.0,
        "structure_score": 0.0,
        "step_clarity": 0.0,
        "overall_quality": 0.0
    }
    
    if not test_examples:
        print("Warning: No test examples to evaluate")
        return metrics
    
    # Generate and score
    print("Generating and evaluating responses...")
    valid_examples = 0
    
    for i, example in enumerate(test_examples):
        try:
            # Check if the example has the required fields
            if not isinstance(example, dict):
                print(f"Warning: Example {i} is not a dictionary, skipping...")
                continue
                
            if "prompt" not in example or "response" not in example:
                print(f"Warning: Example {i} is missing required fields (prompt/response), skipping...")
                continue
                
            prompt = example["prompt"]
            reference = example["response"]
            
            # Extract ingredients from prompt
            ingredients = ""
            ingredients_match = re.search(r"ingredients:?\s*([^<]+)", prompt, re.IGNORECASE)
            if not ingredients_match:
                ingredients_match = re.search(r"using these ingredients:?\s*([^<]+)", prompt, re.IGNORECASE)
            
            if ingredients_match:
                ingredients = ingredients_match.group(1).strip()
            
            # Generate text
            generated = generate_text(model, tokenizer, prompt, generation_params)
            
            # Calculate ROUGE scores
            scores = scorer.score(reference, generated)
            
            # Calculate recipe-specific metrics
            recipe_metrics = evaluate_recipe_quality(generated, ingredients, reference)
            
            # Accumulate metrics
            metrics["rouge1_precision"] += scores["rouge1"].precision
            metrics["rouge1_recall"] += scores["rouge1"].recall
            metrics["rouge1_fmeasure"] += scores["rouge1"].fmeasure
            
            metrics["rouge2_precision"] += scores["rouge2"].precision
            metrics["rouge2_recall"] += scores["rouge2"].recall
            metrics["rouge2_fmeasure"] += scores["rouge2"].fmeasure
            
            metrics["rougeL_precision"] += scores["rougeL"].precision
            metrics["rougeL_recall"] += scores["rougeL"].recall
            metrics["rougeL_fmeasure"] += scores["rougeL"].fmeasure
            
            # Add recipe-specific metrics
            metrics["ingredient_coverage"] += recipe_metrics["ingredient_coverage"]
            metrics["structure_score"] += recipe_metrics["structure_score"]
            metrics["step_clarity"] += recipe_metrics["step_clarity"]
            metrics["overall_quality"] += recipe_metrics["overall_quality"]
            
            valid_examples += 1
            
            # Print progress and detailed metrics for this example
            print(f"\nExample {i+1}/{len(test_examples)}:")
            print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
            print(f"Generated: {generated[:100]}..." if len(generated) > 100 else f"Generated: {generated}")
            print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}, ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
            print(f"Ingredient Coverage: {recipe_metrics['ingredient_coverage']:.2f}")
            print(f"Structure Score: {recipe_metrics['structure_score']:.2f}")
            print(f"Step Clarity: {recipe_metrics['step_clarity']:.2f}")
            print(f"Overall Quality: {recipe_metrics['overall_quality']:.2f}")
            
        except Exception as e:
            print(f"Error processing example {i}: {str(e)}")
            continue
    
    # Average metrics
    if valid_examples > 0:
        for key in metrics:
            metrics[key] /= valid_examples
    else:
        print("Warning: No valid examples were processed.")
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate text generation model")
    parser.add_argument("--config", type=str, default="config/text_generation.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to model directory (overrides config)")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Path to test examples file (overrides config)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get model path from args or config
    if args.model_dir:
        model_path = args.model_dir
    elif "model" in config and "save_dir" in config["model"]:
        model_path = config["model"]["save_dir"]
    else:
        model_path = "./models/recipe_assistant"  # Default fallback
        print(f"Warning: Model path not specified in config, using default: {model_path}")
    
    # Get test file path from args or config
    if args.test_file:
        test_file_path = args.test_file
    elif "testing" in config and "test_examples_file" in config["testing"]:
        test_file_path = config["testing"]["test_examples_file"]
    else:
        test_file_path = "data/processed/recipe_test_examples.json"  # Default fallback
        print(f"Warning: Test file path not specified in config, using default: {test_file_path}")
    
    # Get generation parameters from config or use defaults
    if "testing" in config and "generation_params" in config["testing"]:
        generation_params = config["testing"]["generation_params"]
    else:
        generation_params = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True
        }
        print(f"Warning: Generation parameters not specified in config, using defaults")
    
    # Load model and test examples
    try:
        model, tokenizer, model_info = load_model_and_tokenizer(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    try:
        test_examples = load_test_examples(test_file_path)
    except Exception as e:
        print(f"Error loading test examples: {str(e)}")
        return
    
    # Evaluate
    try:
        metrics = evaluate_generation(model, tokenizer, test_examples, generation_params)
    
        # Print results
        print("\n======= MODEL EVALUATION RESULTS =======")
        print(f"ROUGE-1 F1: {metrics['rouge1_fmeasure']:.4f}")
        print(f"ROUGE-2 F1: {metrics['rouge2_fmeasure']:.4f}")
        print(f"ROUGE-L F1: {metrics['rougeL_fmeasure']:.4f}")
        print("\n--- Recipe-Specific Metrics ---")
        print(f"Ingredient Coverage: {metrics['ingredient_coverage']:.4f} - How well required ingredients are used")
        print(f"Structure Score: {metrics['structure_score']:.4f} - Presence of title, ingredients list, instructions")
        print(f"Step Clarity: {metrics['step_clarity']:.4f} - Clear, numbered recipe steps")
        print(f"Overall Quality: {metrics['overall_quality']:.4f} - Combined recipe quality score")
        print("========================================\n")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return

if __name__ == "__main__":
    main()
