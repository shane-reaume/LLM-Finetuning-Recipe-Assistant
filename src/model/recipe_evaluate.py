import os
import json
import argparse
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
        examples = json.load(f)
    
    print(f"Loaded {len(examples)} test examples from {test_file_path}")
    return examples

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
    }
    
    # Generate and score
    print("Generating and evaluating responses...")
    for i, example in enumerate(test_examples):
        prompt = example["prompt"]
        reference = example["response"]
        
        # Generate text
        generated = generate_text(model, tokenizer, prompt, generation_params)
        
        # Calculate ROUGE scores
        scores = scorer.score(reference, generated)
        
        # Update metrics
        metrics["rouge1_precision"] += scores["rouge1"].precision
        metrics["rouge1_recall"] += scores["rouge1"].recall
        metrics["rouge1_fmeasure"] += scores["rouge1"].fmeasure
        metrics["rouge2_precision"] += scores["rouge2"].precision
        metrics["rouge2_recall"] += scores["rouge2"].recall
        metrics["rouge2_fmeasure"] += scores["rouge2"].fmeasure
        metrics["rougeL_precision"] += scores["rougeL"].precision
        metrics["rougeL_recall"] += scores["rougeL"].recall
        metrics["rougeL_fmeasure"] += scores["rougeL"].fmeasure
        
        # Print progress and example
        print(f"\nExample {i+1}/{len(test_examples)}:")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Generated: {generated[:100]}...")
        print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}, ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
    
    # Average metrics
    for key in metrics:
        metrics[key] /= len(test_examples)
    
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
    
    # Get paths from args or config
    model_path = args.model_dir if args.model_dir else config["model"]["save_dir"]
    test_file_path = args.test_file if args.test_file else config["testing"]["test_examples_file"]
    
    # Load model and test examples
    model, tokenizer, model_info = load_model_and_tokenizer(model_path)
    test_examples = load_test_examples(test_file_path)
    
    # Evaluate
    metrics = evaluate_generation(model, tokenizer, test_examples, config["testing"]["generation_params"])
    
    # Print results
    print("\n======= MODEL EVALUATION RESULTS =======")
    print(f"ROUGE-1 F1: {metrics['rouge1_fmeasure']:.4f}")
    print(f"ROUGE-2 F1: {metrics['rouge2_fmeasure']:.4f}")
    print(f"ROUGE-L F1: {metrics['rougeL_fmeasure']:.4f}")
    print("========================================\n")

if __name__ == "__main__":
    main()
