import os
import json
import time
import argparse
from tqdm import tqdm
import numpy as np

from src.utils.config_utils import load_config
from src.model.inference import SentimentClassifier

def load_test_examples(test_examples_file):
    """
    Load test examples from file
    
    Args:
        test_examples_file (str): Path to test examples JSON file
        
    Returns:
        tuple: Lists of texts and labels
    """
    with open(test_examples_file, 'r') as f:
        examples = json.load(f)
    
    texts = [example["text"] for example in examples]
    
    # Ensure labels are integers
    labels = [int(example["label"]) for example in examples]
    
    print(f"Label distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    return texts, labels

def evaluate_model(config, model_dir=None, detailed=False, test_file=None):
    """
    Evaluate a trained sentiment analysis model on test examples
    
    Args:
        config (dict): Configuration dictionary
        model_dir (str, optional): Path to model directory
        detailed (bool): Whether to print detailed examples
        test_file (str, optional): Override the test examples file path
        
    Returns:
        dict: Evaluation metrics
    """
    # Use provided model_dir or default from config
    model_path = model_dir if model_dir else config["model"]["save_dir"]
    
    print(f"Loading model from {model_path}...")
    classifier = SentimentClassifier(model_path)
    
    # Load test examples from specified file or config
    test_file_path = test_file if test_file else config["testing"]["test_examples_file"]
    print(f"Loading test examples from {test_file_path}...")
    texts, labels = load_test_examples(test_file_path)
    print(f"Loaded {len(texts)} test examples")
    
    # Evaluate model
    print("Evaluating model...")
    metrics = classifier.evaluate(texts, labels)
    
    # Print results
    print("\n======= MODEL EVALUATION RESULTS =======")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"High-Confidence Accuracy: {metrics['high_confidence_accuracy']:.4f} " 
          f"({metrics['high_confidence_examples']}/{metrics['total_examples']} examples)")
    print(f"Average Inference Time: {metrics['avg_inference_time_ms']:.2f} ms/example")
    print("========================================\n")
    
    # Print detailed examples if requested
    if detailed:
        print("Detailed Examples (first 10):")
        for i in range(min(10, len(texts))):
            prediction, confidence, _ = classifier.predict(texts[i], return_confidence=True)
            correct = prediction == labels[i]
            sentiment = "Positive" if prediction == 1 else "Negative"
            status = "✓" if correct else "✗"
            print(f"{status} [{confidence:.4f}] {sentiment}: {texts[i][:100]}...")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment analysis model")
    parser.add_argument("--config", type=str, default="config/sentiment_analysis.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to model directory (overrides config)")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed examples")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Path to test examples file (overrides config)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Evaluate model
    evaluate_model(config, args.model_dir, args.detailed, args.test_file)

if __name__ == "__main__":
    main()
