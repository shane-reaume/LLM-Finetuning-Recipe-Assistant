import os
import sys
import json
import argparse
from src.model.inference import SentimentClassifier
from src.utils.config_utils import load_config

def print_colored(text, sentiment, confidence):
    """Print colored text based on sentiment prediction"""
    if sentiment == 1:  # Positive
        color = '\033[92m'  # Green
        label = "POSITIVE"
    else:  # Negative
        color = '\033[91m'  # Red
        label = "NEGATIVE"
    
    reset = '\033[0m'
    confidence_pct = f"{confidence * 100:.1f}%"
    
    print(f"{text}")
    print(f"{color}→ {label} ({confidence_pct} confidence){reset}\n")

def interactive_demo(classifier):
    """Run an interactive demo allowing user input"""
    print("\n===== Sentiment Analysis Demo =====")
    print("Enter text to analyze sentiment (type 'exit' to quit):")
    
    while True:
        text = input("\nYour text: ")
        if text.lower() in ['exit', 'quit', 'q']:
            break
        
        # Get prediction
        prediction, confidence, _ = classifier.predict(text, return_confidence=True)
        print_colored(text, prediction, confidence)

def batch_demo(classifier, examples_file=None):
    """Run a batch demo using pre-defined examples"""
    
    if examples_file and os.path.exists(examples_file):
        # Load examples from file
        with open(examples_file, 'r') as f:
            examples = json.load(f)
        
        # Select a few examples to show
        examples = examples[:5]
        texts = [ex["text"] for ex in examples]
    else:
        # Use built-in examples if no file is provided
        texts = [
            "This movie was absolutely fantastic! The acting was superb.",
            "Complete waste of time and money. Terrible plot and acting.",
            "It was okay, not great but not terrible either.",
            "I've never been so bored in my life. Avoid at all costs.",
            "The visuals were stunning but the story was lacking."
        ]
    
    print("\n===== Sentiment Analysis Batch Demo =====")
    
    # Get batch predictions
    predictions, confidences, inference_time = classifier.predict_batch(texts)
    
    # Print results
    for i, (text, pred, conf) in enumerate(zip(texts, predictions, confidences)):
        print(f"\nExample {i+1}:")
        print_colored(text, pred, conf)
    
    # Print timing info
    avg_time = inference_time / len(texts) * 1000
    print(f"Average inference time: {avg_time:.2f} ms per example")

def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Demo")
    parser.add_argument('--config', type=str, default='config/sentiment_analysis.yaml',
                        help='Path to configuration file')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--examples', type=str, default=None,
                        help='Path to examples file (default: use built-in examples)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    model_dir = config["model"]["save_dir"]
    
    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"Error: Model not found at {model_dir}")
        print("Please train the model first by running: python -m src.model.train")
        sys.exit(1)
    
    # Load classifier
    print(f"Loading model from {model_dir}...")
    classifier = SentimentClassifier(model_dir)
    print("Model loaded!")
    
    # Run demo
    if args.interactive:
        interactive_demo(classifier)
    else:
        examples_file = args.examples or config["testing"].get("test_examples_file")
        batch_demo(classifier, examples_file)

if __name__ == "__main__":
    main() 