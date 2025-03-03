import os
import json
import argparse
from datasets import load_dataset
from src.utils.config_utils import load_config

def create_test_set(config, num_examples=100):
    """
    Create a balanced test set with equal numbers of positive and negative examples
    for consistent evaluation and testing.
    
    Args:
        config (dict): Configuration dictionary with data parameters
        num_examples (int): Total number of examples to extract (half positive, half negative)
        
    Returns:
        list: List of test examples with text and expected labels (balanced between positive and negative)
    """
    try:
        # Load dataset
        dataset = load_dataset(
            config["data"]["dataset_name"],
            cache_dir=config["data"]["cache_dir"]
        )
        
        # Get validation/test split
        test_data = dataset[config["data"]["validation_split"]]
        
        # Separate positive and negative examples
        positive_examples = [i for i, item in enumerate(test_data) if item["label"] == 1]
        negative_examples = [i for i, item in enumerate(test_data) if item["label"] == 0]
        
        print(f"Found {len(positive_examples)} positive and {len(negative_examples)} negative examples")
        
        # Ensure we have enough examples of each class
        half_examples = num_examples // 2
        if len(positive_examples) < half_examples or len(negative_examples) < half_examples:
            print(f"Warning: Not enough examples of each class. Using max available: "
                  f"{min(len(positive_examples), len(negative_examples))} per class")
            half_examples = min(len(positive_examples), len(negative_examples))
        
        # Select balanced samples
        selected_positives = positive_examples[:half_examples]
        selected_negatives = negative_examples[:half_examples]
        
        # Combine indices and get corresponding examples
        selected_indices = selected_positives + selected_negatives
        balanced_subset = test_data.select(selected_indices)
        
        # Format as a list of dictionaries
        test_examples = [
            {"text": example["text"], "label": example["label"]}
            for example in balanced_subset
        ]
        
        return test_examples
        
    except Exception as e:
        print(f"Error creating balanced test examples: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Create balanced test examples for model evaluation")
    parser.add_argument("--config", type=str, default="config/sentiment_analysis.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--num_examples", type=int, default=100,
                        help="Number of test examples to create (half positive, half negative)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (overrides config)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create balanced test examples
    test_examples = create_test_set(config, num_examples=args.num_examples)
    
    # Determine output path
    output_path = args.output if args.output else config["testing"]["test_examples_file"]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save test examples to file
    with open(output_path, 'w') as f:
        json.dump(test_examples, f, indent=2)
    
    print(f"Created balanced test set with {len(test_examples)} examples ({len(test_examples)//2} positive, {len(test_examples)//2} negative)")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
