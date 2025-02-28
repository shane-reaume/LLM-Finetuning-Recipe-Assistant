import os
import json
import argparse
from src.utils.config_utils import load_config
from src.data.dataset import create_test_examples

def main():
    """
    Create a test set with examples from the dataset and save it to a file
    for consistent evaluation and testing.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create test examples for model evaluation")
    parser.add_argument("--config", type=str, default="config/sentiment_analysis.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--num_examples", type=int, default=100,
                        help="Number of test examples to create")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Ensure the testing section exists
    if "testing" not in config:
        config["testing"] = {}
    if "test_examples_file" not in config["testing"]:
        config["testing"]["test_examples_file"] = "data/processed/test_examples.json"
    
    # Create test examples
    test_examples = create_test_examples(config, num_examples=args.num_examples)
    
    # Add robust error checking
    if test_examples is None:
        test_examples = []
        print("Warning: test_examples is None, using empty list instead")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(config["testing"]["test_examples_file"])
    os.makedirs(output_dir, exist_ok=True)
    
    # Save test examples to file
    with open(config["testing"]["test_examples_file"], "w") as f:
        json.dump(test_examples, f, indent=2)
    
    print(f"Created {len(test_examples)} test examples and saved to {config['testing']['test_examples_file']}")

if __name__ == "__main__":
    main()