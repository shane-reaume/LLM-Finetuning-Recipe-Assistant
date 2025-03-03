import os
import json
import argparse
from datasets import load_dataset
from src.utils.config_utils import load_config

def prepare_generation_dataset(config):
    """
    Prepare dataset for text generation fine-tuning
    
    Args:
        config (dict): Configuration dictionary with data parameters
        
    Returns:
        tuple: processed train and validation datasets ready for training
    """
    dataset = load_dataset(
        config["data"]["dataset_name"],
        cache_dir=config["data"]["cache_dir"]
    )
    
    # Define data processing function
    def format_for_generation(examples):
        prompts = []
        responses = []
        
        for example in examples:
            # Format prompt using the template from config
            ingredients = example.get("ingredients", "")
            prompt = config["data"]["prompt_template"].format(ingredients=ingredients)
            prompts.append(prompt)
            
            # Get response
            response = example.get(config["data"]["response_key"], "")
            responses.append(response)
        
        return {"prompt": prompts, "response": responses}
    
    # Apply processing
    train_data = dataset[config["data"]["train_split"]].map(
        format_for_generation,
        batched=True,
        remove_columns=dataset[config["data"]["train_split"]].column_names
    )
    
    val_data = dataset[config["data"]["validation_split"]].map(
        format_for_generation,
        batched=True,
        remove_columns=dataset[config["data"]["validation_split"]].column_names
    )
    
    return train_data, val_data

def create_test_examples(config, num_examples=10):
    """
    Create a small set of test examples for evaluation
    
    Args:
        config (dict): Configuration dictionary with data parameters
        num_examples (int): Number of examples to extract
        
    Returns:
        list: List of test examples with prompts and expected responses
    """
    try:
        dataset = load_dataset(
            config["data"]["dataset_name"],
            cache_dir=config["data"]["cache_dir"]
        )
        
        # Take a subset of the validation set for testing
        test_subset = dataset[config["data"]["validation_split"]].select(range(num_examples))
        
        # Format examples
        test_examples = []
        for example in test_subset:
            ingredients = example.get("ingredients", "")
            prompt = config["data"]["prompt_template"].format(ingredients=ingredients)
            response = example.get(config["data"]["response_key"], "")
            
            test_examples.append({
                "prompt": prompt,
                "response": response
            })
        
        return test_examples
        
    except Exception as e:
        print(f"Error creating generation test examples: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for text generation")
    parser.add_argument("--config", type=str, default="config/text_generation.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--num_examples", type=int, default=10,
                        help="Number of test examples to create")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create test examples
    test_examples = create_test_examples(config, args.num_examples)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(config["testing"]["test_examples_file"]), exist_ok=True)
    
    # Save test examples
    with open(config["testing"]["test_examples_file"], "w") as f:
        json.dump(test_examples, f, indent=2)
    
    print(f"Created {len(test_examples)} test examples for generation")
    print(f"Saved to {config['testing']['test_examples_file']}")

if __name__ == "__main__":
    main()
