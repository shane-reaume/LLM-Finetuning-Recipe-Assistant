import os
import json
import argparse
from datasets import load_dataset, Dataset, DatasetDict
from src.utils.config_utils import load_config

def prepare_generation_dataset(config, data_dir=None):
    """
    Prepare dataset for text generation fine-tuning
    
    Args:
        config (dict): Configuration dictionary with data parameters
        data_dir (str): Path to directory with manually downloaded dataset
        
    Returns:
        tuple: processed train and validation datasets ready for training
    """
    print("Loading recipe dataset...")
    try:
        # If data_dir is provided, use it to load the manual dataset
        if data_dir:
            dataset = load_dataset(
                config["data"]["dataset_name"],
                data_dir=data_dir,
                trust_remote_code=True,
                cache_dir=config["data"]["cache_dir"]
            )
        else:
            # Default loading without data_dir
            dataset = load_dataset(
                config["data"]["dataset_name"],
                trust_remote_code=True,
                cache_dir=config["data"]["cache_dir"]
            )
        
        print(f"Dataset loaded successfully. Available splits: {dataset.keys()}")
        
        # Debug: print an example to understand the structure
        print("Dataset example structure:")
        for key, value in dataset["train"][0].items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, str) and len(value) > 50:
                print(f"    Value (truncated): {value[:50]}...")
            else:
                print(f"    Value: {value}")
        
        # Create a validation split if it doesn't exist
        train_split = config["data"].get("train_split", "train")
        val_split = config["data"].get("validation_split", "validation")
        
        if val_split not in dataset:
            print(f"Validation split '{val_split}' not found. Creating from {train_split} split...")
            
            # Use a fixed seed for reproducibility
            train_val_split = dataset[train_split].train_test_split(
                test_size=0.1, seed=42
            )
            
            # Create a new dataset with train and validation splits
            dataset = DatasetDict({
                train_split: train_val_split["train"],
                "validation": train_val_split["test"]
            })
            
            print(f"Created validation split with {len(dataset['validation'])} examples")
        
        # Define data processing function
        def format_for_generation(examples):
            prompts = []
            responses = []
            
            # Get the batch size from the first key's length
            batch_size = len(examples[list(examples.keys())[0]])
            
            for i in range(batch_size):
                try:
                    # Handle different field structures based on the dataset schema
                    if "ingredients" in examples:
                        ingredients_value = examples["ingredients"][i]
                        # If ingredients is a list, join it with commas
                        if isinstance(ingredients_value, list):
                            ingredients = ", ".join(ingredients_value)
                        else:
                            ingredients = str(ingredients_value)
                    else:
                        # Fallback if ingredients field isn't found
                        ingredients = "unknown ingredients"
                    
                    # Format the prompt
                    prompt = config["data"]["prompt_template"].format(ingredients=ingredients)
                    prompts.append(prompt)
                    
                    # Get response based on the configured key
                    response_key = config["data"]["response_key"]
                    if response_key in examples:
                        response = examples[response_key][i]
                    else:
                        # Fallback for other field structures
                        response = examples.get("directions", [""])[i]
                    
                    responses.append(response)
                    
                except Exception as e:
                    print(f"Error processing example {i}: {e}")
                    # Add a placeholder to keep batch sizes aligned
                    prompts.append(config["data"]["prompt_template"].format(ingredients="error"))
                    responses.append("")
            
            return {"prompt": prompts, "response": responses}
        
        # Apply processing
        print(f"Processing {train_split} split...")
        train_data = dataset[train_split].map(
            format_for_generation,
            batched=True,
            remove_columns=dataset[train_split].column_names
        )
        
        print(f"Processing validation split...")
        val_data = dataset["validation"].map(  # Always use "validation" as we've guaranteed it exists
            format_for_generation,
            batched=True,
            remove_columns=dataset["validation"].column_names
        )
        
        return train_data, val_data
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        print("\nMANUAL DOWNLOAD INSTRUCTIONS:")
        print("1. Go to https://recipenlg.cs.put.poznan.pl/")
        print("2. Download the dataset zip file")
        print("3. Unzip the file and locate the full_dataset.csv file")
        print("4. Create a directory to store the file (e.g., ~/manual_data)")
        print("5. Move full_dataset.csv to your created directory")
        print("6. Run this script again with --data_dir pointing to your directory:")
        print("   python -m src.data.recipe_prepare_dataset --data_dir ~/manual_data")
        return None, None

def create_test_examples(config, data_dir=None, num_examples=10):
    """
    Create a small set of test examples for evaluation
    
    Args:
        config (dict): Configuration dictionary with data parameters
        data_dir (str): Path to directory with manually downloaded dataset
        num_examples (int): Number of examples to extract
        
    Returns:
        list: List of test examples with prompts and expected responses
    """
    try:
        # Try to load the dataset
        if data_dir:
            dataset = load_dataset(
                config["data"]["dataset_name"],
                data_dir=data_dir,
                trust_remote_code=True,
                cache_dir=config["data"]["cache_dir"]
            )
        else:
            dataset = load_dataset(
                config["data"]["dataset_name"],
                trust_remote_code=True,
                cache_dir=config["data"]["cache_dir"]
            )
            
        # Check if validation split exists, if not create it
        val_split = config["data"].get("validation_split", "validation")
        
        if val_split not in dataset:
            print(f"Validation split '{val_split}' not found. Creating from train split...")
            
            train_split = config["data"].get("train_split", "train")
            # Use a fixed seed for reproducibility
            train_val_split = dataset[train_split].train_test_split(
                test_size=0.1, seed=42
            )
            
            # Use the test portion as our validation set
            validation_dataset = train_val_split["test"]
        else:
            validation_dataset = dataset[val_split]
        
        # Take a subset of the validation set for testing
        test_subset = validation_dataset.select(range(min(num_examples, len(validation_dataset))))
        
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
        import traceback
        traceback.print_exc()
        print("\nMANUAL DOWNLOAD INSTRUCTIONS:")
        print("1. Go to https://recipenlg.cs.put.poznan.pl/")
        print("2. Download the dataset zip file")
        print("3. Unzip the file and locate the full_dataset.csv file")
        print("4. Create a directory to store the file (e.g., ~/manual_data)")
        print("5. Move full_dataset.csv to your created directory")
        print("6. Run this script again with --data_dir pointing to your directory")
        return []

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for text generation")
    parser.add_argument("--config", type=str, default="config/text_generation.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--num_examples", type=int, default=10,
                        help="Number of test examples to create")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to directory containing manually downloaded dataset")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create test examples
    test_examples = create_test_examples(config, args.data_dir, args.num_examples)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(config["testing"]["test_examples_file"]), exist_ok=True)
    
    # Save test examples
    with open(config["testing"]["test_examples_file"], "w") as f:
        json.dump(test_examples, f, indent=2)
    
    print(f"Created {len(test_examples)} test examples for generation")
    print(f"Saved to {config['testing']['test_examples_file']}")

if __name__ == "__main__":
    main()
