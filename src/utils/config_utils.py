import os
import yaml
import logging

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Ensure numeric values are properly typed
    if "training" in config:
        if "learning_rate" in config["training"]:
            config["training"]["learning_rate"] = float(config["training"]["learning_rate"])
        if "weight_decay" in config["training"]:
            config["training"]["weight_decay"] = float(config["training"]["weight_decay"])
        if "num_train_epochs" in config["training"]:
            config["training"]["num_train_epochs"] = float(config["training"]["num_train_epochs"])
    
    return config
