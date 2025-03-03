import os
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_name, num_labels=2):
    """
    Load a pre-trained model and tokenizer for text classification.
    
    Args:
        model_name (str): Name or path of the pre-trained model
        num_labels (int): Number of classification labels
        
    Returns:
        tuple: (model, tokenizer) loaded from the specified source
    """
    # Load the model with the proper number of labels for classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    
    # Load the tokenizer associated with the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

def load_trained_model(model_dir):
    """
    Load a fine-tuned model and tokenizer for inference
    
    Args:
        model_dir (str): Path to the directory containing the saved model
        
    Returns:
        tuple: model and tokenizer
    """
    # Check if the model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} not found")
    
    # Load model info if available
    model_info_path = os.path.join(model_dir, "model_info.json")
    if os.path.exists(model_info_path):
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
        num_labels = model_info.get("num_labels", 2)
    else:
        num_labels = 2
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer
