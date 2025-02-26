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
