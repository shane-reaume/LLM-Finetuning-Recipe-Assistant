import os
import json
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.model.model_loader import load_model
from src.data.dataset import load_and_preprocess_data

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for model predictions.
    
    Args:
        eval_pred: Tuple of predictions and labels
        
    Returns:
        dict: Dictionary of metric scores
    """
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(config):
    """
    Train a text classification model using the Hugging Face Trainer.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: Trained model and tokenizer
    """
    # Load model and tokenizer
    model, tokenizer = load_model(
        config["model"]["name"], 
        num_labels=2  # Binary classification for sentiment
    )
    
    # Load and preprocess datasets
    train_dataset, eval_dataset = load_and_preprocess_data(config)
    
    # Set up training arguments from config
    training_config = config["training"]
    training_args = TrainingArguments(
        output_dir=config["model"]["save_dir"],
        num_train_epochs=float(training_config["num_train_epochs"]),
        per_device_train_batch_size=int(training_config["batch_size"]),
        per_device_eval_batch_size=int(training_config["batch_size"]),
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        learning_rate=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
        warmup_steps=int(training_config["warmup_steps"]),
        save_steps=int(training_config["save_steps"]),
        save_total_limit=int(training_config["save_total_limit"]),
        logging_dir=training_config["logging_dir"],
        evaluation_strategy=training_config["evaluation_strategy"],
        eval_steps=int(training_config["eval_steps"]),
        fp16=training_config["fp16"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model and tokenizer
    trainer.save_model(config["model"]["save_dir"])
    tokenizer.save_pretrained(config["model"]["save_dir"])
    
    # Save model configuration for testing
    with open(os.path.join(config["model"]["save_dir"], "model_info.json"), "w") as f:
        model_info = {
            "model_name": config["model"]["name"],
            "num_labels": 2,
            "max_length": config["data"]["max_length"]
        }
        json.dump(model_info, f)
    
    return model, tokenizer

if __name__ == "__main__":
    from src.utils.config_utils import load_config
    
    # Load configuration
    config = load_config("config/sentiment_analysis.yaml")
    
    # Create output directory if it doesn't exist
    os.makedirs(config["model"]["save_dir"], exist_ok=True)
    
    # Train the model
    train_model(config)
