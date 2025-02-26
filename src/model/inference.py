import os
import json
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class SentimentClassifier:
    """Class for sentiment classification inference and evaluation"""
    
    def __init__(self, model_dir):
        """
        Initialize the classifier with a fine-tuned model.
        
        Args:
            model_dir (str): Path to the directory containing the saved model
        """
        # Load model info
        with open(os.path.join(model_dir, "model_info.json"), "r") as f:
            self.model_info = json.load(f)
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def predict(self, text, return_confidence=False):
        """
        Predict sentiment for a single text input.
        
        Args:
            text (str): Text to classify
            return_confidence (bool): Whether to return confidence score
            
        Returns:
            int or tuple: Predicted label (0=negative, 1=positive) and optionally confidence score
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.model_info["max_length"],
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(**inputs)
            inference_time = time.time() - start_time
        
        # Get predicted class and confidence
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
        if return_confidence:
            return prediction, confidence, inference_time
        return prediction
    
    def predict_batch(self, texts):
        """
        Predict sentiment for a batch of text inputs.
        
        Args:
            texts (list): List of text strings to classify
            
        Returns:
            list: List of predicted labels and confidences
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.model_info["max_length"],
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(**inputs)
            inference_time = time.time() - start_time
        
        # Get predicted classes and confidences
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1).tolist()
        confidences = [probabilities[i][predictions[i]].item() for i in range(len(predictions))]
        
        return predictions, confidences, inference_time
    
    def evaluate(self, texts, labels):
        """
        Evaluate model performance on a test set.
        
        Args:
            texts (list): List of text strings to classify
            labels (list): List of ground truth labels
            
        Returns:
            dict: Dictionary with performance metrics
        """
        # Get predictions
        predictions, confidences, inference_time = self.predict_batch(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        
        # Calculate average inference time
        avg_inference_time = inference_time / len(texts) * 1000  # in milliseconds
        
        # Calculate high confidence metrics
        high_conf_indices = [i for i, conf in enumerate(confidences) if conf >= 0.7]
        if high_conf_indices:
            high_conf_preds = [predictions[i] for i in high_conf_indices]
            high_conf_labels = [labels[i] for i in high_conf_indices]
            high_conf_accuracy = accuracy_score(high_conf_labels, high_conf_preds)
        else:
            high_conf_accuracy = 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "high_confidence_accuracy": high_conf_accuracy,
            "avg_inference_time_ms": avg_inference_time,
            "total_examples": len(texts),
            "high_confidence_examples": len(high_conf_indices)
        }


def load_classifier(model_dir):
    """
    Helper function to load a trained classifier.
    
    Args:
        model_dir (str): Path to the directory containing the saved model
        
    Returns:
        SentimentClassifier: Loaded classifier
    """
    return SentimentClassifier(model_dir)
