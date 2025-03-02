import os
import json
import pytest
import numpy as np
from src.utils.config_utils import load_config
from src.model.sentiment_inference import SentimentClassifier

# Load configuration
@pytest.fixture
def config():
    return load_config("config/sentiment_analysis.yaml")

# Sample test texts
@pytest.fixture
def test_samples():
    return [
        # Positive samples
        {"text": "This movie was amazing! I loved every minute of it.", "label": 1},
        {"text": "Great performances by all the actors. Highly recommended!", "label": 1},
        {"text": "One of the best films I've seen this year.", "label": 1},
        
        # Negative samples
        {"text": "I hated this film. Complete waste of time and money.", "label": 0},
        {"text": "Terrible acting and a confusing plot. Avoid at all costs.", "label": 0},
        {"text": "So boring I fell asleep halfway through.", "label": 0},
        
        # Ambiguous/challenging samples
        {"text": "It was okay, but I expected more from the director.", "label": 0},
        {"text": "Great visuals but the story was lacking.", "label": 0}
    ]

class TestModelLoading:
    """Tests for model loading and basic inference"""
    
    def test_model_loading(self, config):
        """Test that the model can be loaded from a saved directory"""
        model_dir = config["model"]["save_dir"]
        
        # Skip if model doesn't exist yet
        if not os.path.exists(model_dir):
            pytest.skip("Model not yet trained, skipping test")
        
        # Check that model info file exists
        info_path = os.path.join(model_dir, "model_info.json")
        assert os.path.exists(info_path), "Model info file not found"
        
        # Load model info
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        
        # Check that model info contains expected keys
        assert "model_name" in model_info
        assert "num_labels" in model_info
        assert "max_length" in model_info
        
        # Try loading the classifier
        try:
            classifier = SentimentClassifier(model_dir)
            assert classifier is not None
        except Exception as e:
            pytest.fail(f"Failed to load model: {str(e)}")


class TestModelPrediction:
    """Tests for model prediction functionality"""
    
    @pytest.fixture
    def classifier(self, config):
        """Load classifier for testing"""
        model_dir = config["model"]["save_dir"]
        
        # Skip if model doesn't exist yet
        if not os.path.exists(model_dir):
            pytest.skip("Model not yet trained, skipping test")
        
        return SentimentClassifier(model_dir)
    
    def test_single_prediction(self, classifier):
        """Test prediction for a single text sample"""
        # Positive example
        pos_text = "This movie was fantastic, I really enjoyed it!"
        pos_pred = classifier.predict(pos_text)
        assert isinstance(pos_pred, int), "Prediction should be an integer"
        
        # Negative example
        neg_text = "Terrible film, complete waste of time."
        neg_pred = classifier.predict(neg_text)
        assert isinstance(neg_pred, int), "Prediction should be an integer"
    
    def test_prediction_with_confidence(self, classifier):
        """Test prediction with confidence scores"""
        text = "This was a great movie!"
        pred, conf, time = classifier.predict(text, return_confidence=True)
        
        assert isinstance(pred, int), "Prediction should be an integer"
        assert isinstance(conf, float), "Confidence should be a float"
        assert 0 <= conf <= 1, "Confidence should be between 0 and 1"
        assert isinstance(time, float), "Inference time should be a float"
    
    def test_batch_prediction(self, classifier, test_samples):
        """Test batch prediction functionality"""
        texts = [sample["text"] for sample in test_samples]
        predictions, confidences, inference_time = classifier.predict_batch(texts)
        
        assert len(predictions) == len(texts), "Should return prediction for each input"
        assert len(confidences) == len(texts), "Should return confidence for each input"
        assert all(isinstance(p, int) for p in predictions), "All predictions should be integers"
        assert all(0 <= c <= 1 for c in confidences), "All confidences should be between 0 and 1"
        assert isinstance(inference_time, float), "Inference time should be a float"


class TestModelPerformance:
    """Tests for model performance evaluation"""
    
    @pytest.fixture
    def classifier(self, config):
        """Load classifier for testing"""
        model_dir = config["model"]["save_dir"]
        
        # Skip if model doesn't exist yet
        if not os.path.exists(model_dir):
            pytest.skip("Model not yet trained, skipping test")
        
        return SentimentClassifier(model_dir)
    
    def test_evaluation_metrics(self, classifier, test_samples):
        """Test evaluation metrics calculation"""
        texts = [sample["text"] for sample in test_samples]
        labels = [sample["label"] for sample in test_samples]
        
        metrics = classifier.evaluate(texts, labels)
        
        # Check that all expected metrics are present
        expected_metrics = [
            "accuracy", "precision", "recall", "f1", 
            "high_confidence_accuracy", "avg_inference_time_ms",
            "total_examples", "high_confidence_examples"
        ]
        for metric in expected_metrics:
            assert metric in metrics, f"Metric {metric} missing from evaluation results"
        
        # Check that metric values are valid
        assert 0 <= metrics["accuracy"] <= 1, "Accuracy should be between 0 and 1"
        assert 0 <= metrics["precision"] <= 1, "Precision should be between 0 and 1"
        assert 0 <= metrics["recall"] <= 1, "Recall should be between 0 and 1"
        assert 0 <= metrics["f1"] <= 1, "F1 should be between 0 and 1"
        assert metrics["total_examples"] == len(texts), "Total examples should match input size"
    
    def test_performance_thresholds(self, classifier, test_samples, config):
        """Test that performance meets minimum requirements"""
        # Skip this test in CI/CD environments or set it to always pass
        # as the model will need to be trained first
        if not os.path.exists(config["model"]["save_dir"]):
            pytest.skip("Model not yet trained, skipping performance test")
        
        texts = [sample["text"] for sample in test_samples]
        labels = [sample["label"] for sample in test_samples]
        
        metrics = classifier.evaluate(texts, labels)
        
        # These thresholds are intentionally low for initial testing
        # and should be adjusted based on actual model performance
        # Note: These may fail until the model is properly trained
        assert metrics["accuracy"] >= 0.5, "Accuracy below minimum threshold"
        assert metrics["f1"] >= 0.4, "F1 score below minimum threshold"
        
        # Check performance requirements from config
        perf_config = config["testing"]["performance_test"]
        assert metrics["avg_inference_time_ms"] <= perf_config["latency_threshold_ms"], \
            f"Inference time exceeds threshold of {perf_config['latency_threshold_ms']}ms"


if __name__ == "__main__":
    pytest.main() 