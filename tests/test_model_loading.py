import pytest
from src.model.model_loader import load_model

def test_load_model():
    model_name = "gpt2"  # Use a small model for testing
    model, tokenizer = load_model(model_name)
    assert model is not None
    assert tokenizer is not None
