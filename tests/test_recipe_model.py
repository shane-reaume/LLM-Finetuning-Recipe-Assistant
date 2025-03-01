import os
import sys
import pytest
import torch

from src.recipe_demo import RecipeGenerator

@pytest.fixture(scope="module")
def recipe_model_dir():
    """Return the path to the recipe model directory"""
    return os.path.join(os.getcwd(), "models", "recipe_assistant")

@pytest.fixture(scope="module")
def recipe_generator(recipe_model_dir):
    """Load the recipe generator model (if available)"""
    if not os.path.exists(os.path.join(recipe_model_dir, "config.json")):
        pytest.skip("Recipe model not found. Run training first.")
    return RecipeGenerator(recipe_model_dir)

def test_model_loading(recipe_model_dir):
    """Test that the model can be loaded successfully"""
    if not os.path.exists(os.path.join(recipe_model_dir, "config.json")):
        pytest.skip("Recipe model not found. Run training first.")
    
    generator = RecipeGenerator(recipe_model_dir)
    assert generator.model is not None
    assert generator.tokenizer is not None

def test_basic_recipe_generation(recipe_generator):
    """Test that the model can generate a recipe"""
    ingredients = "chicken, rice, broccoli"
    recipe, gen_time = recipe_generator.generate_recipe(ingredients, max_new_tokens=100)
    
    assert len(recipe) > 20, "Recipe should be reasonably long"
    assert "chicken" in recipe.lower(), "Recipe should mention the main ingredient"
    assert gen_time > 0, "Generation time should be positive"

def test_different_ingredients(recipe_generator):
    """Test that different ingredients produce different recipes"""
    ingredients1 = "beef, potatoes, carrots"
    ingredients2 = "pasta, tomatoes, basil"
    
    recipe1, _ = recipe_generator.generate_recipe(ingredients1, max_new_tokens=100)
    recipe2, _ = recipe_generator.generate_recipe(ingredients2, max_new_tokens=100)
    
    assert recipe1 != recipe2, "Different ingredients should produce different recipes"
    assert "beef" in recipe1.lower(), "Recipe should mention beef"
    assert "pasta" in recipe2.lower(), "Recipe should mention pasta"
