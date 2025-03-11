import os

def get_enhanced_prompt(ingredients: str) -> str:
    return f"""<|system|>You are a recipe assistant. Create detailed recipes using the provided ingredients with exact measurements and clear instructions.<|endoftext|>
<|user|>Create a recipe using these ingredients: {ingredients}<|endoftext|>
<|assistant|>Title: Spicy Garlic Chicken Stir-Fry

Ingredients:
- 2 chicken breasts, sliced
- 2 cups jasmine rice
- 4 cloves garlic, minced
- 2 bell peppers, julienned
- 2 tablespoons vegetable oil
- 1 tablespoon soy sauce
- Salt and pepper to taste

Instructions:
1. Cook rice according to package instructions
2. Heat oil in a large wok over medium-high heat
3. Add minced garlic and stir-fry for 30 seconds
4. Add chicken and cook until golden, about 5 minutes
5. Add bell peppers and cook for 3 minutes
6. Add soy sauce and seasonings
7. Stir-fry for 2 more minutes
8. Serve hot over rice

Serves: 4
Cooking time: 25 minutes<|endoftext|>"""

def get_simple_prompt(ingredients):
    """Get a simple prompt for recipe generation"""
    return f"Create a recipe using these ingredients: {ingredients}\n\n"
