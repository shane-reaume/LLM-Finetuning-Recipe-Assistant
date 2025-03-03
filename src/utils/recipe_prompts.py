import os

def get_enhanced_prompt(ingredients_list):
    """
    Get an enhanced prompt template for better recipe generation
    
    Args:
        ingredients_list (str): Comma-separated list of ingredients
        
    Returns:
        str: Formatted prompt to generate better recipes
    """
    # Try to load the prompt template file
    template_path = os.path.join("config", "recipe_prompt.txt")
    try:
        with open(template_path, "r") as f:
            template = f.read()
    except:
        # Fallback if file not found
        template = """Create a detailed recipe using these ingredients: {ingredients}

Your recipe must include:
- A creative recipe name
- Complete ingredients list with measurements
- Clear cooking instructions
- Cooking time
- Number of servings"""
    
    # Process ingredients to make them easier to work with
    ingredients = [i.strip() for i in ingredients_list.split(',')]
    
    # Format the main ingredients as a bulleted list
    ingredients_formatted = "\n• " + "\n• ".join(ingredients)
    
    # Format the full prompt - simpler format
    prompt = template.format(ingredients=ingredients_formatted)
    
    return prompt
