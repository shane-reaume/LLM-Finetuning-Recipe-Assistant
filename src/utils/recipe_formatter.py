import re

def extract_recipe_sections(raw_text):
    """Extract recipe sections from raw text"""
    
    # Look for common section headers
    recipe_name = ""
    ingredients = []
    instructions = []
    cooking_time = ""
    servings = ""
    
    # Try to extract recipe name (usually the first line)
    lines = raw_text.strip().split('\n')
    if lines:
        recipe_name = lines[0].strip()
        # Remove common prefixes
        recipe_name = re.sub(r'^(recipe(\s+)?name:?\s*)', '', recipe_name, flags=re.IGNORECASE).strip()
        
    # Find ingredients section
    ingredients_section = re.search(
        r'(?:ingredients:?)(.*?)(?:(?:instructions|directions|steps|preparation):?|$)', 
        raw_text, re.IGNORECASE | re.DOTALL
    )
    
    if ingredients_section:
        ingredient_text = ingredients_section.group(1).strip()
        # Extract individual ingredients
        for line in ingredient_text.split('\n'):
            line = line.strip()
            if line and not re.match(r'^ingredients:?$', line, re.IGNORECASE):
                # Remove bullet points and dashes
                item = re.sub(r'^[-•*]\s*', '', line)
                ingredients.append(item)
    
    # Find instructions section
    instructions_section = re.search(
        r'(?:instructions|directions|steps|preparation):?(.*?)(?:(?:notes|time|servings):?|$)',
        raw_text, re.IGNORECASE | re.DOTALL
    )
    
    if instructions_section:
        instruction_text = instructions_section.group(1).strip()
        # Extract individual steps
        for line in instruction_text.split('\n'):
            line = line.strip()
            if line and not re.match(r'^(instructions|directions|steps|preparation):?$', line, re.IGNORECASE):
                # Remove step numbers
                step = re.sub(r'^[0-9]+[.)\s]*', '', line).strip()
                if step:
                    instructions.append(step)
    
    # If no clear sections, try to infer structure
    if not ingredients and not instructions:
        in_ingredients = False
        in_instructions = False
        
        for line in lines[1:]:  # Skip the first line (recipe name)
            line = line.strip()
            if not line:
                continue
                
            # Check for section markers
            if re.match(r'ingredients:?', line, re.IGNORECASE):
                in_ingredients = True
                in_instructions = False
                continue
            elif re.match(r'(instructions|directions|steps|preparation):?', line, re.IGNORECASE):
                in_ingredients = False
                in_instructions = True
                continue
            
            # Add to appropriate section
            if in_ingredients:
                ingredients.append(line)
            elif in_instructions:
                instructions.append(line)
            elif ':' in line and len(ingredients) == 0:  # Probably an ingredient
                ingredients.append(line)
            elif len(ingredients) > 0 and len(instructions) == 0:  # Probably an instruction
                instructions.append(line)
    
    # Extract cooking time if present
    time_match = re.search(r'(?:cooking|preparation|total)\s+time:?\s*([^.]+)', raw_text, re.IGNORECASE)
    if time_match:
        cooking_time = time_match.group(1).strip()
    
    # Extract servings if present
    servings_match = re.search(r'(?:serves|servings|yield):?\s*([^.]+)', raw_text, re.IGNORECASE)
    if servings_match:
        servings = servings_match.group(1).strip()
    
    return {
        "name": recipe_name,
        "ingredients": ingredients,
        "instructions": instructions,
        "cooking_time": cooking_time,
        "servings": servings,
        "raw_text": raw_text
    }

def format_recipe_as_markdown(recipe_data):
    """Format recipe data as markdown"""
    markdown = ""
    
    # Recipe name
    if recipe_data["name"]:
        markdown += f"# {recipe_data['name']}\n\n"
    
    # Metadata
    metadata = []
    if recipe_data["cooking_time"]:
        metadata.append(f"⏱️ **Time:** {recipe_data['cooking_time']}")
    if recipe_data["servings"]:
        metadata.append(f"👥 **Servings:** {recipe_data['servings']}")
        
    if metadata:
        markdown += " | ".join(metadata) + "\n\n"
    
    # Ingredients
    if recipe_data["ingredients"]:
        markdown += "## Ingredients\n\n"
        for ingredient in recipe_data["ingredients"]:
            markdown += f"- {ingredient}\n"
        markdown += "\n"
    
    # Instructions
    if recipe_data["instructions"]:
        markdown += "## Instructions\n\n"
        for i, step in enumerate(recipe_data["instructions"]):
            markdown += f"{i+1}. {step}\n"
            
    return markdown
