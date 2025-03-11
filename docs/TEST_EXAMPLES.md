# Test Examples for Recipe Generation

This document explains the purpose and format of test examples used for the Recipe Assistant model.

## What Are Test Examples?

Test examples are predefined pairs of inputs (ingredients) and expected outputs (recipes) that help:

1. **Evaluate model quality** - By comparing generated recipes against known good examples
2. **Ensure consistency** - By providing a fixed set of test cases
3. **Validate format adherence** - By demonstrating the correct structure for recipes

## Location of Test Examples

Test examples are stored in the JSON file:
```
data/processed/recipe_test_examples.json
```

This file contains multiple examples, each with:
- A set of ingredients
- A complete recipe in the correct format

## Example Format

Each example follows the exact format used for training the model:

```json
{
  "examples": [
    {
      "ingredients": "chicken breast, rice, garlic, bell peppers",
      "recipe": "<|system|>You are a recipe assistant. Create detailed recipes with exact measurements and clear instructions.<|endoftext|>\n<|user|>Write a complete recipe using these ingredients: chicken breast, rice, garlic, bell peppers. Include a title, ingredients list with measurements, and numbered cooking steps.<|endoftext|>\n<|assistant|>Title: Garlic Chicken and Bell Pepper Rice Bowl\n\nIngredients:\n- 2 chicken breasts, diced\n- 2 cups jasmine rice\n- 4 cloves garlic, minced\n- 2 bell peppers, sliced\n- 2 tablespoons olive oil\n- 1 cup chicken broth\n- 1 teaspoon paprika\n- Salt and pepper to taste\n- 1 tablespoon fresh parsley, chopped (optional)\n\nInstructions:\n1. Rinse the rice under cold water until the water runs clear.\n2. Cook the rice according to package instructions, then set aside.\n3. Heat 1 tablespoon of olive oil in a large skillet over medium heat.\n4. Add the minced garlic and cook until fragrant, about 1 minute.\n5. Add the diced chicken and cook until golden brown, about 5-7 minutes.\n6. Add the sliced bell peppers and cook for another 3-4 minutes until tender-crisp.\n7. Pour in the chicken broth and bring to a simmer.\n8. Season with paprika, salt, and pepper.\n9. Reduce heat and cook for an additional 5 minutes until sauce thickens slightly.\n10. Serve the chicken and bell pepper mixture over the cooked rice.\n11. Garnish with fresh parsley if desired.\n\nServes 4. Cooking time: 30 minutes."
    }
  ]
}
```

## Importance in Testing

The test examples are used by various components of the project:

1. The test script can use these examples to verify the model is working correctly
2. The quality checker evaluates the model's outputs against these examples
3. Developers can use these as reference when creating new interfaces

## Creating New Test Examples

When creating new test examples, ensure they:

1. Follow the **exact same format** as existing examples
2. Include all special tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|endoftext|>`)
3. Contain complete recipes with title, ingredients list, and numbered instructions
4. Use a diverse set of ingredients representing different cuisines and dietary preferences

## Using Test Examples

The test examples are referenced in the configuration file:

```yaml
# Testing settings
testing:
  test_examples_file: "data/processed/recipe_test_examples.json"
```

This allows various scripts to load and use these examples for testing and evaluation.

## Relationship to Prompt Format

The test examples demonstrate the correct prompt format in context. Each example shows:

1. How system messages should be formatted
2. How user queries should be structured
3. How assistant responses should be formatted

For more information on general prompt formatting, see the [Prompt Format Guide](PROMPT_FORMAT.md). 