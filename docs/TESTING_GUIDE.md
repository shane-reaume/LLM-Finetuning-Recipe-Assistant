# Testing Guide for Recipe Generation Model

This document outlines the proper methodology for testing the Recipe Assistant model to ensure it's functioning correctly and producing high-quality outputs.

## Available Testing Tools

The project includes several tools for testing the recipe model:

1. **test-recipe-3060 Makefile target**: Primary testing tool for quick evaluation
   ```bash
   make test-recipe-3060 INGREDIENTS="chicken, rice, garlic"
   ```

2. **recipe-quality-check Makefile target**: Evaluates quality metrics
   ```bash
   make recipe-quality-check INGREDIENTS="chicken, rice, garlic" MODEL_DIR="models/test_recipe_assistant_3060_geforce_rtx"
   ```

3. **src/test_3060_recipe_model.py**: Direct script for detailed testing and debugging
   ```bash
   python -m src.test_3060_recipe_model --ingredients="chicken, rice, garlic, bell peppers"
   ```

4. **Web UI Testing**: Browser-based interface for interactive testing
   ```bash
   python -m src.recipe_web_demo
   ```

## Testing Best Practices

### 1. Use Consistent Prompt Format

Always ensure you're using the correct prompt format:

```
<|system|>You are a recipe assistant. Create detailed recipes with exact measurements and clear instructions.<|endoftext|>
<|user|>Write a complete recipe using these ingredients: {ingredients}. Include a title, ingredients list with measurements, and numbered cooking steps.<|endoftext|>
<|assistant|>
```

### 2. Test with Diverse Ingredients

Test the model with a variety of ingredient combinations:
- Simple (2-3 ingredients)
- Complex (5+ ingredients)
- Common combinations (e.g., "chicken, rice, vegetables")
- Unusual combinations (e.g., "banana, coffee, cardamom")
- Different cuisines (e.g., "tofu, soy sauce, ginger" for Asian cuisine)

### 3. Check Quality Metrics

The quality checker evaluates recipes on several dimensions:
- **Ingredient Coverage**: Does the recipe use all the provided ingredients?
- **Structure Score**: Does the recipe include title, ingredients list, and instructions?
- **Step Clarity**: Are the cooking steps clear and detailed?
- **Overall Quality**: Combined score of the above metrics

Aim for an overall quality score above 0.8 for good recipes.

### 4. Assess Generation Parameters

Experiment with different generation parameters:
- **Temperature**: Higher (0.7-0.9) for creativity, lower (0.3-0.5) for consistency
- **Max Tokens**: 512 is recommended for complete recipes
- **Repetition Penalty**: Use 1.2 to prevent repetitive text

### 5. Validate Before Deployment

Before deploying a new model version:
1. Run the quality check on at least 5 different ingredient combinations
2. Verify with both the specialized test script and the quality checker
3. Ensure scores are consistent across multiple test runs

## Troubleshooting Common Issues

### Poor Quality Scores

If your model is generating low-quality recipes:

1. **Check prompt format**: Ensure it matches exactly the format used during training
2. **Verify model path**: Confirm you're pointing to the correct trained model
3. **Inspect generation parameters**: Try adjusting temperature, max tokens, etc.
4. **Check recipe structure**: Ensure recipes have a title, ingredients list, and steps

### Model Repeating the Prompt

If the model is repeating the prompt or not completing the recipe:

1. **Verify special tokens**: Make sure all special tokens are included
2. **Check max tokens**: Increase the token limit to allow longer completions
3. **Examine training configuration**: Confirm the model was trained with the correct prompt format

### Inconsistent Results

If you get inconsistent results between test runs:

1. **Set a fixed seed**: Use `--seed 42` to make generation deterministic
2. **Lower the temperature**: Reduce to 0.3-0.5 for more consistent outputs
3. **Run multiple tests**: Generate 3-5 recipes with the same ingredients to assess variance

## Integration Testing

When integrating the model into applications:

1. **Test error handling**: Ensure your application gracefully handles model errors
2. **Validate output format**: Check that you can correctly parse the generated recipes
3. **Measure performance**: Track generation time and resource usage

## Next Steps

After successful testing, consider:
1. **Exporting to Ollama**: Use `make recipe-export` for deployment
2. **Fine-tuning further**: Add more training examples focused on areas where quality is lower
3. **Creating specialized model variants**: Train domain-specific models (vegetarian, desserts, etc.)

For more information on prompt formatting, see the [Prompt Format Guide](PROMPT_FORMAT.md). 