# Recipe Model Evaluation Guide

This guide explains how to interpret the recipe-specific evaluation metrics and provides strategies for improving your recipe generation model.

## Understanding Recipe Evaluation Metrics

We've implemented custom metrics that better reflect recipe quality than standard text generation metrics like ROUGE:

### 1. Ingredient Coverage (0.0 - 1.0)

Measures how well the model uses the requested ingredients in the generated recipe.

- **1.0**: All requested ingredients are mentioned in the recipe
- **0.5**: Only half of the requested ingredients are used
- **0.0**: None of the requested ingredients appear in the recipe

### 2. Structure Score (0.0 - 1.0)

Evaluates whether the recipe follows the expected structure with three key components:

- **Title**: A clear recipe name
- **Ingredients List**: Properly formatted list of ingredients with measurements
- **Instructions**: Clear cooking directions

Each component is worth 0.33 points, for a maximum of 1.0.

### 3. Step Clarity (0.0 - 1.0)

Assesses whether the recipe has clear, numbered steps:

- **1.0**: Recipe has 10+ well-defined steps
- **0.5**: Recipe has 5 steps
- **0.0**: Recipe lacks numbered steps or has fewer than 3 steps

### 4. Overall Quality (0.0 - 1.0)

A weighted combination of the above metrics:
- 40% Ingredient Coverage
- 30% Structure Score
- 30% Step Clarity

## Interpreting Your Results

Based on our evaluation, your current model shows:

- **Very low ingredient coverage**: The model often doesn't incorporate the requested ingredients
- **Poor structure**: The generated text doesn't follow recipe format (title, ingredients, instructions)
- **Lack of step clarity**: No clear cooking steps are provided
- **Incomplete generation**: The model seems to be generating partial responses or continuations of the prompt

## Common Issues and Solutions

### 1. Incomplete Generation

**Issue**: Your model is generating very short, incomplete responses.

**Solutions**:
- Increase `max_new_tokens` parameter (try 512 or 1024)
- Check if your training examples have complete recipes
- Ensure your model was trained for enough epochs

### 2. Poor Structure

**Issue**: The model doesn't generate properly structured recipes.

**Solutions**:
- Update training examples to have consistent structure
- Add explicit format markers in your prompt template
- Include more examples with clear title, ingredients list, and instructions sections

### 3. Low Ingredient Coverage

**Issue**: The model ignores requested ingredients.

**Solutions**:
- Modify prompt to emphasize using ALL provided ingredients
- Include examples where all ingredients are clearly used
- Add a system message that instructs the model to incorporate all ingredients

## Recommended Next Steps

Based on your current results, here are the recommended steps to improve your model:

1. **Check Training Data Format**:
   ```bash
   # View a few examples from your training data
   head -n 50 ~/recipe_manual_data/full_dataset.csv
   ```

2. **Examine Training Configuration**:
   - Review `config/text_generation_3060_geforce_rtx.yaml`
   - Ensure `prompt_template` includes clear instructions about format
   - Check that `max_length` is sufficient (at least 512)

3. **Create Better Test Examples**:
   - Update `data/processed/recipe_test_examples.json` with more structured examples
   - Ensure examples follow the exact format you want the model to learn

4. **Retrain with More Data**:
   - Increase `max_train_samples` from 10 to at least 1000
   - Train for more epochs (8-10 instead of 4)

5. **Test with Different Generation Parameters**:
   ```bash
   make recipe-quality-check INGREDIENTS="chicken, rice, garlic" MODEL_DIR="models/recipe_assistant_3060_geforce_rtx" MAX_TOKENS=512 TEMPERATURE=0.5
   ```

## Advanced Troubleshooting

If the above steps don't improve results:

1. **Check Model Loading**:
   - Verify the model is loading correctly
   - Ensure LoRA adapters are properly applied

2. **Examine Raw Model Outputs**:
   - Use the `recipe_demo.py` script to see full model outputs
   - Check for truncation or formatting issues

3. **Try Different Base Models**:
   - TinyLlama may be too small for complex recipe generation
   - Consider using a larger base model (7B+ parameters)

4. **Analyze Training Loss**:
   - Check if training loss decreased properly
   - Look for signs of overfitting or underfitting

## Using This Guide

Run the evaluation tools regularly as you make changes:

```bash
# Full evaluation
python -m src.model.recipe_evaluate --config config/text_generation_3060_geforce_rtx.yaml

# Quick check with specific ingredients
make recipe-quality-check INGREDIENTS="chicken, rice, garlic" MODEL_DIR="models/recipe_assistant_3060_geforce_rtx"
```

Track your progress by comparing metrics before and after changes to identify what improvements are most effective. 