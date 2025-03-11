# Recipe Model Prompt Format Guide

This document explains how to properly format prompts for the Recipe Assistant model to get high-quality outputs.

## The Correct Prompt Format

The recipe model requires this specific format:

```
<|system|>You are a recipe assistant. Create detailed recipes with exact measurements and clear instructions.<|endoftext|>
<|user|>Write a complete recipe using these ingredients: {ingredients}. Include a title, ingredients list with measurements, and numbered cooking steps.<|endoftext|>
<|assistant|>
```

## Why This Format Is Critical

The model was fine-tuned using this exact format with special tokens. When using the model for inference:

1. **Special Tokens**: The `<|system|>`, `<|user|>`, and `<|assistant|>` markers help the model understand the conversation structure
2. **End of Text Markers**: The `<|endoftext|>` tokens indicate where each part of the conversation ends
3. **Prompt Content**: The instructions requesting a title, ingredients with measurements, and numbered steps guide the model to produce properly structured output

## Common Issues When Using Incorrect Formats

If you don't use the correct format, you may encounter:

- Model repeating the instruction text instead of generating a recipe
- Missing or incomplete sections (no title, missing measurements, unnumbered steps)
- Poor structure and low quality scores
- Very short or truncated responses

## Examples

### Correct Usage (CLI):

```bash
python -m src.recipe_demo --model_dir="models/recipe_assistant" --ingredients="chicken, rice, garlic" --enhanced
```

### Correct Usage (Python code):

```python
def generate_recipe(model, tokenizer, ingredients, temperature=0.5, max_tokens=512):
    """Generate a recipe using the ingredients provided"""
    # Format the prompt correctly
    prompt = f"<|system|>You are a recipe assistant. Create detailed recipes with exact measurements and clear instructions.<|endoftext|>\n<|user|>Write a complete recipe using these ingredients: {ingredients}. Include a title, ingredients list with measurements, and numbered cooking steps.<|endoftext|>\n<|assistant|>"
    
    # Generate the recipe
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
        )
    
    generation_time = time.time() - generation_start_time
    print(f"Recipe generated in {generation_time:.2f} seconds")
    
    recipe_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract just the assistant's response
    assistant_response = recipe_text.split("<|assistant|>")[1].strip()
    
    return assistant_response
```

## Extending the Format

If you need to extend the format for more complex interactions, maintain the same structure:

```
<|system|>You are a recipe assistant. Create detailed recipes with exact measurements and clear instructions.<|endoftext|>
<|user|>Write a complete recipe using these ingredients: {ingredients}. Include a title, ingredients list with measurements, and numbered cooking steps.<|endoftext|>
<|assistant|>{first_response}<|endoftext|>
<|user|>Can you modify this recipe to be vegetarian?<|endoftext|>
<|assistant|>
```

## Troubleshooting

If you're getting poor results:

1. **Double-check the prompt format** exactly matches the one shown above
2. **Ensure all special tokens** are included (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|endoftext|>`)
3. **Verify that the model path** is correctly pointing to the trained recipe model
4. **Check generation parameters** like temperature, max_tokens, etc.

For more advanced usage and examples, see the [Getting Started Guide](../GETTING_STARTED.md). 