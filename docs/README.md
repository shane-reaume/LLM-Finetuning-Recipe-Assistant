# Recipe Assistant Documentation

This directory contains detailed documentation for the Recipe Assistant model.

## Available Documentation

- [Prompt Format Guide](PROMPT_FORMAT.md) - How to correctly format prompts for the model
- [Test Examples Guide](TEST_EXAMPLES.md) - Information about test examples and their format
- [Testing Guide](TESTING_GUIDE.md) - Best practices for testing the model

## Main Project Documentation

- [Project README](../README.md) - Project overview and structure
- [Getting Started Guide](../GETTING_STARTED.md) - Detailed setup and usage instructions

## Topics Covered

### Prompt Format

Our documentation explains the critical importance of using the correct prompt format:
```
<|system|>You are a recipe assistant. Create detailed recipes with exact measurements and clear instructions.<|endoftext|>
<|user|>Write a complete recipe using these ingredients: {ingredients}. Include a title, ingredients list with measurements, and numbered cooking steps.<|endoftext|>
<|assistant|>
```

### Testing Methodology

We provide guidance on:
- Testing tools available in the project
- Best practices for comprehensive testing
- Quality metrics and their interpretation
- Troubleshooting common issues

### Examples and Templates

The test examples documentation shows how to create and use high-quality examples for:
- Model evaluation
- Format validation
- Quality assessment

## Contributing to Documentation

When contributing to documentation:
1. Follow the existing format and style
2. Include practical examples where applicable
3. Explain why certain practices are important, not just what they are
4. Update this index when adding new documentation files 