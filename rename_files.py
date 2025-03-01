import os
import shutil

# Mapping of current file paths to new file paths
file_mappings = {
    "src/data/create_test_set.py": "src/data/sentiment_create_test_set.py",
    "src/data/create_balanced_test_set.py": "src/data/sentiment_create_balanced_test_set.py",
    "src/data/prepare_gen_dataset.py": "src/data/recipe_prepare_dataset.py",
    "src/model/train.py": "src/model/sentiment_train.py",
    "src/model/evaluate.py": "src/model/sentiment_evaluate.py",
    "src/model/inference.py": "src/model/sentiment_inference.py",
    "src/model/generation/train.py": "src/model/recipe_train.py",
    "src/model/generation/evaluate.py": "src/model/recipe_evaluate.py",
    "src/model/generation/export_to_ollama.py": "src/model/recipe_export_to_ollama.py"
}

# Perform renaming
for old_path, new_path in file_mappings.items():
    old_full_path = os.path.join(os.getcwd(), old_path)
    new_full_path = os.path.join(os.getcwd(), new_path)
    
    # Skip files that don't exist
    if not os.path.exists(old_full_path):
        print(f"Warning: {old_full_path} not found, skipping")
        continue
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(new_full_path), exist_ok=True)
    
    # Copy the file (use copy in case imports need updating)
    shutil.copy2(old_full_path, new_full_path)
    print(f"Copied {old_path} to {new_path}")

print("\nFile renaming complete. You'll need to update imports in the files.")
print("Old files were kept in case imports need to be fixed.")
