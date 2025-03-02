import argparse
import os
from huggingface_hub import HfApi

def update_model_card(repo_name, model_card_path):
    """
    Update the model card (README.md) on the Hugging Face Hub
    
    Args:
        repo_name (str): Name of the repository on HF Hub (username/repo-name)
        model_card_path (str): Path to the model card markdown file
    """
    # Read the model card content
    with open(model_card_path, 'r') as f:
        model_card_content = f.read()
    
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Upload the README.md file
    api.upload_file(
        path_or_fileobj=model_card_path,
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="model"
    )
    
    print(f"Model card successfully updated for: https://huggingface.co/{repo_name}")

def main():
    parser = argparse.ArgumentParser(description="Update model card on Hugging Face Hub")
    parser.add_argument("--repo_name", type=str, required=True,
                        help="Repository name on Hugging Face Hub (e.g., username/model-name)")
    parser.add_argument("--model_card", type=str, default="model_card.md",
                        help="Path to the model card markdown file")
    args = parser.parse_args()
    
    # Check if model card file exists
    if not os.path.exists(args.model_card):
        raise FileNotFoundError(f"Model card file not found: {args.model_card}")
    
    # Update the model card
    update_model_card(args.repo_name, args.model_card)

if __name__ == "__main__":
    main()
