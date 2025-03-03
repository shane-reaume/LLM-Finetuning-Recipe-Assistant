import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.utils.config_utils import load_config

def publish_model(model_dir, repo_name, description=None):
    """
    Publish fine-tuned model to Hugging Face Hub
    
    Args:
        model_dir (str): Path to model directory
        repo_name (str): Name for the repository on Hugging Face Hub
        description (str, optional): Description for the model
    """
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Create default description if not provided
    if description is None:
        description = """
        # Sentiment Analysis Model
        
        This is a fine-tuned DistilBERT model for sentiment analysis on the IMDB dataset.
        
        ## Performance
        
        - Accuracy: 84.00%
        - F1 Score: 0.8462
        - Precision: 81.48%
        - Recall: 88.00%
        
        ## Usage
        
        ```python
        from transformers import pipeline
        
        sentiment = pipeline("sentiment-analysis", model="{repo_name}")
        result = sentiment("I really enjoyed this movie!")
        print(result)
        ```
        """.format(repo_name=repo_name)
    
    # Push to Hub
    model.push_to_hub(repo_name, use_auth_token=True)
    tokenizer.push_to_hub(repo_name, use_auth_token=True)
    
    print(f"Model successfully published to: https://huggingface.co/{repo_name}")
    
def main():
    parser = argparse.ArgumentParser(description="Publish model to Hugging Face Hub")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to model directory")
    parser.add_argument("--repo_name", type=str, required=True,
                        help="Repository name on Hugging Face Hub (e.g., username/model-name)")
    parser.add_argument("--description", type=str, default=None,
                        help="Description for the model")
    parser.add_argument("--config", type=str, default="config/sentiment_analysis.yaml", 
                        help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Use provided model_dir or default from config
    model_path = args.model_dir if args.model_dir else config["model"]["save_dir"]
    
    # Publish model
    publish_model(model_path, args.repo_name, args.description)

if __name__ == "__main__":
    main()
