import os
import json
import argparse
from datasets import concatenate_datasets
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

from src.utils.config_utils import load_config
# Fix the import to use the correct file
from src.data.recipe_prepare_dataset import prepare_generation_dataset

def prepare_model_and_tokenizer(config):
    """
    Load and prepare the model and tokenizer for training
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: model and tokenizer
    """
    print(f"Loading base model: {config['model']['name']}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization if specified
    load_in_8bit = config["model"].get("load_in_8bit", False)
    if load_in_8bit:
        print("Loading model in 8-bit mode for memory efficiency")
        
    # Always use FP32 for training
    print("Using FP32 precision for training")
    dtype = torch.float32
    
    # Load model with memory optimizations
    print("Loading model with memory optimizations...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=dtype,
        device_map="auto",
        load_in_8bit=load_in_8bit,
        low_cpu_mem_usage=config["model"].get("low_cpu_mem_usage", True),
        use_cache=config["model"].get("use_cache", False),  # Disable KV cache to save memory
    )
    
    # Enable gradient checkpointing if specified
    if config["model"].get("gradient_checkpointing", False):
        print("Enabling gradient checkpointing for memory efficiency")
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # Disable KV cache when using gradient checkpointing
    
    # Apply LoRA if enabled in config
    if config["training"].get("lora", {}).get("enabled", False):
        print("Applying LoRA for efficient fine-tuning")
        lora_config = config["training"]["lora"]
        
        # Prepare model for k-bit training only if using quantization
        if load_in_8bit:
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("alpha", 16),
            lora_dropout=lora_config.get("dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"])
        )
        
        # Get PEFT model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Apply additional memory optimizations
    print("Applying additional memory optimizations...")
    
    # Set max memory for the model
    if "max_memory_MB" in config["training"]:
        max_memory = config["training"]["max_memory_MB"]
        print(f"Setting max memory to {max_memory}MB")
        # This is a placeholder - actual implementation would depend on the model
        # and the specific memory optimization techniques available
    
    # Disable gradient for non-trainable parameters
    for param in model.parameters():
        if not param.requires_grad:
            param.grad = None
    
    return model, tokenizer

def preprocess_data(dataset, tokenizer, config):
    """
    Tokenize and format dataset for training
    
    Args:
        dataset: Hugging Face dataset
        tokenizer: Tokenizer to use
        config (dict): Configuration dictionary
    
    Returns:
        dataset: Tokenized dataset
    """
    prompt_template = config["data"]["prompt_template"]
    max_length = config["data"]["max_length"]
    
    def tokenize_function(examples):
        # Combine prompt and response
        texts = []
        for prompt, response in zip(examples["prompt"], examples["response"]):
            # Handle case where response is a list of strings (recipe steps)
            if isinstance(response, list):
                response = "\n".join(response)
            elif not isinstance(response, str):
                response = str(response)
                
            # Format: <prompt><response><eos>
            text = prompt + response + tokenizer.eos_token
            texts.append(text)
            
        # Tokenize
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Set labels equal to input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Apply preprocessing
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "response"]
    )
    
    return tokenized_dataset

def train_model(config, data_dir=None):
    """
    Train a text generation model
    
    Args:
        config (dict): Configuration dictionary
        data_dir (str): Path to directory with manually downloaded dataset
    """
    # Load model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(config)
    
    # Prepare dataset
    train_data, val_data = prepare_generation_dataset(config, data_dir)
    
    # Check if dataset preparation was successful
    if train_data is None or val_data is None:
        print("Dataset preparation failed. Please check the error messages above.")
        return
    
    # Limit training samples if specified
    max_train_samples = config["data"].get("max_train_samples", None)
    if max_train_samples is not None:
        print(f"Limiting training data to {max_train_samples} examples")
        train_data = train_data.select(range(min(max_train_samples, len(train_data))))
    
    # Tokenize datasets
    print("Preprocessing data...")
    train_dataset = preprocess_data(train_data, tokenizer, config)
    eval_dataset = preprocess_data(val_data, tokenizer, config)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked
    )
    
    # Set up training arguments
    training_config = config["training"]
    
    # Create a dictionary of training arguments
    training_args_dict = {
        "output_dir": config["model"]["save_dir"],
        "num_train_epochs": float(training_config["num_train_epochs"]),
        "per_device_train_batch_size": int(training_config["batch_size"]),
        "per_device_eval_batch_size": int(training_config["batch_size"]),
        "gradient_accumulation_steps": int(training_config["gradient_accumulation_steps"]),
        "learning_rate": float(training_config["learning_rate"]),
        "weight_decay": float(training_config["weight_decay"]),
        "warmup_steps": int(training_config["warmup_steps"]),
        "save_steps": int(training_config["save_steps"]),
        "save_total_limit": int(training_config["save_total_limit"]),
        "logging_dir": training_config["logging_dir"],
        "evaluation_strategy": training_config["evaluation_strategy"],
        "eval_steps": int(training_config["eval_steps"]),
        "fp16": training_config["fp16"],
        "bf16": training_config.get("bf16", False),
        "load_best_model_at_end": True,
        "gradient_checkpointing": training_config.get("gradient_checkpointing", False),
    }
    
    # Add dataloader settings if specified
    if "dataloader_num_workers" in training_config:
        training_args_dict["dataloader_num_workers"] = int(training_config["dataloader_num_workers"])
    
    if "dataloader_pin_memory" in training_config:
        training_args_dict["dataloader_pin_memory"] = training_config["dataloader_pin_memory"]
    
    # Remove optim_args from training arguments to avoid parsing issues
    optim_args = None
    if "optim_args" in training_config:
        optim_args = training_config["optim_args"]
    
    # Create training arguments
    training_args = TrainingArguments(**training_args_dict)
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Set optimizer arguments after trainer creation if needed
    if optim_args:
        print(f"Setting optimizer arguments: {optim_args}")
        # We'll set these manually after the optimizer is created
        trainer.optimizer_kwargs = optim_args
    
    # Apply memory optimizations before training
    print("Applying memory optimizations before training...")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache")
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model and tokenizer
    print(f"Saving model to {config['model']['save_dir']}")
    trainer.save_model(config["model"]["save_dir"])
    tokenizer.save_pretrained(config["model"]["save_dir"])
    
    # Save model configuration for testing
    model_info = {
        "model_name": config["model"]["name"],
        "max_length": config["data"]["max_length"],
        "prompt_template": config["data"]["prompt_template"],
        "lora_enabled": config["training"].get("lora", {}).get("enabled", False)
    }
    
    with open(os.path.join(config["model"]["save_dir"], "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("Training complete!")

def main():
    parser = argparse.ArgumentParser(description="Train a text generation model")
    parser.add_argument("--config", type=str, default="config/text_generation.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to directory containing manually downloaded dataset")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Ensure output directories exist
    os.makedirs(config["model"]["save_dir"], exist_ok=True)
    os.makedirs(config["training"]["logging_dir"], exist_ok=True)
    
    # Train the model
    train_model(config, args.data_dir)

if __name__ == "__main__":
    main()
