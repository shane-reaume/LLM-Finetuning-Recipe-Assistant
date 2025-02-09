import os
from transformers import Trainer, TrainingArguments
from src.model.model_loader import load_model
from src.data.dataset import get_datasets  # Assume this function returns train and eval datasets

def train_model(config):
    model, tokenizer = load_model(config["model_name"])
    train_dataset, eval_dataset = get_datasets(config["data_path"])

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        logging_dir=config.get("logging_dir", "./logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])

if __name__ == "__main__":
    from src.utils.config_utils import load_config
    config = load_config()
    train_model(config)
