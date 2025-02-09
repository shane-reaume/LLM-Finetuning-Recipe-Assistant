from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
