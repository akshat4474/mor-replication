import torch
from transformers import AutoTokenizer
from models.mor_model import MoRModel
import yaml
import os

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_model(config, checkpoint_path, device):
    model_config = config["model"]
    model = MoRModel(**model_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_text(prompt, tokenizer, model, config, device, max_new_tokens=50):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    input_dim = model.input_proj.in_features

    # Convert to one-hot (or embedding-style) input vector
    # NOTE: this assumes input_dim == vocab_size
    input_onehot = torch.nn.functional.one_hot(input_ids, num_classes=input_dim).float()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_onehot)  # (batch, seq_len, vocab_size)
        next_token_logits = outputs[:, -1, :]  # Last token
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # Greedy decoding

        # Append new token
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        input_onehot = torch.nn.functional.one_hot(input_ids, num_classes=input_dim).float()

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def run_inference(model, tokenizer, prompt, device):
    config = load_config()  
    output_text = generate_text(prompt, tokenizer, model, config, device)
    print("\n[Generated Text]:")
    print(output_text)


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using:",device)

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    checkpoint_path = "checkpoints/mor_epoch4.pt"  # Change this if needed

    model = load_model(config, checkpoint_path, device)

    prompt = input("Enter a prompt: ")
    output_text = generate_text(prompt, tokenizer, model, config, device)
    print("\n[Generated Text]:")
    print(output_text)

if __name__ == "__main__":
    main()
