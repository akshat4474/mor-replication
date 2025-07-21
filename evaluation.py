import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer
import yaml
import os

from models.mor_model import MoRModel
from training.dataset import TinyStoriesDataset

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()

        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    dataset_path = os.path.join("data", config["dataset"], "tokenized")
    dataset = load_from_disk(dataset_path)
    val_dataset = TinyStoriesDataset(dataset, tokenizer, split=config.get("val_split", "validation"))

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    # Load model
    model = MoRModel(**config["model"])
    model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))
    model.to(device)

    # Evaluation
    criterion = nn.CrossEntropyLoss()
    val_loss, val_ppl = evaluate(model, val_loader, criterion, device)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Perplexity: {val_ppl:.4f}")


if __name__ == "__main__":
    main()
