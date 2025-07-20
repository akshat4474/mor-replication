from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def get_dataset(cfg):
    dataset_name = cfg["dataset"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["dataset"]["tokenizer_name"])
    
    if dataset_name == "tinystories":
        dataset = load_dataset("roneneldan/TinyStories", split="train")
    elif dataset_name == "wikitext-2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    else:
        raise ValueError("Unknown dataset: " + dataset_name)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=cfg["dataset"]["max_length"])

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids"])
    return DataLoader(tokenized, batch_size=cfg["dataset"]["batch_size"], shuffle=True)
