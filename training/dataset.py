import os
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset as TorchDataset, IterableDataset as TorchIterableDataset

# Regular Dataset for non-streaming mode
class TinyStoriesDataset(TorchDataset):
    def __init__(self, hf_dataset, tokenizer, split="train"):
        if isinstance(hf_dataset, DatasetDict):
            self.dataset = hf_dataset[split]
        else:
            self.dataset = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item.get("attention_mask", [1] * len(item["input_ids"])), dtype=torch.long)
        }

# IterableDataset for streaming mode
class TinyStoriesIterableDataset(TorchIterableDataset):
    def __init__(self, hf_streaming_dataset, tokenizer, max_length=128):
        self.dataset = hf_streaming_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for item in self.dataset:
            tokens = self.tokenizer(
                item["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            yield {
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0)
            }

# Loader function
def get_dataset(cfg):
    dataset_name = cfg["dataset"]["name"]
    tokenizer_name = cfg["dataset"]["tokenizer_name"]
    max_length = cfg["dataset"]["max_length"]
    streaming = cfg["dataset"].get("streaming", False)
    save_dir = os.path.join(cfg["dataset"]["save_path"], dataset_name, "tokenized")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # STREAMING MODE
    if streaming:
        print(f"[INFO] Streaming dataset: {dataset_name}")
        if dataset_name == "tinystories":
            stream_dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        elif dataset_name == "wikitext-2":
            stream_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
        else:
            raise ValueError(f"Unknown dataset for streaming: {dataset_name}")
        return stream_dataset, tokenizer

    # NORMAL MODE
    if os.path.exists(save_dir):
        print(f"[INFO] Loading tokenized dataset from: {save_dir}")
        dataset = load_from_disk(save_dir)
    else:
        print(f"[INFO] Tokenizing and saving dataset: {dataset_name}")
        if dataset_name == "tinystories":
            raw_dataset = load_dataset("roneneldan/TinyStories")
        elif dataset_name == "wikitext-2":
            raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        def tokenize(batch):
            return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

        dataset = raw_dataset.map(tokenize, batched=True)
        dataset.set_format(type="torch", columns=["input_ids"])
        os.makedirs(save_dir, exist_ok=True)
        DatasetDict(dataset).save_to_disk(save_dir)

    return dataset, tokenizer
