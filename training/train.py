import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer
import yaml
import os
from tqdm import tqdm
import time
from itertools import islice



from models.mor_model import MoRModel
from training.dataset import TinyStoriesDataset

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)



def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scaler=None,
    start_batch=0,
    global_step=0,
    max_batches=None
):
    model.train()
    total_loss = 0
    is_iterable = not hasattr(dataloader.dataset, '__len__')

    # Apply islice if iterable and resuming or limiting
    if is_iterable and (start_batch > 0 or max_batches is not None):
        end_batch = None if max_batches is None else start_batch + max_batches
        dataloader = islice(dataloader, start_batch, end_batch)

    if not is_iterable:
        total_batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
        dataloader = islice(dataloader, start_batch, start_batch + total_batches)
    else:
        total_batches = max_batches

    progress = tqdm(
        dataloader,
        desc="Training",
        ncols=120,
        initial=start_batch,
        total=total_batches
    )

    start_time = time.perf_counter()

    for batch_idx, batch in enumerate(progress, start=start_batch):
        input_ids = batch['input_ids'].to(device)
        labels = input_ids.clone()

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        global_step += 1

        # ETA calculation
        elapsed = time.perf_counter() - start_time
        steps_done = batch_idx - start_batch + 1
        time_per_step = elapsed / steps_done
        remaining = (total_batches - steps_done) if total_batches else 0
        eta = time_per_step * remaining

        progress.set_postfix(
            batch=batch_idx + 1,
            loss=loss.item(),
            global_step=global_step,
            eta=f"{int(eta // 60)}m {int(eta % 60)}s"
        )

        # Stop early if max_batches reached
        if max_batches and steps_done >= max_batches:
            break

    return total_loss / steps_done, global_step, batch_idx + 1  # batch_idx + 1 is next_start_batch

def main():
    config = load_config()
    dataset_name = config['dataset']
    model_config = config['model']
    train_config = config['training']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    dataset_path = os.path.join("data", dataset_name, "tokenized")
    dataset = load_from_disk(dataset_path)
    train_dataset = TinyStoriesDataset(dataset, tokenizer)

    dataloader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        drop_last=True
    )

    # Initialize model
    model = MoRModel(**model_config)
    model.to(device)

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"])
    criterion = nn.CrossEntropyLoss()

    # Training
    global_step = 0
    for epoch in range(train_config["epochs"]):
        avg_loss, global_step = train_one_epoch(
            model, dataloader, optimizer, criterion, device, start_step=global_step
        )
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        # Save model checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/mor_epoch{epoch+1}.pt")

if __name__ == "__main__":
    main()
