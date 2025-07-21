import argparse
import yaml
import torch
from transformers import AutoTokenizer


from training.dataset import get_dataset, TinyStoriesDataset, TinyStoriesIterableDataset
from training.train import train_one_epoch
from models.mor_model import MoRModel
from evaluation import evaluate
from inference import run_inference

from torch.utils.data import DataLoader
from torch import nn, optim
import os


def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--infer", type=str, help="Run inference on input text")
    args = parser.parse_args()

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} | Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")

    # Load dataset and tokenizer
    dataset, tokenizer = get_dataset(config)

    if config["dataset"].get("streaming", False):
        train_dataset = TinyStoriesIterableDataset(dataset, tokenizer, max_length=config["dataset"]["max_length"])
        train_loader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"])
        val_loader = None
    else:
        sample_limit = config["dataset"].get("sample_limit")
        if sample_limit:
            if "train" in dataset:
                dataset["train"] = dataset["train"].select(range(min(sample_limit, len(dataset["train"]))))
            else:
                raise ValueError("No 'train' split found in the dataset. Available splits: {}".format(dataset.keys()))
            val_split = config["dataset"].get("val_split", "validation")
            if val_split in dataset:
                dataset[val_split] = dataset[val_split].select(
                    range(min(sample_limit // 5, len(dataset[val_split])))
                )
            else:
                raise ValueError(f"No '{val_split}' split found in the dataset. Available splits: {dataset.keys()}")

        train_dataset = TinyStoriesDataset(dataset, tokenizer, split="train")
        val_dataset = TinyStoriesDataset(dataset, tokenizer, split=config["dataset"].get("val_split", "validation"))
        train_loader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config["dataset"]["batch_size"], drop_last=False)



    # Initialize model
    model = MoRModel(**config["model"]).to(device)
    config["training"]["learning_rate"] = float(config["training"]["learning_rate"])
    config["training"]["weight_decay"] = float(config["training"]["weight_decay"])
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                            weight_decay=config["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    use_amp = config["training"].get("use_amp", True)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    # Train
    if args.train:
        start_epoch = 0
        interrupt_ckpt = "checkpoints/mor_interrupt.pt"
        default_ckpt = config.get("checkpoint_path", "checkpoints/mor_last.pt")
        checkpoint_path = interrupt_ckpt if os.path.exists(interrupt_ckpt) else default_ckpt

        if os.path.exists(checkpoint_path):
            print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scaler and checkpoint['scaler_state_dict']:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            global_step = checkpoint.get('global_step', 0)
        else:
            print("[INFO] No checkpoint found. Starting from scratch.")
            global_step = 0

        try:
            for epoch in range(start_epoch, config["training"]["epochs"]):
                print(f"[INFO] Starting Epoch {epoch+1}/{config['training']['epochs']}")
                start_time = time.time()
                resume_batch = checkpoint.get('resume_batch', 0) if os.path.exists(checkpoint_path) else 0
                max_batches = config["dataset"].get("max_batches", None)

                avg_loss, global_step, next_batch = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    scaler,
                    start_batch=resume_batch,
                    global_step=global_step,
                    max_batches=max_batches
                )
                duration = time.time() - start_time
                print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f} | Time: {duration:.2f}s")

                os.makedirs("checkpoints", exist_ok=True)

                # Save both epoch-specific and 'last' checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'epoch': epoch,
                    'global_step': global_step,
                    'resume_batch': next_batch
                }
                torch.save(checkpoint, f"checkpoints/mor_epoch{epoch+1}.pt")
                torch.save(checkpoint, "checkpoints/mor_last.pt")
                # Remove interrupt checkpoint if it existed
                if os.path.exists(interrupt_ckpt):
                    os.remove(interrupt_ckpt)

        except KeyboardInterrupt:
            print("\n[WARNING] Training interrupted. Saving checkpoint to 'mor_interrupt.pt'.")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'epoch': epoch,
                'global_step': global_step
            }, interrupt_ckpt)
            print("[INFO] You can resume later using this interrupt checkpoint.")


    # Evaluate
    if args.eval:
        model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))
        model.eval()
        evaluate(model, val_loader, device)

    # Inference
    if args.infer:
        model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))
        model.eval()
        run_inference(model, tokenizer, args.infer, device)

if __name__ == "__main__":
    main()