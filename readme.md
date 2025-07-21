# 🧠 Mixture-of-Recursions (MoR) - Unofficial Replication
This project is an unofficial implementation of Google DeepMind's Mixture-of-Recursions architecture for efficient language modeling.

---

## 📦 Python Environment

- Python: `>=3.9`
- PyTorch: `>=2.0`
- CUDA: Optional but recommended for training efficiency
- Platform: Linux/Windows/macOS (tested on Windows)

---

## 🧰 Required Python Libraries

| Library        | Purpose                                           | Version     |
|----------------|---------------------------------------------------|-------------|
| `torch`        | Core deep learning framework                      | `2.7.1`     |
| `transformers` | Tokenizers, comparison baselines                  | `4.53.2`    |
| `datasets`     | For loading datasets like WikiText or TinyStories | `4.0.0`     |
| `numpy`        | Tensor manipulation                               | `2.2.6`     |
| `tqdm`         | Progress bars                                     | `4.67.1`    |
| `wandb`        | Experiment tracking                               | `0.21.0`    |
| `yaml`         | Config file parsing                               | `6.0.2`     |
| `matplotlib`   | Visualizing depth assignment                      | `3.10.3`    |
| `tokenizers`   | Efficient tokenization backend                    | `0.21.2`    |

```bash
pip install torch datasets transformers numpy tqdm pyyaml matplotlib tokenizers wandb
```

---

## 🗂️ Directory Structure

```bash
mor-replication/
│
├── main.py                      # Entry point for training or evaluation
├── config.yaml                  # Configuration file for dataset, model, and training parameters
├── requirements.md              # List of Python dependencies
│
├── data/                        # Local storage for cached datasets (e.g., TinyStories, WikiText-2)
│   └── ...                      # Automatically populated by HuggingFace Datasets
│
├── models/                      # Model architecture components
│   ├── mor_block.py             # Core Mixture-of-Recursions block definition
│   ├── router.py                # Routing logic for dynamic module selection
│
├── training/                    # Training and dataset preprocessing
│   ├── train.py                 # Training loop and model optimization
│   ├── dataset.py               # Dataset loading, tokenization, and DataLoader setup
│
├── utils/                       # Utility functions
│   ├── masking.py               # Custom masking logic for attention
