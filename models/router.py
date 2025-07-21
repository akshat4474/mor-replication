import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k=1):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        logits = self.gate(x)  # (batch, seq_len, num_experts)
        scores = F.softmax(logits, dim=-1)

        # Get top-k expert indices and weights
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        return topk_scores, topk_indices
