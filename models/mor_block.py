import torch
import torch.nn as nn
from models.router import Router  


class Expert(nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


class MoRBlock(nn.Module):
    def __init__(self, hidden_dim, ff_dim, num_experts, top_k=1, dropout=0.1, use_depth_embed=False, max_depth=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.use_depth_embed = use_depth_embed

        self.norm = nn.LayerNorm(hidden_dim)
        self.router = Router(hidden_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ff_dim, dropout) for _ in range(num_experts)
        ])
        self.depth_embed = nn.Embedding(max_depth, hidden_dim) if use_depth_embed else None

    def forward(self, x, depth=0):
        """
        x: (batch, seq_len, hidden_dim)
        depth: int, recursion depth (if using depth embeddings)
        """
        residual = x
        x = self.norm(x)

        if self.depth_embed is not None:
            max_allowed_depth = self.depth_embed.num_embeddings - 1
            safe_depth = min(depth, max_allowed_depth)

            # Optional: print or assert for debugging
            assert 0 <= safe_depth <= max_allowed_depth, f"Invalid depth {depth}, clamped to {safe_depth}"
            depth_tensor = torch.tensor(safe_depth, device=x.device, dtype=torch.long)
            depth_emb = self.depth_embed(depth_tensor)
            x = x + depth_emb

        topk_scores, topk_indices = self.router(x)  # (batch, seq_len, top_k), (batch, seq_len, top_k)

        output = torch.zeros_like(x)

        for k in range(self.top_k):
            scores = topk_scores[:, :, k].unsqueeze(-1)  # (batch, seq_len, 1)
            indices = topk_indices[:, :, k]  # (batch, seq_len)

            for i, expert in enumerate(self.experts):
                mask = (indices == i).float().unsqueeze(-1)  # (batch, seq_len, 1)
                if mask.sum() == 0:
                    continue
                x_masked = x * mask
                expert_out = expert(x_masked)
                output += scores * expert_out

        return residual + output  # Residual connection

