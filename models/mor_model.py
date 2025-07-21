import torch
import torch.nn as nn
from models.mor_block import MoRBlock

class MoRModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        ff_dim,
        num_experts,
        num_layers=4,
        top_k=1,
        dropout=0.1,
        use_depth_embed=False,
        max_depth=4,
        max_length=256
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_length, hidden_dim)

        self.blocks = nn.ModuleList([
            MoRBlock(
                hidden_dim,
                ff_dim,
                num_experts,
                top_k=top_k,
                dropout=dropout,
                use_depth_embed=use_depth_embed,
                max_depth=max_depth
            )
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        for depth, block in enumerate(self.blocks):
            x = block(x, depth=depth)

        x = self.ln_f(x)
        logits = self.output_proj(x)
        return logits
