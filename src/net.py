from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from morphers.base.base import Morpher

from cmlk import Transformer

class DarvishNet(nn.Module):
    
    def __init__(
        self,
        feature_morphers: dict[str, Morpher],
        pitcher_morpher: Morpher,
        sequence_length: int,
        d_model: int,
        n_transformer_layers: int,
        n_kv: int,
        n_q: int,
        positional_encoding: str,
        attn_args: dict[str, Any],
    ):
        super().__init__()
        self.d_model = d_model

        self.input_embedders = {
            column: morpher.make_embedding(d_model)
            for column, morpher in feature_morphers.items()
        }
        self.pitcher_embedder = pitcher_morpher.make_embedding(d_model)
        
        transformer_layer_args = {
            "d_model": self.d_model,
            "n_kv_heads": n_kv,
            "n_q_heads": n_q,
            "ff_dim": self.d_model * 3, # sure
        }

        self.transformer = Transformer(
            n_layers=n_transformer_layers,
            position_encoding=positional_encoding,
            layer_args=transformer_layer_args,
            attn_args=attn_args,
        )
        self.input_activation = nn.GELU()
        

    def forward(self, x: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:

        # (n, e)
        pitcher_embedding = self.pitcher_embedder(x["pitcher"])
        # (n, s, e)
        x = sum(embedder(x[column]) for column, embedder in self.input_embedders.items())

        x = self.input_activation(x)
        x = self.transformer(x, mask=mask)
        multiplicative_mask = mask.where(mask.isinf(), 0.0, 1.0)
        # Mean pool
        # (n, e)
        x = (x * multiplicative_mask).sum(dim=1) / multiplicative_mask.sum(dim=1)

        # (n)
        return (x * pitcher_embedding).sum(dim=1)