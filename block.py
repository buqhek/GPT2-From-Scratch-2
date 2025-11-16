"""Attention Block for Transformer model."""

import torch
import torch.nn as nn
from attention import CausalSelfAttention
from mlp import MLP

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # TODO: Define the two layer normalization layers and the attention layer
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
        # Need to define custom MLP because it is not just one feedforward layer but many in paralell for each token.
        # TODO: Define the MLP layer
        self.mlp = MLP(config)

    def forward(self, x):
        # TODO: Feedforward the result through the attention layer and the MLP layer to get the final output
        # Will need more than just this line
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x