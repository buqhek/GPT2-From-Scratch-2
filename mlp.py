"""MLP Module for GPT-2 Transformer Block"""

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define the fully connected layer for th MLP (hint: it should output 4 times the embedding dimensionality)
        self.c_fc = nn.Linear(in_features=config.n_embd, out_features=4*config.n_embd)
        # Define the projection layer for the MLP (hint: it should output the same dimensionality as the input) (n_embd)
        self.c_proj = nn.Linear(in_features=4*config.n_embd, out_features=config.n_embd)
        # Define a GeLU activation function. We use GeLU as it is the activation function used in GPT-2
        self.gelu = nn.GELU()

    def forward(self, x):
        # Feedforward the input through the fully connected layer, apply the GeLU activation, then project back to the embedding dimensionality
        x = self.c_fc(x)
        x = self.gelu(x) # Replace this with your code to feedforward through the fully connected layer, you may need more than one line
        x = self.c_proj(x)
        return x