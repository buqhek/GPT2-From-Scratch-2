# GPT-2 From Scratch

A clean, modular PyTorch implementation of the GPT-2 transformer architecture built from first principles. The implementation reproduces the original OpenAI GPT-2 design and validates correctness by loading pretrained weights directly from HuggingFace.

---

## Overview

This project implements GPT-2 from the ground up — no high-level wrappers, no shortcuts. Each component of the transformer is written as a standalone, readable module. The goal was to develop a deep understanding of how large language models actually work at the code level: how attention is computed, how tokens flow through transformer blocks, and how a trained model generates text.

---

## Architecture

The model follows the original GPT-2 paper spec with four supported sizes:

| Model       | Layers | Heads | Embedding Dim | Parameters |
|-------------|--------|-------|----------------|------------|
| gpt2        | 12     | 12    | 768            | 124M       |
| gpt2-medium | 24     | 16    | 1024           | 350M       |
| gpt2-large  | 36     | 20    | 1280           | 774M       |
| gpt2-xl     | 48     | 25    | 1600           | 1558M      |

### Key implementation details

- **Causal self-attention** (`attention.py`) — scaled dot-product attention with `is_causal=True`, proper Q/K/V splitting across heads, and output projection
- **Transformer block** (`block.py`) — pre-norm architecture combining `CausalSelfAttention` and `MLP` with residual connections
- **MLP** (`mlp.py`) — two-layer feed-forward network with GELU activation
- **Full GPT-2 model** (`gpt2.py`) — token and positional embeddings, stacked transformer blocks, final layer norm, and language model head
- **Weight tying** — input token embeddings and output projection share weights, matching the original GPT-2 design
- **Weight initialization** — linear layers initialized with `N(0, 0.02)`, following the GPT-2 paper
- **Pretrained weight loading** (`pretrained_gpt2.py`) — loads any of the four GPT-2 checkpoints from HuggingFace, handling the Conv1D → Linear weight transposition required by OpenAI's original format

---

## File Structure

```
GPT2-From-Scratch-2/
├── gpt2.py              # Main GPT model class, config, and from_pretrained loader
├── attention.py         # CausalSelfAttention module
├── block.py             # Transformer block (attention + MLP + residual)
├── mlp.py               # Feed-forward MLP module
├── dataloader.py        # Custom DataLoader for text datasets
├── pretrained_gpt2.py   # Script to load and run pretrained GPT-2 weights
├── input.txt            # Sample training text (Shakespeare)
├── utils.py             # Utility functions
└── requirements.txt     # Dependencies
```

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `transformers`

### Load a pretrained model

```python
from gpt2 import GPT

# Load pretrained GPT-2 weights from HuggingFace
model = GPT.from_pretrained('gpt2')  # or 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
model.eval()
```

### Train from scratch on a text dataset

```python
from gpt2 import GPT, GPTConfig
from dataloader import DataLoader

config = GPTConfig()  # defaults: 124M param GPT-2
model = GPT(config)

# Feed tokenized text through the DataLoader and run a training loop
# See dataloader.py for usage
```

---

## What I Learned

Building this from scratch rather than using a library forced engagement with details that are easy to miss:

- Why weight tying between input embeddings and the output head works and how to implement it in PyTorch without duplicating parameters
- How causal masking is enforced via `is_causal=True` in scaled dot-product attention
- The Conv1D → Linear transposition issue when importing OpenAI's original checkpoint format into standard PyTorch Linear layers
- How residual stream scaling and weight initialization interact with training stability at depth

---

## Acknowledgments

Built following [Andrej Karpathy's](https://github.com/karpathy) neural network series, with additional collaboration from [@ForkBombTwenty](https://github.com/ForkBombTwenty) and [@JasonCodesC](https://github.com/JasonCodesC).
