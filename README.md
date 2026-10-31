# LLaMA from Scratch

A small educational LLaMA-style language model implemented in PyTorch, with character-level tokenization, RoPE, RMSNorm, SwiGLU, causal attention, and KV caching.

## Setup

```sh
pip install -r requirements.txt
```

## Run

```sh
python train.py       # Train on a short example sentence and save checkpoint.pt
python inference.py  # Run checkpoint-based inference
python generate.py   # Demonstrate KV-cached generation with randomly initialized weights
```

Model components live in `model/`. To run the architecture smoke check:

```sh
python -m model.lamma
```

Training uses a small fixed corpus for a memorization experiment. Generated checkpoints, Python caches, and local editor settings are excluded from version control.
