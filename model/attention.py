"""
Day 4: Multi-Head Attention (MHA) with Rotary Position Embeddings (RoPE) and Causal Masking.

1. Attention Intuition & Softmax:
   - Raw logits indicate relative magnitudes but lack probabilistic interpretability.
   - Softmax transforms unbounded dot-product scores into a valid probability distribution [0, 1] summing to 1.
   - Equation: S(z_i) = exp(z_i) / sum_j(exp(z_j)). The exponential ensures non-negativity and widens peak margins.

2. Causal Masking:
   - Autoregressive decoders must not attend to future tokens.
   - Upper-triangular positions are set to -inf; softmax maps exp(-inf) to exactly 0.0 attention weight.

3. Key Projections:
   - Query (Q): Contextual criteria of what the current token is seeking.
   - Key (K): Descriptive indices of what information the token holds.
   - Value (V): Raw semantic content to be propagated downstream.
   - Scaled Dot-Product: QK^T / sqrt(d_k) mitigates vanishing gradients caused by high-dimensional variance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rope import apply_rotary_emb, precompute_freqs_cis


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads  # Split embedding space across heads (e.g., 64 // 4 = 16)

        assert dim % n_heads == 0, "Embedding dimension must be divisible by the number of heads."

        # Linear projections into Q, K, V subspaces (bias=False saves memory and stabilizes weight decay)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)

        # Output projection layer: Fuses concatenated head representations back into the residual stream
        self.wo = nn.Linear(dim, dim, bias=False)
        
        self.register_buffer( "cache_k",
                             torch.zeros(
                (max_batch_size, n_heads, max_seq_len, self.head_dim)
            ),
            persistent=False,#register the tensro  we have to maintain
         
        )
        self.register_buffer(
            "cache_v",
            torch.zeros(#dont store cache buffer at weight file
                (max_batch_size, n_heads, max_seq_len, self.head_dim)
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape  # Extract batch size and sequence length from input shape (B, S, D)

        # 1. Project inputs: (Batch, Seq_Len, Dim) @ (Dim, Dim)^T -> (Batch, Seq_Len, Dim)
        # Represents 8 tokens evaluated against 64 orthogonal transformation axes
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # 2. Reshape into multi-head format: (Batch, Seq_Len, n_heads, head_dim)
        # e.g., (2, 8, 64) -> (2, 8, 4, 16)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # 3. Apply Rotary Position Embedding (RoPE)
        # Embeds relative positional distance into 2D complex subspace rotations on Q and K exclusively
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        # 4. Transpose axes for batched matrix multiplication across heads: (Batch, n_heads, Seq_Len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        
        
        if use_cache:
            end_pos = start_pos + seq_len
            if start_pos < 0 or end_pos > self.cache_k.shape[2]:
                raise ValueError("Token positions exceed the KV cache capacity.")
            if batch_size > self.cache_k.shape[0]:
                raise ValueError("Batch size exceeds max_batch_size for KV caching.")
            self.cache_k[:batch_size, :, start_pos:end_pos] = xk
            self.cache_v[:batch_size, :, start_pos:end_pos] = xv
            keys = self.cache_k[:batch_size, :, :end_pos]
            values = self.cache_v[:batch_size, :, :end_pos]
        else:
            keys = xk
            values = xv

        # Cached decoding attends to the prefix and current causal segment.
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        prefix_len = start_pos if use_cache else 0
        query_positions = prefix_len + torch.arange(seq_len, device=x.device)
        key_positions = torch.arange(keys.shape[2], device=x.device)
        mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))

        # 7. Normalize attention distribution via Softmax and compute weighted sum of values
        # Cast to float32 before softmax for numerical stability, then restore original dtype
        probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(probs, values)  # (Batch, n_heads, Seq_Len, head_dim)

        # 8. Recombine heads: (B, H, S, D) -> (B, S, H, D) -> (B, S, Dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        # 9. Final linear projection back to residual path
        return self.wo(output)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dummy input: (Batch=2, Seq_Len=8, Dim=64)
    x = torch.randn(2, 8, 64, device=device)

    # Precompute RoPE complex frequencies for head_dim=16
    head_dim = 64 // 4  # 16
    freqs_cis = precompute_freqs_cis(dim=head_dim, end=16).to(device)

    # Instantiate 4-head attention block
    mha = MultiHeadAttention(dim=64, n_heads=4).to(device)

    out = mha(x, freqs_cis)
    print("입력 형태:", x.shape)
    print("출력 형태:", out.shape)
    print("Multi-Head Attention 블록 연산 성공!")