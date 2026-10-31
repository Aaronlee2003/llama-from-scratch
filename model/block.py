"""
Day 5: LLaMA Transformer Block (Decoder Layer)

1. Pre-Norm Architecture:
   - Modern LLMs replace Post-Norm with Pre-Norm:
     Post-Norm: x_{l+1} = Norm(x_l + SubLayer(x_l))
     Pre-Norm:  x_{l+1} = x_l + SubLayer(Norm(x_l))
   - Mitigates gradient distortion during backpropagation, eliminating the strict
     need for warm-up schedules and stabilizing deep-layer training.

2. Residual Highway & Gradient Preservation:
   - Formulation: y = x + f(x)
   - Partial Derivative: dL/dx = dL/dy * (1 + f'(x)) = dL/dy + dL/dy * f'(x)
   - The identity '+ 1' term acts as a gradient highway, ensuring error signals 
     propagate to the initial layers without attenuation.

3. Mathematical Formulation:
   h = x + Attention(RMSNorm(x), freqs_cis)
   out = h + FeedForward(RMSNorm(h))
"""

import torch
import torch.nn as nn
from model.normalization import RMSNorm
from model.attention import MultiHeadAttention
from model.feedforward import FeedForward
from model.rope import precompute_freqs_cis


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        hidden_dim: int = None,
        multiple_of: int = 256,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        # 1. Attention Pre-Normalization: Stabilizes token variance before attention routing
        self.attention_norm = RMSNorm(dim=dim)

        # 2. Multi-Head Attention Sublayer: Captures intra-sequence contextual dependencies
        self.attention = MultiHeadAttention(
            dim=dim,
            n_heads=n_heads,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        # 3. FFN Pre-Normalization: Re-normalizes representations before non-linear gating
        self.ffn_norm = RMSNorm(dim=dim)

        # 4. SwiGLU FeedForward Sublayer: Projects and filters fine-grained token representations
        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim, multiple_of=multiple_of)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        # Step 1: Pre-Norm -> Multi-Head Attention -> Residual Addition (x + f(x))
        # Tensor Shape: (Batch, Seq_Len, Dim)
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cis,
            start_pos=start_pos,
            use_cache=use_cache,
        )

        # Step 2: Pre-Norm -> SwiGLU FeedForward -> Residual Addition (h + f(h))
        # Tensor Shape: (Batch, Seq_Len, Dim)
        out = h + self.feed_forward(self.ffn_norm(h))

        return out


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dummy tensor: (Batch=2, Seq_Len=8, Dim=64)
    x = torch.randn(2, 8, 64, device=device)

    # Precompute RoPE complex frequencies for head_dim = 64 // 4 = 16
    head_dim = 64 // 4
    freqs_cis = precompute_freqs_cis(dim=head_dim, end=16).to(device)

    # Instantiate Transformer Block
    block = TransformerBlock(dim=64, n_heads=4).to(device)

    out = block(x, freqs_cis)
    print("입력 형태:", x.shape)
    print("출력 형태:", out.shape)
    print("Transformer Block 단일 레이어 통합 연산 성공!")