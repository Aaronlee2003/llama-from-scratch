"""
Day 6: Complete LLaMA Architecture Stack & End-to-End Inference Validation

Architectural Overview:
1. Token Embedding:
   - Projects discrete integer token IDs into a continuous latent vector space (B, S) -> (B, S, D).
   - Functions as an O(1) lookup table rather than a dense matrix multiplication.

2. Deep Stacking with nn.ModuleList:
   - Chains N identical Transformer Blocks sequentially.
   - Outputs of layer l serve as inputs to layer l+1, enabling hierarchical context extraction.
   - nn.ModuleList properly registers submodules for GPU memory transfers (.to(device)) and autograd tracking.

3. Final RMSNorm:
   - Normalizes accumulated activations across residual additions before projecting to the vocabulary.

4. Language Model (LM) Head:
   - Projects final d_model representations into unnormalized vocabulary logits (B, S, Vocab_Size).
   - Facilitates next-token probability prediction over the entire vocabulary space.

5. Precomputed RoPE Buffer:
   - Caches rotary complex frequencies up to max_seq_len via register_buffer (persistent=False).
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from model.normalization import RMSNorm
from model.block import TransformerBlock
from model.rope import precompute_freqs_cis


# model/llama.py 상단부
@dataclass
class ModelArgs:
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    vocab_size: int = -1
    multiple_of: int = 256
    max_seq_len: int = 256
    max_batch_size: int = 1  # <-- [추가] 추론 시 사용할 최대 배치 크기

class LLaMA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # 1. Token Embedding Layer: Discrete token indices -> Dense latent vectors
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        # 2. Sequential Transformer Blocks: Stacked decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=args.dim,
                    n_heads=args.n_heads,
                    multiple_of=args.multiple_of,
                    max_batch_size=args.max_batch_size,  # <-- [추가]
                    max_seq_len=args.max_seq_len,        # <-- [추가]
                )
                for _ in range(args.n_layers)
            ]
        )
        # Remove this redundant line as the list comprehension already handles layer creation
        

        # 3. Final Pre-Head Normalization: Stabilizes variance after multi-layer residual accumulation
        self.norm = RMSNorm(dim=args.dim)

        # 4. LM Projection Head: Projects hidden states to vocabulary logits (bias=False for parameter efficiency)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 5. Precompute RoPE Frequency Buffer:
        # head_dim specifies the 2D subspace dimensionality for rotational positioning
        head_dim = args.dim // args.n_heads
        freqs_cis = precompute_freqs_cis(dim=head_dim, end=args.max_seq_len)
        # register_buffer ensures freqs_cis moves seamlessly with the model across devices without tracking gradients
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,        # Starting position for RoPE frequency slicing
        use_cache: bool = False,   # Whether to utilize the KV cache for faster autoregressive decoding
    ) -> torch.Tensor:
        _batch_size, seq_len = tokens.shape

        # Step 1: Token Embedding Lookup
        h = self.tok_embeddings(tokens)

        # Step 2: Slice frequencies for current token segment from start_pos
        # (내부에서 start_pos = 0 선언했던 두 줄 삭제 완료)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        # Step 3: Sequentially pass representations through all stacked Transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cis, start_pos=start_pos, use_cache=use_cache)

        # Step 4: Apply final normalization
        h = self.norm(h)

        # Step 5: Project to unnormalized vocabulary logits
        logits = self.output(h)

        return logits

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model configuration
    args = ModelArgs(
        dim=64,
        n_layers=4,
        n_heads=4,
        vocab_size=1000,
        max_seq_len=32
    )

    # Initialize complete architecture and push to accelerator
    model = LLaMA(args).to(device)

    # Generate synthetic token sequence: (Batch=2, Seq_Len=8)
    tokens = torch.randint(0, args.vocab_size, (2, 8), device=device)

    # End-to-end forward inference pass
    logits = model(tokens)

    print("입력 토큰 형태:", tokens.shape)
    print("최종 로짓 형태:", logits.shape)
    print("전체 LLaMA 아키텍처 End-to-End 파이프라인 검증 완료!")