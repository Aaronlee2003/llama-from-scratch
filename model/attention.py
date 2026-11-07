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
   
   
   
   
   
   - GQA (Grouped Query Attention) allows the number of query heads to exceed the number of key-value heads.
   generalizes standard multi-head attention by decoupling the number of query heads from key-value heads.
   every token we havbe to sotre kv size memopry
   KV Cache Size (MHA)=2×B×S×H×d 
k
​	
 =2BSd
 s is increade 2k 4k 8k  this will happen model weight usage is samller than KV cache memory usage which grows linearly with sequence length. happen memory wall 
 so we maintain head size h  key and value decrease hkv size  head ratio $R = \frac{H}{H_{kv}}$, g∈{0,1,…,H 
kv
​	
 −1} 번째 KV 헤드는 다음 범위의 Query 헤드 집합과 공유됩니다:
 and cache mempory size is  1of r
 
 
 
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rope import apply_rotary_emb, precompute_freqs_cis


def repeat_kv(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
    """
    #(batch_size, seq_len, n_kv_heads, head_dim) -> (batch_size, seq_len, n_heads, head_dim)
    small quantity of key-value heads are repeated to match the number of attention heads.
    
    Previouis code is query head 8 must be key value head 8 
    cqg use less memory  just use 2 key-value heads.
    using 2 kv head  each 4 times duplicated to match 8 query heads.
    """
    if n_repeat == 1:# if size is standard, no repetition needed
        return x
    n_rep = n_repeat    
    bsz, slen, n_kv_heads, head_dim = x.shape# just read tensor  axis  present cache buffer (b, s, n_kv_heads, head_dim)
    return (
        x[:, :, :, None, :]# injection back  first index make a virtual head dimension
        .expand(bsz, slen, n_kv_heads, n_rep, head_dim)#after transformation(B, S, H_kv, n_rep, D_k)  new  two axis broadcasting until n_rep tiimes dont use phsycal tensor just deal stride meta data operation time o(1) 
        .reshape(bsz, slen, n_kv_heads * n_rep, head_dim)#flatten first axis and second axis to merge repeated heads into the head dimension can multiple query tensor use same key-value heads efficiently
    )
    

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
        self.wq = nn.Linear(dim, dim, bias=False)#query projection matrix    input vector  transformed into  all head space
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)#key,value projectio matrix  parameter matrixs  size decreased  weigght memory and multiple  decrease
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        # Output projection layer: Fuses concatenated head representations back into the residual stream
        self.wo = nn.Linear(dim, dim, bias=False)
        
        self.register_buffer(
            "cache_k",
            torch.zeros((max_batch_size, self.n_kv_heads, max_seq_len, self.head_dim)), #fixed cache head buffer size ex if we use just 4 head  all head is 32 but we only use 4 for key-value buffer use 1/8 of the total head memory
            persistent=False,#if we store  model.state_dict()  empty cache tensor  do not get saved so we can keep  light check point memory usage low
        )
        self.register_buffer(
            "cache_v",
            torch.zeros((max_batch_size, self.n_kv_heads, max_seq_len, self.head_dim)),
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
            self.cache_v[:batch_size, :, start_pos:end_pos] = xv#Leave the past tokens as they are, and slice and overwrite only the current input sequence segment with an O(1) index
            keys = self.cache_k[:batch_size, :, :end_pos]
            values = self.cache_v[:batch_size, :, :end_pos]
            keys = repeat_kv(keys, self.n_rep)#just store hkv state  if we enter the core attention computation temporerily expand  head dimension and just inner product with query
            values = repeat_kv(values, self.n_rep)
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

    x = torch.randn(2, 8, 64, device=device)
    head_dim = 64 // 4  # 16
    freqs_cis = precompute_freqs_cis(dim=head_dim, end=16).to(device)

    # Query 헤드는 4개, Key/Value 헤드는 2개인 GQA 인스턴스 생성 (캐시 메모리 50% 절감)
    gqa = MultiHeadAttention(dim=64, n_heads=4, n_kv_heads=2).to(device)

    out = gqa(x, freqs_cis)
    print("입력 형태:", x.shape)
    print("출력 형태:", out.shape)
    print("GQA 블록 연산 성공!")