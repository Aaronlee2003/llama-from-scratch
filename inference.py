"""
Day 7: Autoregressive Text Generation with Temperature, Top-K, and Top-P (Nucleus) Sampling.

Generation Pipeline:
1. Pass the evolving token sequence into the LLaMA backbone.
2. Extract unnormalized logits strictly from the terminal position: logits[:, -1, :].
3. Apply Temperature Scaling (z / T) to sharpen or flatten the categorical probability landscape.
4. Apply Top-K truncation to eliminate low-probability trailing noise.
5. Apply Top-P (Nucleus) truncation dynamically over the cumulative distribution function (CDF).
6. Softmax re-normalization over the truncated distribution.
7. Sample token indices via categorical multinomial sampling (or greedy argmax if T=0).
8. Concatenate the sampled token along the sequence dimension and iterate autoregressively.
"""

import torch
import torch.nn.functional as F
from model.lamma import LLaMA, ModelArgs


def sample_top_p_top_k(logits: torch.Tensor, temperature: float = 0.8, top_p: float = 0.9, top_k: int = 50) -> torch.Tensor:
    """
    Applies temperature scaling, Top-K, and Top-P (Nucleus) filtering to next-token logits.
    
    Args:
        logits: Unnormalized predictions of shape (Batch, Vocab_Size).
        temperature: Controls output entropy. T -> 0 yields deterministic greedy search, 
                     while T > 1 flattens the distribution, encouraging lexical variety.
        top_p: Cumulative probability threshold for nucleus filtering (0 < top_p < 1).
        top_k: Truncation threshold retaining only the top K candidate tokens.
    
    Returns:
        Sampled token index tensor of shape (Batch, 1).
    """
    # 1. Deterministic Greedy Search: Directly extract argmax mode when temperature collapses to zero
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. Temperature Scaling: Scales variance of logit distribution
    # Lower temperature amplifies the margin between peak and tail; higher temperature smooths it
    logits = logits / temperature

    # 3. Top-K Truncation: Restrict candidate pool to the highest K logits
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safeguard against top_k exceeding vocabulary bounds
        k_values, _ = torch.topk(logits, top_k, dim=-1)
        k_threshold = k_values[:, -1].unsqueeze(-1)  # Lowest acceptable logit score within top-k
        # Mask all subordinate logits falling below the cutoff threshold to -inf
        logits = torch.where(logits < k_threshold, torch.tensor(float("-inf"), device=logits.device), logits)

    # 4. Top-P (Nucleus) Truncation: Dynamically adapt candidate pool based on cumulative probability mass
    if 0.0 < top_p < 1.0:
        # Sort logits descending to construct a proper empirical CDF
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Flag candidates whose cumulative mass strictly exceeds the top_p boundary
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right by one position to retain the boundary-crossing token inside the nucleus
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False  # Guarantee that the top-1 candidate is never discarded

        # Scatter the boolean removal mask back to the original unshuffled vocabulary positions
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = float("-inf")

    # 5. Probability Re-normalization & Categorical Sampling
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)  # Stochastic sampling according to normalized weights
    return next_token


@torch.no_grad()
def generate(model: LLaMA, prompt_tokens: torch.Tensor, max_gen_len: int = 15, temperature: float = 0.8, top_p: float = 0.9, top_k: int = 50) -> torch.Tensor:
    """
    Autoregressive inference loop generating tokens step-by-step.
    """
    model.eval()  # Disable non-deterministic behaviors like dropout
    tokens = prompt_tokens.clone()  # Base prompt initialization

    for _ in range(max_gen_len):
        # Prevent context window overflow against precomputed RoPE frequency limits
        if tokens.shape[1] >= model.args.max_seq_len:
            break

        # 1. Forward Pass across current context sequence: (Batch, Current_Seq_Len, Vocab_Size)
        logits = model(tokens)

        # 2. Causal Slicing: Only the terminal logit vector conditions the upcoming token
        next_token_logits = logits[:, -1, :]

        # 3. Apply stochastic decoding pipeline
        next_token = sample_top_p_top_k(
            logits=next_token_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

        # 4. Autoregressive Concatenation: Append new token along sequence axis (dim=1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model architectural configuration
    args = ModelArgs(
        dim=64,
        n_layers=4,
        n_heads=4,
        vocab_size=1000,
        max_seq_len=64
    )
    model = LLaMA(args).to(device)

    # Prompt specification: Synthetic sequence of 3 discrete tokens (Batch=1, Seq_Len=3)
    prompt = torch.tensor([[42, 108, 9]], device=device)
    print("초기 프롬프트 토큰:", prompt.tolist())

    # Execute autoregressive sequence expansion
    generated = generate(
        model=model,
        prompt_tokens=prompt,
        max_gen_len=10,
        temperature=0.7,
        top_p=0.9,
        top_k=40
    )

    print("생성 완료 토큰 시퀀스:", generated.tolist())
    print("추가로 생성된 토큰 개수:", generated.shape[1] - prompt.shape[1])
    print("Day 7 자기회귀 텍스트 생성 파이프라인 검증 성공!")