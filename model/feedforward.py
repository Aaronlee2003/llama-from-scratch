import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None, multiple_of: int = 256):
        super().__init__()  # Initialize base class to register PyTorch module parameters

        # LLaMA hidden dimension scaling strategy:
        # Standard Transformers expand hidden dimension by 4x.
        # SwiGLU introduces three projection matrices (gate, up, down), inflating parameter count.
        # Scaling by 2/3 yields approximately (8/3) * dim, maintaining a similar parameter budget.
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            # Integer ceiling trick: Rounds up to the nearest multiple of 256 for GPU hardware tiling
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Gate projection: Computes gating coefficients to dynamically filter feature pathways
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)  # bias=False saves memory and stabilizes weight decay
        
        # Up projection: Projects raw representations into the expanded hidden space
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        
        # Down projection: Reverse-projects filtered features back to the original model dimension
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Project to hidden dimension and apply non-linear SiLU activation (gate score in range ~0 to 1)
        # Tensor Shape: (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, Hidden_Dim)
        gate = F.silu(self.w_gate(x))

        # 2. Linear projection preserving raw contextual information without activation
        # Tensor Shape: (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, Hidden_Dim)
        up = self.w_up(x)

        # 3. Hadamard product (element-wise multiplication):
        # Gate acts as a dynamic valve—values near 0 suppress noise, while larger values let features pass through
        h = gate * up  # Shape remains (Batch, Seq_Len, Hidden_Dim)

        # 4. Project back down to model dimension to support residual addition: x + FFN(x)
        # Tensor Shape: (Batch, Seq_Len, Hidden_Dim) -> (Batch, Seq_Len, Dim)
        return self.w_down(h)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dummy tensor: (Batch=2, Seq_Len=8, Dim=64)
    x = torch.randn(2, 8, 64, device=device)

    # Initialize FeedForward module and move weights to accelerator (Apple Silicon MPS or CPU)
    ffn = FeedForward(dim=64, hidden_dim=None).to(device)

    out = ffn(x)
    print("입력 형태:", x.shape)
    print("출력 형태:", out.shape)
    print("SwiGLU 피드포워드 블록 연산 성공!")
        
        
        