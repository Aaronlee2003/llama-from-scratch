import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 학습 가능한 가중치 파라미터 (gamma)
        self.weight = nn.Parameter(torch.ones(dim))#reset all elements to 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch, Seq_Len, Dim)에서 마지막 Dim 축으로 RMS 계산
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # (배치=2, 문장길이=8, 차원=64) 더미 데이터로 테스트
    x = torch.randn(2, 8, 64, device=device)
    norm = RMSNorm(dim=64).to(device)
    
    
    
    output = norm(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("successfully normalized the input tensor using RMSNorm.")
        
        