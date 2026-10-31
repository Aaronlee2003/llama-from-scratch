import torch
import torch.nn as nn


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
      "computer angular each position frequencies"
      "dim:head dimension,must be even so can match sin and cos"    
      "end: max sequence length to precompute"
      
      
      freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
      #mapping to complex plane 2 칸마다 인덱스 생성 i가 작을 수록  각이 크게 회전 하고 i가 커질수록 회전이 느려짐   다양한 주기를 줘서 위치를 촘촘히 만듬
      #2로 나누어 64를 32로 줄여서 32개의 주기를 만들어줌
  
      t = torch.arange(end, device=freqs.device)
      #단어내 인덱스 생성
      freqs = torch.outer(t, freqs).float()
      # t,feqs 행렬을 만듬 위치 인덱스와 차원별 주파수를 곱해서 m번쨰 위치에서 i번쨰 차원이 얼마나 회전해야하는지 계산
      freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
      #r,세타를 받아서  복서 텐서를 만듬  반지름은  1, 각도는 freqs로 주어짐,각위치와 차원에 대한 회전값을 복소수로 만들어줌
      return freqs_cis
     
     
     
     
     
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # 1. 마지막 차원을 2개씩 묶어 복소수로 변환: (..., head_dim // 2)
    # x shape: (batch_size, seq_len, n_heads, head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # 2. 현재 입력의 시퀀스 길이에 맞춰 freqs_cis 슬라이싱
    seq_len = x.shape[1]
    freqs_cis = freqs_cis[:seq_len]

    # 3. 브로드캐스팅 차원 맞추기: (seq_len, head_dim // 2) -> (1, seq_len, 1, head_dim // 2)
    # n_heads 축(2번 축)에 1을 넣어주어야 x_complex (B, S, H, D)와 정상 곱셈됩니다.
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # 4. 회전 복소수 곱 적용
    x_rotated = x_complex * freqs_cis

    # 5. 다시 실수 텐서로 복원 후 원래 shape으로 되돌림
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.type_as(x)

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # (배치=2, 문장길이=8, 차원=64) 더미 Query/Key 텐서
    x = torch.randn(2, 8, 64, device=device)

    # 1. 최대 문장 길이 16까지 회전 각도 미리 계산
    freqs_cis = precompute_freqs_cis(dim=64, end=16).to(device)

    # 2. 위치 임베딩 회전 적용
    x_rotated = apply_rotary_emb(x, freqs_cis)

    print("원본 형태:", x.shape)
    print("회전 후 형태:", x_rotated.shape)
    print("성공적으로 위치 정보가 회전 변환되었습니다!") 