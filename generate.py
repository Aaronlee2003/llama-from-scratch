"""
Day 10: Autoregressive Text Generation using KV Cache.
Leverages pre-allocated KV Cache and start_pos indexing to generate tokens in O(1) compute per step.
"""

import torch
from dataset import CharTokenizer
from model.lamma import LLaMA, ModelArgs

#this model read words at onmce this word of vector  k,v fill in cache
#and just input previous leeter and add the letter using cache previoud words and attention
@torch.no_grad()
def generate(
    model: LLaMA,
    prompt: str,
    tokenizer: CharTokenizer,
    max_gen_len: int = 50,
    device: torch.device = torch.device("cpu"),
) -> str:
    model.eval()

    # 1. 프롬프트 인코딩
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        raise ValueError("프롬프트는 최소 1글자 이상이어야 합니다.")

    generated = list(prompt_tokens)
    prompt_len = len(prompt_tokens)

    # -------------------------------------------------------------
    # 2. Prefill 단계: 프롬프트 전체를 한 번에 밀어넣어 캐시 채우기
    # -------------------------------------------------------------
    x_prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    # start_pos=0에서 시작, use_cache=True로 프롬프트의 KV를 캐시 버퍼에 기록
    logits = model(x_prompt, start_pos=0, use_cache=True)

    # 프롬프트의 마지막 토큰 위치에서 다음 토큰 1개 결정 (Greedy)
    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
    generated.append(next_token)

    # -------------------------------------------------------------
    # 3. Decode 루프: 새로 생성된 토큰 1개만 넣으며 1글자씩 추가
    # -------------------------------------------------------------
    # 다음 토큰이 들어갈 위치는 프롬프트 길이(인덱스)부터 시작
    for cur_pos in range(prompt_len, prompt_len + max_gen_len - 1):
        # 방금 생성된 토큰 '딱 1개'만 입력 텐서로 구성 (Shape: (1, 1))
        x_curr = torch.tensor([[next_token]], dtype=torch.long, device=device)

        # 모델 호출: seq_len=1, 현재 위치(cur_pos), 캐시 활성화
        logits = model(x_curr, start_pos=cur_pos, use_cache=True)

        # 다음 토큰 선택
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        generated.append(next_token)

    return tokenizer.decode(generated)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"추론 디바이스: {device}")

    # 학습용 말뭉치 및 토크나이저 설정
    corpus = "Machine learning research engineering requires deep mathematical intuition."
    tokenizer = CharTokenizer(corpus)

    # 모델 생성 (Day 9에서 학습한 가중치를 쓰거나 테스트용으로 초기화)
    args = ModelArgs(
        dim=128,
        n_layers=4,
        n_heads=4,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=256,
        max_batch_size=1,
    )
    model = LLaMA(args).to(device)

    # 추론 테스트
    prompt_text = "Machine"
    print(f"입력 프롬프트: '{prompt_text}'")

    generated_text = generate(
        model=model,
        prompt=prompt_text,
        tokenizer=tokenizer,
        max_gen_len=40,
        device=device,
    )
    print(f"생성 결과: {generated_text}")