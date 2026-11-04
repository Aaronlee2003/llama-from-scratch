"""
Day 10: Autoregressive Text Generation using KV Cache.
Leverages pre-allocated KV Cache and start_pos indexing to generate tokens in O(1) compute per step.
"""
#day 11   decoding way  tiny shakespeare for pretraing  expansion 
# inference upgrade temperature sampling and top-k sampling
# mathmathical logic logtics'=logit/t if  t<1.0 coreect word choose and t>1.0 distribution flatten  so can choose more creative word
#top-p smapling  After sorting in order of probability, keep only the top candidate groups with a cumulative probability of p (e.g., 0.9), discard the rest as -∞, and then torch.Sample using multinomial.
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
    next_token = sample_top_p(logits[:, -1, :], temperature=0.7, top_p=0.9)
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
def sample_top_p(logits: torch.Tensor, temperature: float = 0.8, top_p: float = 0.9) -> int:
    # temperature scaling: l and top logtit flitering and probablity token sampling
     logits = logits / temperature #grouping logits by temperature for sampling
     probs = torch.softmax(logits, dim=-1)
     sorted_probs, sorted_indices = torch.sort(probs, descending=True)# sort probabilities for top-p filtering and can restore original order later
     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)#sum from highest probability to current token
     sorted_indices_to_remove = cumulative_probs > top_p# if top-p is 0.8 if total summurize is 0.85 include third word it must be include
     #if we didnt shift  and apply the mask  this value deleted
     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()# this make lower probablity token make 0 tottal probability sum is under 1
     sorted_indices_to_remove[..., 0] = False
     sorted_probs[sorted_indices_to_remove] = 0.0
     sorted_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)
     next_token_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)#just pick one token based on filtered probabilities
     next_token = torch.gather(sorted_indices, -1, next_token_sorted_idx)

     return next_token.item()


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. train.py가 저장한 파일 불러오기
    checkpoint = torch.load("checkpoint.pt", map_location=device)
    args = checkpoint["args"]
    chars = checkpoint["chars"]

    # 2. 토크나이저 및 모델 가중치 복원
    tokenizer = CharTokenizer("".join(chars))
    args.max_batch_size = 1  # 1개씩 생성하므로
    model = LLaMA(args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 3. 생성 호출
    output = generate(model, "KING:", tokenizer, max_gen_len=200, device=device)
    print(output)