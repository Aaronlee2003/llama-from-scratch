"""
Day 9: Real Text Overfitting Test with Character-Level Tokenizer.
Verifies that the LLaMA model can memorize and reconstruct input text sequences.
"""

import torch
import torch.nn.functional as F
from dataset import CharTokenizer
from model.lamma import LLaMA, ModelArgs


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Corpus Preparation & Tokenizer Setup
    raw_text = "Machine learning research engineering requires deep mathematical intuition."
    tokenizer = CharTokenizer(raw_text)
    token_ids = tokenizer.encode(raw_text)
    
    # Shape: (1, Seq_Len)
    tokens_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    seq_len = tokens_tensor.size(1)

    print(f"Corpus: '{raw_text}'")
    print(f"Vocab Size: {tokenizer.vocab_size} | Sequence Length: {seq_len}")

    # 2. Model Configuration matched to the Tokenizer Vocabulary
    args = ModelArgs(
        dim=128,
        n_layers=4,
        n_heads=4,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len + 1
    )
    model = LLaMA(args).to(device)
    model.train()

    # 3. Next-Token Shift: [t_1 ... t_{N-1}] -> [t_2 ... t_N]
    inputs = tokens_tensor[:, :-1]
    targets = tokens_tensor[:, 1:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # 4. Training Loop (Memorization Test)
    steps = 150
    print("\nBeginning Overfitting Optimization...")
    for step in range(1, steps + 1):
        optimizer.zero_grad()

        logits = model(inputs)
        loss = compute_loss(logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 30 == 0 or step == 1:
            print(f"Step [{step:3d}/{steps}] | Loss: {loss.item():.4f}")

    # 5. Reconstruction Verification (Greedy Generation from the first character)
    model.eval()
    with torch.no_grad():
        # Start prompt with the first character
        generated_ids = [token_ids[0]]
        
        for _ in range(seq_len - 1):
            curr_input = torch.tensor([generated_ids], dtype=torch.long, device=device)
            out_logits = model(curr_input)
            
            # Pick highest probability token for next step
            next_token = torch.argmax(out_logits[:, -1, :], dim=-1).item()
            generated_ids.append(next_token)

        reconstructed_text = tokenizer.decode(generated_ids)
        print("\n--- Generation Test ---")
        print(f"Original Text     : {raw_text}")
        print(f"Reconstructed Text: {reconstructed_text}")

        if reconstructed_text == raw_text:
            print("성공: 모델이 문장을 완벽히 복원(Memorization)했습니다!")
        else:
            print("주의: 일부 토큰 불일치 (학습 스텝을 늘려보세요).")
            
    # -------------------------------------------------------------
    # 6. 체크포인트 저장 (가중치 + 토크나이저 메타데이터)
    # -------------------------------------------------------------
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "chars": tokenizer.chars,  # previous word list for decoding
        "args": args,              #model hyperparameters for reconstruction
    }
    torch.save(checkpoint, "checkpoint.pt")
    print("\n모델 가중치가 'checkpoint.pt'로 성공적으로 저장되었습니다!")

    
if __name__ == "__main__":
    train()