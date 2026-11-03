"""
Day 9: Real Text Overfitting Test with Character-Level Tokenizer.
Verifies that the LLaMA model can memorize and reconstruct input text sequences.
"""

"""
day 12 today math thoery   if we given sequence of tokens, text joint probability can be modeled as product of conditional probabilities of each token given previous tokens.
input xt   xt+1 is mathced to yt  so model learns to predict the next token given the previous tokens.
#nll,cross-entropy loss logit  projection to probability sapce
#cross-entropy loss is commonly used for next-token prediction tasks in language modeling.
# gradient decent p target-1<0  according to gradient descent rule, the model parameters are updated to minimize the loss.
#p(i)>0  Drag the wrong-answer robots down.
gradient cliffing  is used to prevent exploding gradients by capping the maximum value of the gradient norm.
# adam  first  gradeint direction and second moment(gradeint sqaure and  adaptive learning rate) and l2 regularization and
seperate decouppled weight decay



"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import CharTokenizer, get_batch
from model.lamma import LLaMA, ModelArgs


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    # 1. 코퍼스 로드 및 토크나이저 초기화
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    print(f"데이터셋 전체 문자 수: {len(text):,}개")  # exract alphabet and number of unique characters
    print(f"고유 어휘 사전 크기 (vocab_size): {tokenizer.vocab_size}개")

    # 전체 데이터를 정수 텐서로 변환
    data = torch.tensor(
        tokenizer.encode(text), dtype=torch.long
    )  # all text  incoding to integer tensor
    n = int(
        0.9 * len(data)
    )  # prevent data leak  set the standard split for train and validation data
    train_data = data[:n]  # 90% for training
    val_data = data[n:]  # 10% for validation

    # 2. 하이퍼파라미터 설정
    batch_size = 32  # number of sequences per batch
    block_size = 64  # (max_seq_len)
    max_iters = 1500  # total number of training steps
    eval_interval = 250  # validation loss evaluation interval
    learning_rate = 3e-4  # adamw learning rate

    # 3. LLaMA 모델 초기화
    args = ModelArgs(  # make model instance with specified arguments weight tensor batch on accelation memory
        dim=128,
        n_layers=4,
        n_heads=4,
        vocab_size=tokenizer.vocab_size,
        multiple_of=256,
        max_seq_len=block_size,
        max_batch_size=batch_size,
    )
    model = LLaMA(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate
    )  # weight decay optimization
    criterion = (
        nn.CrossEntropyLoss()
    )  # next Declare the cross-entropy loss function for measuring token classification error.

    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}개")
    print("사전 학습(Pre-training) 시작...")

    model.train()
    for step in range(1, max_iters + 1):
        # 훈련 미니배치 샘플링: x, y 형태는 (32, 64)
        x_batch, y_batch = get_batch(
            train_data, batch_size, block_size, device
        )  # just  pick a random batch from the training data

        # Forward Pass (사전 학습 단계이므로 start_pos=0, use_cache=False)
        logits = model(
            x_batch, start_pos=0, use_cache=False
        )  # at the learning time all sequence toekn are calculated using the model without caching

        # Loss 계산: (Batch * Seq_Len, Vocab_Size) vs (Batch * Seq_Len)
        loss = criterion(
            logits.view(-1, args.vocab_size), y_batch.view(-1)
        )  # flatten and 2D to 1D for cross-entropy calculation calculation between predicted logits and target tokens

        # Backward & Optimize
        optimizer.zero_grad(
            set_to_none=True
        )  # empty previous step  gradients before backward pass
        loss.backward()  # using backpropagation to compute gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # prevent gradient explosion limited l2 norm of gradients 1.0
        optimizer.step()

        # 주기적 손실 로깅
        if step % eval_interval == 0 or step == max_iters:
            model.eval()  # turn for evaluation mode (disables dropout, etc.)
            with torch.no_grad():
                vx, vy = get_batch(val_data, batch_size, block_size, device)
                v_logits = model(
                    vx, start_pos=0, use_cache=False
                )  # vx,vy are validation batch inputs and targets
                val_loss = criterion(
                    v_logits.view(-1, args.vocab_size), vy.view(-1)
                )
            print(
                f"Step {step:4d}/{max_iters} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}"
            )
            model.train()

    # 4. 체크포인트 저장 (generate.py에서 사용할 가중치 및 메타데이터)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "chars": tokenizer.chars,
        "args": args,
    }
    torch.save(checkpoint, "checkpoint.pt")
    print("\n'checkpoint.pt' 저장 완료! 사전 학습이 성공적으로 종료되었습니다.")


if __name__ == "__main__":
    train()