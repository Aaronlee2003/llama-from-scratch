

#just repeart number tneosr  real text make  number sequnece  the minin token nizer and 
# model can understand(overfitting) short sentences  and recover the  text

"""Character-level Tokenizer for Toy Transformer Training.
Maps unique characters to integer indices and reconstructs text from predictions.
"""
#character-level tokenizer
#it can map characters to integers and back
#with out complexity libraray    directly implemented  dictionary based tokenizer
#overfitting model very tiny sample make loss  0
#if short sentences  recover  correctly forward&backward propagation make perfect predictions
from typing import List

import torch



class CharTokenizer:
    def __init__(self, text: str):
        self.chars=sorted(list(set(text)))#removwe duplicated character and exract unique characters
        self.vocab_size = len(self.chars)#choose unique characters as vocabulary
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}#this map character key number is value dictionary o(1)
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}##number id key and find the origiin character reverse diciotnary
    def encode(self, s: str) -> List[int]:
        return [self.char2idx[ch] for ch in s]
    #iteration each charactert s  through char2idx dictionary to get the corresponding integer indices
    def decode(self, indices: List[int]) -> str:
        return "".join([self.idx2char[idx] for idx in indices])
    #iterartiion each iterger  idx2char  combination to reconstruct the original text
  
        
    @staticmethod
    def get_batch(
        data: torch.Tensor, batch_size: int, block_size: int, device: torch.device
    ):
    # 1. pick the random batch_size count
     ix = torch.randint(len(data) - block_size, (batch_size,))#limited the index size to ensure we can slice block_size length without overflow

    # 2. from start indext, slice block_size length to form batch (Stack)
     x = torch.stack([data[i : i + block_size] for i in ix])#model input sequence size is batch_size x block_size
     y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
     #autogressive target: y is the next token sequence for x
     #if x is t times token  y is the next token sequence for each position in x
    # 3. move tensors to the device (MPS / GPU / CPU) for model computation
     return x.to(device), y.to(device)   