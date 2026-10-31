

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
  
        
        