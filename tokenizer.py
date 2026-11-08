#today we will implement a basic tokenizer for text processing
#bpe (Byte Pair Encoding) tokenizer implementation bpe thoery and mathmatical theory
"""byte level  fallback  
  prevent  out-of-vocabulary issues by using byte-level encoding dissemble  utf-8 bytes into subword units
    greedy merge criteriion   it count adjacent  two sysmboil  freqeunct
    (c1,c2) -> merge most frequent pair into a new symbol
    most frequent pairs are merged add to new vocabulary
    repeat until no more merges can be done or desired vocabulary size is reached
    
    
    
    
    """
    
from typing import List, Tuple, Dict

from matplotlib import text
class BPETokenizer:
    def __init__(self):
        #general word level tokenizater  new data  map to unkown toekn happen loss
        #according to  basic dic size  start with 256   new lanuage  lossless compression possible
       
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}#map to 0~255 for basic UTF-8 bytes
        self.merges: Dict[Tuple[int, int], int] = {}#on training level find the new merges and update this dictionary
        self.special_tokens: Dict[str, int] = {"<pad>": 256, "<bos>": 257, "<eos>": 258}#allocation  <pad>, <bos>, <eos> tokens register token in the vocabulary
        
    def _get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        #seqence  x1,x2,,xn  calculate frequency of adjacent pairs this between two symbol  mutual information and transition probability can be derived from this
      counts: Dict[Tuple[int, int], int] = {}
      for pair in zip(ids, ids[1:]):#Group sequences with a one-cell gap to generate consecutive pairs of Bigrams (ids[0], ids[1], ids[1], ids[2], ...) in O(N) time complexity.
        counts[pair] = counts.get(pair, 0) + 1#using dictionary to count frequency of each adjacent pair
      return counts
    def _merge(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        #sequence compaction  and entropy reduction 
        #merge the most frequent pair into a new symbol
        #If the two symbols with the highest bond strength, A and B, are replaced by a single symbol C=AB, the total valid length L of the sequence is immediately shortened:
        #so sequnce size is reducedce so attention calculation is decreased
        """
        using pointer find the linear search 
        current index i  and i+1 is 
        If this pair exactly matches the target pair to merge, insert a new token with new_id and increment the pointer by 2.
        Otherwise, keep the current token and increment the pointer by 1.
        """
        
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    def train(self, text: str, vocab_size: int = 300):
        """
        unitl to reach the v target  whenever  loop   
        Greedyly select the optimal pair with the highest co-occurrence frequency:
         at text  reepaeat voca  natuaraly  merge one big token
        """
        assert vocab_size >= 259, "vocab_size는 기본 바이트 및 특수 토큰 수(259개)보다 커야 합니다."
        num_merges = vocab_size - len(self.vocab)

        tokens = list(text.encode("utf-8"))#Serialize the Nicode string into a list of UTF-8 byte integers (0–255).

        for i in range(num_merges):
         stats = self._get_stats(tokens)
         if not stats:
            break
        
        best_pair = max(stats, key=stats.get)#Extract the key tuple with the highest value (number of occurrences) from the frequency dictionary in $O(K)$ time.
        new_id = len(self.vocab)

        tokens = self._merge(tokens, best_pair, new_id)#두 토큰의 바이트 조각을 이어 붙여(Concatenate) 새로운 바이트 단어를 어휘 사전에 등록합니다
        self.merges[best_pair] = new_id
        self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
    def encode(self, text: str) -> List[int]:
     """ learning merge rule  according to the  learned sequence they have proirity
     By sequentially applying the highest-priority (most first-trained, most frequent) rule to any given input text, it achieves deterministic splitting, ensuring that the same word is consistently split into the same subwords

       _
    """
     tokens = list(text.encode("utf-8"))
     while len(tokens) >= 2:
        stats = self._get_stats(tokens)
        pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))#find the pair with the highest priority (lowest merge index)
        if pair not in self.merges:#in sequence no more merge possible for this pair stop the token split returnd id
            break
        new_id = self.merges[pair]
        tokens = self._merge(tokens, pair, new_id)
     return tokens
    def decode(self, ids: List[int]) -> str:
        """_
        decoding and incoding  between byte sequence and string have relation invertable
        bpe is make a token split the byte  token id are subsitution  origin bye block  and just concatenate them  recover origin utf-8 string
        """
        tokens_bytes = b"".join(
        self.vocab[idx] for idx in ids if idx not in self.special_tokens.values()
    )#concetate  all token of byte array to reconstruct the original byte sequence
        #recovered byte  stream are decode to utf-8 string if byte uncompletable or invalid, errors="replace" dont stop the error and deal with it gracefully
        return tokens_bytes.decode("utf-8", errors="replace")