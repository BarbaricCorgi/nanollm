import json
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        for i, token in enumerate(self.special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
    def _tokenize_text(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'([.!?,\'"()])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().split()
    
    def build_vocab(self, texts: List[str]):
        word_freq = Counter()
        
        for text in texts:
            tokens = self._tokenize_text(text)
            word_freq.update(tokens)
        
        most_common = word_freq.most_common(self.vocab_size - len(self.special_tokens))
        
        current_id = len(self.special_tokens)
        for word, _ in most_common:
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self._tokenize_text(text)
        ids = []
        
        if add_special_tokens:
            ids.append(self.bos_token_id)
        
        for token in tokens:
            ids.append(self.word_to_id.get(token, self.unk_token_id))
        
        if add_special_tokens:
            ids.append(self.eos_token_id)
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for id in ids:
            if id in self.id_to_word:
                token = self.id_to_word[id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        text = ' '.join(tokens)
        text = re.sub(r'\s+([.!?,\'"()])', r'\1', text)
        return text
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'word_to_id': self.word_to_id,
                'id_to_word': {str(k): v for k, v in self.id_to_word.items()},
                'special_tokens': self.special_tokens
            }, f, indent=2)
    
    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.word_to_id = data['word_to_id']
        self.id_to_word = {int(k): v for k, v in data['id_to_word'].items()}
        self.special_tokens = data['special_tokens']
    
    def __len__(self):
        return len(self.word_to_id)