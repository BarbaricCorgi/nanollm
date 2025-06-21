from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ff_size: int = 2048
    max_seq_length: int = 512
    dropout: float = 0.1
    
    layer_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    use_bias: bool = True
    
    activation: str = "gelu"
    position_encoding: str = "learned"
    
    gradient_checkpointing: bool = False
    
    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        self.head_dim = self.hidden_size // self.num_heads