import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.size()
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        present_key_value = (k, v)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.ff_size, bias=config.use_bias)
        self.fc2 = nn.Linear(config.ff_size, config.hidden_size, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)
        
        if config.activation == "gelu":
            self.activation = F.gelu
        elif config.activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {config.activation}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output, present_key_value = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = residual + self.dropout(attn_output)
        
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + self.dropout(ffn_output)
        
        return hidden_states, present_key_value


class NanoLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        if config.position_encoding == "learned":
            self.position_embeddings = nn.Embedding(config.max_seq_length, config.hidden_size)
        else:
            self.position_embeddings = PositionalEncoding(config.hidden_size, config.max_seq_length)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        if config.tie_embeddings:
            self.lm_head = self.embeddings
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def get_input_embeddings(self):
        return self.embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings = value
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> dict:
        batch_size, seq_length = input_ids.shape
        
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * self.config.num_layers
        else:
            past_length = past_key_values[0][0].size(2)
        
        position_ids = torch.arange(
            past_length, seq_length + past_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        hidden_states = self.embeddings(input_ids)
        
        if isinstance(self.position_embeddings, nn.Embedding):
            position_embeds = self.position_embeddings(position_ids)
            hidden_states = hidden_states + position_embeds
        else:
            hidden_states = self.position_embeddings(hidden_states)
        
        hidden_states = self.dropout(hidden_states)
        
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        present_key_values = []
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
            )
            if use_cache:
                present_key_values.append(present_key_value)
        
        hidden_states = self.ln_f(hidden_states)
        
        if self.config.tie_embeddings:
            logits = F.linear(hidden_states, self.embeddings.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": present_key_values if use_cache else None,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        past_key_values = None
        
        with torch.no_grad():
            while input_ids.shape[1] < max_length:
                outputs = self.forward(
                    input_ids=input_ids[:, -1:] if past_key_values is not None else input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                logits = outputs["logits"][:, -1, :] / temperature
                past_key_values = outputs["past_key_values"]
                
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                if eos_token_id is not None:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                
                if eos_token_id is not None:
                    unfinished_sequences = unfinished_sequences.mul(
                        next_tokens.ne(eos_token_id).long()
                    )
                    if unfinished_sequences.max() == 0:
                        break
        
        return input_ids