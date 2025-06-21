import torch
from typing import List, Optional, Union
from .model import NanoLLM
from .config import ModelConfig
from .tokenizer import SimpleTokenizer
import os


class TextGenerator:
    def __init__(
        self,
        model: NanoLLM,
        tokenizer: SimpleTokenizer,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None):
        model_path = os.path.join(checkpoint_path, "model.pt")
        tokenizer_path = os.path.join(checkpoint_path, "tokenizer.json")
        
        checkpoint = torch.load(model_path, map_location="cpu")
        config_dict = checkpoint["config"].copy()
        if "head_dim" in config_dict:
            del config_dict["head_dim"]
        config = ModelConfig(**config_dict)
        
        model = NanoLLM(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        tokenizer = SimpleTokenizer()
        tokenizer.load(tokenizer_path)
        
        return cls(model, tokenizer, device)
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: int = 100,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        do_sample: bool = True,
    ) -> Union[str, List[str]]:
        
        if isinstance(prompt, str):
            prompts = [prompt]
            single_prompt = True
        else:
            prompts = prompt
            single_prompt = False
        
        if num_return_sequences > 1:
            prompts = prompts * num_return_sequences
        
        input_ids_list = []
        for p in prompts:
            tokens = self.tokenizer.encode(p, add_special_tokens=True)
            input_ids_list.append(torch.tensor(tokens, dtype=torch.long))
        
        max_prompt_length = max(len(ids) for ids in input_ids_list)
        padded_input_ids = []
        attention_masks = []
        
        for ids in input_ids_list:
            padding_length = max_prompt_length - len(ids)
            padded_ids = torch.cat([
                torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long),
                ids
            ])
            attention_mask = torch.cat([
                torch.zeros(padding_length, dtype=torch.long),
                torch.ones(len(ids), dtype=torch.long)
            ])
            padded_input_ids.append(padded_ids)
            attention_masks.append(attention_mask)
        
        input_ids = torch.stack(padded_input_ids).to(self.device)
        attention_mask = torch.stack(attention_masks).to(self.device)
        
        if max_new_tokens is not None:
            max_length = input_ids.shape[1] + max_new_tokens
        
        with torch.no_grad():
            if do_sample:
                generated = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                generated = self._greedy_decode(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                )
        
        generated_texts = []
        for seq in generated:
            seq = seq.tolist()
            if self.tokenizer.pad_token_id in seq:
                first_pad = seq.index(self.tokenizer.pad_token_id)
                seq = seq[first_pad + 1:]
            
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            generated_texts.append(text)
        
        if single_prompt and num_return_sequences == 1:
            return generated_texts[0]
        elif single_prompt:
            return generated_texts
        else:
            return [generated_texts[i:i+num_return_sequences] 
                    for i in range(0, len(generated_texts), num_return_sequences)]
    
    def _greedy_decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
    ) -> torch.Tensor:
        
        past_key_values = None
        
        for _ in range(max_length - input_ids.shape[1]):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids[:, -1:] if past_key_values is not None else input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]
            
            next_token_id = torch.argmax(logits, dim=-1)
            
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
            ], dim=-1)
            
            if (next_token_id == self.tokenizer.eos_token_id).all():
                break
        
        return input_ids
    
    def score(self, text: str) -> float:
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(self.device)
        labels = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
        
        return -loss.item()