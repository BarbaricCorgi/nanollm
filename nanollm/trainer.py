import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List, Callable
from tqdm import tqdm
import os
import json
from .model import NanoLLM
from .config import ModelConfig
from .tokenizer import SimpleTokenizer


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    
    input_ids = []
    labels = []
    attention_mask = []
    
    for item in batch:
        seq_len = len(item["input_ids"])
        padding_len = max_len - seq_len
        
        input_ids.append(
            torch.cat([item["input_ids"], torch.zeros(padding_len, dtype=torch.long)])
        )
        labels.append(
            torch.cat([item["labels"], torch.full((padding_len,), -100, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([torch.ones(seq_len), torch.zeros(padding_len)])
        )
    
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


class Trainer:
    def __init__(
        self,
        model: NanoLLM,
        tokenizer: SimpleTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        save_steps: int = 1000,
        eval_steps: int = 500,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        self.checkpoint_dir = checkpoint_dir
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )
        
        if eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=2,
                pin_memory=True,
            )
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        total_steps = len(self.train_dataloader) * num_epochs // gradient_accumulation_steps
        self.scheduler = self._get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def _get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train(self):
        self.model.train()
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    if self.global_step % self.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")
                    
                    if self.eval_dataset is not None and self.global_step % self.eval_steps == 0:
                        eval_loss = self.evaluate()
                        self.model.train()
                        
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint("best")
                
                epoch_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
            
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            self.save_checkpoint(f"epoch-{epoch + 1}")
    
    def evaluate(self):
        if self.eval_dataset is None:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Evaluation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, name: str):
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "config": self.model.config.__dict__,
        }, os.path.join(checkpoint_path, "model.pt"))
        
        self.tokenizer.save(os.path.join(checkpoint_path, "tokenizer.json"))
        
        with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
            json.dump({
                "global_step": self.global_step,
                "best_eval_loss": self.best_eval_loss,
                "learning_rate": self.scheduler.get_last_lr()[0],
            }, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        model_path = os.path.join(checkpoint_path, "model.pt")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_eval_loss = checkpoint["best_eval_loss"]
        
        self.tokenizer.load(os.path.join(checkpoint_path, "tokenizer.json"))
        
        print(f"Checkpoint loaded from {checkpoint_path}")