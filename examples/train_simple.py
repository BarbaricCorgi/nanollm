#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanollm import NanoLLM, SimpleTokenizer, Trainer, ModelConfig
from nanollm.trainer import TextDataset
import torch


def load_sample_data():
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Neural networks are inspired by the human brain.",
        "Deep learning has revolutionized computer vision and natural language processing.",
        "The transformer architecture has become the foundation of modern language models.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Training large language models requires significant computational resources.",
        "Fine-tuning pre-trained models is an effective transfer learning technique.",
        "Gradient descent is an optimization algorithm used to train neural networks.",
    ] * 100
    
    return sample_texts


def main():
    print("NanoLLM Training Example")
    print("=" * 50)
    
    config = ModelConfig(
        vocab_size=5000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        ff_size=1024,
        max_seq_length=128,
        dropout=0.1,
    )
    
    print("Loading sample data...")
    texts = load_sample_data()
    
    train_texts = texts[:int(len(texts) * 0.8)]
    eval_texts = texts[int(len(texts) * 0.8):]
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Eval samples: {len(eval_texts)}")
    
    print("\nBuilding vocabulary...")
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(texts)
    print(f"Vocabulary size: {len(tokenizer)}")
    
    train_dataset = TextDataset(train_texts, tokenizer, max_length=config.max_seq_length)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_length=config.max_seq_length)
    
    print("\nInitializing model...")
    model = NanoLLM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=16,
        learning_rate=5e-4,
        num_epochs=3,
        warmup_steps=100,
        save_steps=500,
        eval_steps=200,
        device=str(device),
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print("\nTraining completed!")
    print("Model checkpoints saved in 'checkpoints' directory")


if __name__ == "__main__":
    main()