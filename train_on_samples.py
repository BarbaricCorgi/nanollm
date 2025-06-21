#!/usr/bin/env python3

import json
import os
import sys
from datetime import datetime
import torch
from nanollm import NanoLLM, SimpleTokenizer, Trainer, ModelConfig
from nanollm.trainer import TextDataset

def load_sample_texts(file_path):
    """Load sample texts from JSON file."""
    print(f"Loading texts from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    print(f"Loaded {len(texts)} texts")
    return texts

def main():
    # Check if sample data exists
    data_files = ["data/combined_texts.json", "data/literature_texts.json"]
    data_file = None
    
    for file in data_files:
        if os.path.exists(file):
            data_file = file
            break
    
    if data_file is None:
        print("Sample data not found!")
        print("Please run: python3 download_sample_data.py")
        sys.exit(1)
    
    # Load texts
    texts = load_sample_texts(data_file)
    
    # Split into train/eval
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]
    
    print(f"\nDataset statistics:")
    print(f"Training samples: {len(train_texts)}")
    print(f"Evaluation samples: {len(eval_texts)}")
    
    # Model configuration
    config = ModelConfig(
        vocab_size=10000,
        hidden_size=384,
        num_layers=6,
        num_heads=6,
        ff_size=1536,
        max_seq_length=256,
        dropout=0.1,
    )
    
    # Build tokenizer vocabulary
    print("\nBuilding vocabulary...")
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(texts)
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Save tokenizer
    os.makedirs("models", exist_ok=True)
    tokenizer.save("models/sample_tokenizer.json")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TextDataset(train_texts, tokenizer, max_length=config.max_seq_length)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_length=config.max_seq_length)
    
    # Initialize model
    print("\nInitializing model...")
    model = NanoLLM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Training configuration
    batch_size = 16 if device.type == "cuda" else 8
    learning_rate = 5e-4
    num_epochs = 10  # More epochs for smaller dataset
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        warmup_steps=200,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        checkpoint_dir="checkpoints/samples",
        save_steps=500,
        eval_steps=250,
        device=str(device),
    )
    
    # Start training
    print("\nStarting training on sample data...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Checkpoint directory: checkpoints/samples")
    print("-" * 50)
    
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    
    print("\nTraining completed!")
    print(f"Total training time: {end_time - start_time}")
    print(f"Model checkpoints saved in 'checkpoints/samples'")
    
    # Save final model
    trainer.save_checkpoint("final")
    
    print("\nTo generate text with the trained model, run:")
    print("python3 generate_samples.py")

if __name__ == "__main__":
    main()