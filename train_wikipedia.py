#!/usr/bin/env python3

import json
import os
import sys
from datetime import datetime
import torch
from nanollm import NanoLLM, SimpleTokenizer, Trainer, ModelConfig
from nanollm.trainer import TextDataset

def load_wikipedia_articles(file_path):
    """Load Wikipedia articles from JSON file."""
    print(f"Loading Wikipedia articles from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    print(f"Loaded {len(articles)} articles")
    return articles

def prepare_training_texts(articles, max_length=512):
    """Prepare texts for training by splitting long articles."""
    training_texts = []
    
    for article in articles:
        # Split very long articles into chunks
        words = article.split()
        
        # If article is short enough, use it as is
        if len(words) <= max_length:
            training_texts.append(article)
        else:
            # Split into chunks of approximately max_length words
            for i in range(0, len(words), max_length // 2):  # Overlap chunks
                chunk = ' '.join(words[i:i + max_length])
                if len(chunk.split()) > 50:  # Only keep substantial chunks
                    training_texts.append(chunk)
    
    print(f"Prepared {len(training_texts)} training texts")
    return training_texts

def main():
    # Check if Wikipedia data exists
    wikipedia_file = "data/wikipedia_articles.json"
    if not os.path.exists(wikipedia_file):
        print("Wikipedia data not found!")
        print("Please run: python3 download_wikipedia.py")
        sys.exit(1)
    
    # Load Wikipedia articles
    articles = load_wikipedia_articles(wikipedia_file)
    
    # Prepare training texts
    print("\nPreparing training data...")
    texts = prepare_training_texts(articles, max_length=256)
    
    # Split into train/eval
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]
    
    print(f"\nDataset statistics:")
    print(f"Training samples: {len(train_texts)}")
    print(f"Evaluation samples: {len(eval_texts)}")
    
    # Model configuration - larger model for Wikipedia
    config = ModelConfig(
        vocab_size=20000,      # Larger vocabulary for Wikipedia
        hidden_size=512,       # Larger hidden size
        num_layers=6,          # More layers
        num_heads=8,
        ff_size=2048,
        max_seq_length=256,
        dropout=0.1,
    )
    
    # Build tokenizer vocabulary
    print("\nBuilding vocabulary...")
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # Use a subset of texts to build vocabulary (for efficiency)
    vocab_texts = train_texts[:5000] if len(train_texts) > 5000 else train_texts
    tokenizer.build_vocab(vocab_texts)
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Save tokenizer
    tokenizer.save("data/wikipedia_tokenizer.json")
    
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
    learning_rate = 3e-4
    num_epochs = 5
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        warmup_steps=500,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        checkpoint_dir="checkpoints/wikipedia",
        save_steps=1000,
        eval_steps=500,
        device=str(device),
    )
    
    # Start training
    print("\nStarting training on Wikipedia data...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Checkpoint directory: checkpoints/wikipedia")
    print("-" * 50)
    
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    
    print("\nTraining completed!")
    print(f"Total training time: {end_time - start_time}")
    print(f"Model checkpoints saved in 'checkpoints/wikipedia'")
    
    # Save final model
    trainer.save_checkpoint("final")
    
    print("\nTo generate text with the trained model, run:")
    print("python3 generate_with_wikipedia.py")

if __name__ == "__main__":
    main()