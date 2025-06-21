#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanollm import NanoLLM, SimpleTokenizer, ModelConfig
from nanollm.inference import TextGenerator
import torch
import argparse


def generate_with_new_model():
    print("Generating with a new untrained model...")
    
    config = ModelConfig(
        vocab_size=5000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        ff_size=1024,
        max_seq_length=128,
    )
    
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
    ]
    tokenizer.build_vocab(sample_texts)
    
    model = NanoLLM(config)
    
    generator = TextGenerator(model, tokenizer)
    
    prompts = [
        "The quick",
        "Machine learning",
        "Python is",
    ]
    
    print("\nGenerating text (untrained model - expect random output):")
    print("-" * 50)
    
    for prompt in prompts:
        generated = generator.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
        )
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print()


def generate_from_checkpoint(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    generator = TextGenerator.from_checkpoint(checkpoint_path)
    
    prompts = [
        "The quick brown",
        "Machine learning is",
        "Python programming",
        "Neural networks are",
    ]
    
    print("\nGenerating text from trained model:")
    print("-" * 50)
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        print("\nGreedy decoding:")
        generated = generator.generate(
            prompt,
            max_new_tokens=30,
            do_sample=False,
        )
        print(f"  {generated}")
        
        print("\nSampling (temperature=0.8):")
        generated = generator.generate(
            prompt,
            max_new_tokens=30,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
        )
        print(f"  {generated}")
        
        print("\nMultiple samples:")
        generated_list = generator.generate(
            prompt,
            max_new_tokens=20,
            temperature=1.0,
            num_return_sequences=3,
        )
        for i, text in enumerate(generated_list, 1):
            print(f"  {i}. {text}")


def main():
    parser = argparse.ArgumentParser(description="Generate text with NanoLLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint directory (e.g., checkpoints/best)"
    )
    args = parser.parse_args()
    
    print("NanoLLM Text Generation Example")
    print("=" * 50)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        generate_from_checkpoint(args.checkpoint)
    else:
        print("No checkpoint provided or checkpoint not found.")
        print("Demonstrating with untrained model.\n")
        generate_with_new_model()
        
        print("\nTo use a trained model, run:")
        print("  python generate_text.py --checkpoint checkpoints/best")


if __name__ == "__main__":
    main()