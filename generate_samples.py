#!/usr/bin/env python3

import os
import sys
from nanollm.inference import TextGenerator

def main():
    # Check for checkpoint
    checkpoint_paths = [
        "checkpoints/samples/best",
        "checkpoints/samples/final",
        "checkpoints/samples/epoch-10",
        "checkpoints/samples/epoch-9",
        "checkpoints/samples/epoch-8",
        "checkpoints/samples/epoch-7",
        "checkpoints/samples/epoch-6",
        "checkpoints/samples/epoch-5",
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(os.path.join(path, "model.pt")):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        print("No sample-trained model found!")
        print("Please run: python3 train_on_samples.py")
        sys.exit(1)
    
    print(f"Loading model from {checkpoint_path}...")
    generator = TextGenerator.from_checkpoint(checkpoint_path)
    print("Model loaded successfully!")
    
    print("\n" + "=" * 50)
    print("Text Generation Examples")
    print("=" * 50)
    
    # Literary style prompts
    literary_prompts = [
        "Once upon a time",
        "It was the best of times",
        "In a hole in the ground",
        "The sun was setting over",
        "She walked through the garden",
    ]
    
    print("\nLiterary Style Generation:")
    print("-" * 30)
    for prompt in literary_prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = generator.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
        )
        print(f"Generated: {generated}")
    
    # Technical style prompts
    technical_prompts = [
        "Machine learning is",
        "The algorithm works by",
        "Python programming",
        "Neural networks can",
        "Data science involves",
    ]
    
    print("\n\nTechnical Style Generation:")
    print("-" * 30)
    for prompt in technical_prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = generator.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
        )
        print(f"Generated: {generated}")
    
    # Interactive mode
    print("\n" + "=" * 50)
    print("Interactive Generation (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        prompt = input("\nEnter prompt: ").strip()
        if prompt.lower() == 'quit':
            break
        if not prompt:
            continue
        
        print("\nGenerating with different temperatures:")
        
        # Conservative (low temperature)
        print("\nConservative (temp=0.5):")
        generated = generator.generate(prompt, max_new_tokens=40, temperature=0.5)
        print(f"  {generated}")
        
        # Balanced (medium temperature)
        print("\nBalanced (temp=0.8):")
        generated = generator.generate(prompt, max_new_tokens=40, temperature=0.8)
        print(f"  {generated}")
        
        # Creative (high temperature)
        print("\nCreative (temp=1.2):")
        generated = generator.generate(prompt, max_new_tokens=40, temperature=1.2, top_k=100)
        print(f"  {generated}")

if __name__ == "__main__":
    main()