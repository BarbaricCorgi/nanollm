#!/usr/bin/env python3

import os
import sys
from nanollm.inference import TextGenerator

def interactive_generation(generator):
    """Interactive text generation loop."""
    print("\nInteractive Text Generation")
    print("Type 'quit' to exit, 'help' for options")
    print("-" * 50)
    
    # Default parameters
    max_new_tokens = 100
    temperature = 0.8
    top_k = 50
    top_p = 0.95
    
    while True:
        prompt = input("\nEnter prompt: ").strip()
        
        if prompt.lower() == 'quit':
            break
        elif prompt.lower() == 'help':
            print("\nOptions:")
            print("  quit - Exit the program")
            print("  help - Show this help")
            print("  settings - Show current generation settings")
            print("  set <param> <value> - Change generation settings")
            print("    Parameters: max_tokens, temperature, top_k, top_p")
            continue
        elif prompt.lower() == 'settings':
            print(f"\nCurrent settings:")
            print(f"  max_new_tokens: {max_new_tokens}")
            print(f"  temperature: {temperature}")
            print(f"  top_k: {top_k}")
            print(f"  top_p: {top_p}")
            continue
        elif prompt.lower().startswith('set '):
            parts = prompt.split()
            if len(parts) == 3:
                param, value = parts[1], parts[2]
                try:
                    if param == 'max_tokens':
                        max_new_tokens = int(value)
                        print(f"Set max_new_tokens to {max_new_tokens}")
                    elif param == 'temperature':
                        temperature = float(value)
                        print(f"Set temperature to {temperature}")
                    elif param == 'top_k':
                        top_k = int(value)
                        print(f"Set top_k to {top_k}")
                    elif param == 'top_p':
                        top_p = float(value)
                        print(f"Set top_p to {top_p}")
                    else:
                        print(f"Unknown parameter: {param}")
                except ValueError:
                    print(f"Invalid value for {param}")
            else:
                print("Usage: set <param> <value>")
            continue
        
        if not prompt:
            continue
        
        print("\nGenerating...")
        try:
            generated = generator.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            print(f"\n{generated}")
        except Exception as e:
            print(f"Error generating text: {e}")

def main():
    # Check for checkpoint
    checkpoint_paths = [
        "checkpoints/wikipedia/best",
        "checkpoints/wikipedia/final",
        "checkpoints/wikipedia/epoch-5",
        "checkpoints/wikipedia/epoch-4",
        "checkpoints/wikipedia/epoch-3",
        "checkpoints/wikipedia/epoch-2",
        "checkpoints/wikipedia/epoch-1",
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(os.path.join(path, "model.pt")):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        print("No Wikipedia-trained model found!")
        print("Please run: python3 train_wikipedia.py")
        sys.exit(1)
    
    print(f"Loading model from {checkpoint_path}...")
    generator = TextGenerator.from_checkpoint(checkpoint_path)
    print("Model loaded successfully!")
    
    # Example generations
    print("\n" + "=" * 50)
    print("Example Generations")
    print("=" * 50)
    
    example_prompts = [
        "The history of",
        "Machine learning is",
        "The capital city of",
        "In the year 2020",
        "Scientists have discovered",
        "The most important",
    ]
    
    for prompt in example_prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = generator.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
        )
        print(f"Generated: {generated}")
    
    # Interactive mode
    print("\n" + "=" * 50)
    interactive_generation(generator)

if __name__ == "__main__":
    main()