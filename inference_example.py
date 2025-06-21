#!/usr/bin/env python3

from nanollm import NanoLLM, SimpleTokenizer, ModelConfig
from nanollm.inference import TextGenerator

# Initialize model and tokenizer
config = ModelConfig(
    vocab_size=5000,
    hidden_size=256,
    num_layers=4,
    num_heads=4,
    ff_size=1024,
)

model = NanoLLM(config)
tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)

# Build a small vocabulary for demo
sample_texts = [
    "Hello world",
    "This is a test",
    "Machine learning is interesting",
]
tokenizer.build_vocab(sample_texts)

# Create text generator
generator = TextGenerator(model, tokenizer)

# Generate text (will be random since model is untrained)
prompt = "Hello"
generated = generator.generate(
    prompt,
    max_new_tokens=20,
    temperature=0.8,
    top_k=50,
)

print(f"Prompt: {prompt}")
print(f"Generated: {generated}")

# To use a trained model instead:
# generator = TextGenerator.from_checkpoint("checkpoints/best")