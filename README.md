# NanoLLM

A minimal implementation of a small language model for local execution, built with PyTorch.

## Features

- Simple transformer-based architecture
- Configurable model size (layers, heads, hidden dimensions)
- Basic tokenizer with vocabulary building
- Training with gradient accumulation and learning rate scheduling
- Text generation with various sampling strategies
- Checkpoint saving/loading
- Minimal dependencies

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install torch numpy tqdm transformers datasets sentencepiece
```

## Quick Start

### Training a Model

```python
from nanollm import NanoLLM, SimpleTokenizer, Trainer, ModelConfig
from nanollm.trainer import TextDataset

# Configure model
config = ModelConfig(
    vocab_size=5000,
    hidden_size=256,
    num_layers=4,
    num_heads=4,
    ff_size=1024,
    max_seq_length=128,
)

# Prepare data
texts = ["Your training texts here...", "More texts..."]
tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
tokenizer.build_vocab(texts)

# Create datasets
train_dataset = TextDataset(texts, tokenizer)

# Initialize and train
model = NanoLLM(config)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    batch_size=16,
    learning_rate=5e-4,
    num_epochs=3,
)
trainer.train()
```

### Generating Text

```python
from nanollm.inference import TextGenerator

# Load from checkpoint
generator = TextGenerator.from_checkpoint("checkpoints/best")

# Generate text
generated = generator.generate(
    "The quick brown",
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
print(generated)
```

## Examples

Run the training example:
```bash
python examples/train_simple.py
```

Generate text with a trained model:
```bash
python examples/generate_text.py --checkpoint checkpoints/best
```

## Model Architecture

- Transformer decoder with multi-head self-attention
- Learned or sinusoidal position embeddings
- Layer normalization and dropout for regularization
- Tied embedding weights (optional)
- Support for gradient checkpointing

## Configuration

Key model parameters in `ModelConfig`:
- `vocab_size`: Size of the vocabulary
- `hidden_size`: Dimension of hidden states
- `num_layers`: Number of transformer layers
- `num_heads`: Number of attention heads
- `ff_size`: Size of feed-forward layer
- `max_seq_length`: Maximum sequence length
- `dropout`: Dropout probability

## License

MIT