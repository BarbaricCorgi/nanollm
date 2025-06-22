# Suggested Additional Notes for Transformer Study

## 15. Temperature in generation
- Controls randomness: low temp (0.1) = predictable, high temp (1.5) = creative
- Modifies logits before sampling: logits/temperature
- Temperature = 0 becomes greedy decoding (always pick highest probability)

## 16. Top-k and Top-p sampling
- Top-k: only consider k most likely tokens
- Top-p (nucleus): consider tokens until cumulative probability reaches p
- Prevents sampling very unlikely tokens
- Can combine both strategies for better generation

## 17. Gradient accumulation
- Simulates larger batch sizes on limited memory
- Accumulate gradients over N steps before updating weights
- Effective batch size = actual batch size × accumulation steps

## 18. Learning rate scheduling
- Warmup: start small to stabilize training
- Decay: reduce over time for fine-tuning
- Critical for transformer training success
- Common: linear warmup + cosine decay

## 19. Tied embeddings
- Input embeddings and output projection can share weights
- Reduces parameters by vocab_size × hidden_size
- Forces consistent word representations
- Trade-off: less flexibility but more efficient

## 20. Attention patterns discovered by heads
- Head 1: Previous token attention
- Head 2: Syntactic dependencies  
- Head 3: Rare token attention
- Head 4: Semantic similarity
- Different heads specialize automatically through training

## 21. Why transformers beat RNNs
- Parallel processing (not sequential)
- No vanishing gradient over long sequences
- Direct connections between distant tokens
- Much faster training on modern hardware

## 22. Common model sizes
- Small: 100M params (mobile/edge)
- Medium: 1-7B params (laptop/desktop)
- Large: 70B+ params (cloud/cluster)
- Params ≈ 2 × num_layers × hidden_size²

## 23. Tokenization strategies
- Character-level: maximum flexibility, long sequences
- Word-level: simple but huge vocabulary
- Subword (BPE/SentencePiece): balance between both
- Byte-level: can encode any text

## 24. Training data scale
- Small models: 1-10GB text
- Medium models: 100GB-1TB text
- Large models: 10TB+ text
- More data generally helps until model capacity saturated

## 25. Memory requirements
- Training: ~4-6x model size (gradients, optimizer states)
- Inference: ~2x model size (model + KV cache)
- Quantization can reduce by 2-4x

## 26. Common training issues
- Loss spikes: learning rate too high
- Loss plateau: learning rate too low or model too small
- Gradient explosion: need gradient clipping
- Overfitting: need more data or regularization