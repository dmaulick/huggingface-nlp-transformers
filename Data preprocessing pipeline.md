## Deep Dive: NLP Preprocessing Pipeline

### 1. Tokenization Deep Dive

Why it matters: Proper tokenization is crucial for model understanding. Poor tokenization can break semantic meaning and make it impossible for models to learn important patterns. Getting this right is the foundation for all downstream tasks.

#### Contraction Handling
```python
# Standard approach
contractions = {
    "don't": ["do", "n't"],
    "can't": ["can", "n't"],
    "won't": ["will", "n't"]
}

# Advanced handling with context
def handle_contractions(text):
    # Preserve semantic units
    text = re.sub(r"n't\b", " n't", text)  # don't -> do n't
    text = re.sub(r"'ll\b", " 'll", text)  # they'll -> they 'll
    return text
```

#### URL/Email Protection
```python
def protect_special_patterns(text):
    # Find and temporarily replace URLs/emails
    url_pattern = r'https?://\S+|www\.\S+'
    email_pattern = r'\S+@\S+\.\S+'
    
    # Replace with unique tokens
    text = re.sub(url_pattern, '__URL__', text)
    text = re.sub(email_pattern, '__EMAIL__', text)
    return text
```

#### Sentence Boundary Detection
```python
def handle_boundaries(text):
    # Special cases that aren't sentence boundaries
    abbreviations = {'Mr.', 'Mrs.', 'Dr.', 'Prof.', 'Sr.', 'Jr.'}
    
    # Look ahead/behind for better accuracy
    for abbr in abbreviations:
        text = re.sub(f'{abbr}\s+([A-Z])', f'{abbr}__NOBOUNDARY__{1}', text)
    return text
```

### 2. Numericalization Strategy

#### Vocab Size Optimization
```python
def build_vocab(texts, min_freq=3, max_vocab=30000):
    # Count frequencies
    counter = Counter(word for text in texts for word in text.split())
    
    # Filter by frequency and size
    vocab = {
        word: idx 
        for idx, (word, count) in enumerate(counter.most_common(max_vocab))
        if count >= min_freq
    }
    
    # Add special tokens
    vocab['<unk>'] = len(vocab)  # Unknown words
    vocab['<pad>'] = len(vocab)  # Padding
    vocab['<bos>'] = len(vocab)  # Beginning of sequence
    vocab['<eos>'] = len(vocab)  # End of sequence
    
    return vocab
```

### 3. Efficient Batching

#### Length-Based Sorting
```python
def create_batches(sequences, batch_size=32):
    # Sort by length for efficient padding
    sorted_seqs = sorted(sequences, key=len)
    
    # Group similar lengths
    batches = [
        sorted_seqs[i:i + batch_size] 
        for i in range(0, len(sorted_seqs), batch_size)
    ]
    
    # Add minimal required padding
    padded_batches = [
        pad_sequence(batch, max_len=len(max(batch, key=len)))
        for batch in batches
    ]
    
    return padded_batches
```

#### Dynamic Batch Sizing
```python
def dynamic_batch_size(sequence_length):
    # Adjust batch size based on sequence length
    # Maintain roughly constant memory usage
    base_batch_size = 32
    return max(1, base_batch_size * (512 // sequence_length))
```

### Performance Impact

| Optimization | Memory Savings | Speed Impact | Quality Impact | Real-World Example |
|--------------|---------------|--------------|----------------|-------------------|
| Length Sorting | -30% padding | +40% speed | Neutral | Processing Twitter feeds (varied lengths) |
| Dynamic Batching | -20% memory | +25% speed | Slight+ | News article analysis (long docs) |
| Vocab Optimization | -15% memory | +10% speed | Model dependent | Customer support tickets |

### Key Considerations

1. **Tokenization Balance**:
   - Too aggressive splitting → loses compound meaning (e.g., "New York" becomes meaningless as "New" + "York")
   - Too conservative → increases vocab size (storing "playing", "played", "plays" separately)
   - Sweet spot: Split only when semantically meaningful

2. **Vocab Size Trade-offs**:
   ```python
   # Real-world vocab size guidelines based on data type
   vocab_sizes = {
       # Social media - 10k words because:
       # - ~5k common words cover 90% of social posts
       # - ~2k emoji/hashtag special tokens
       # - ~3k buffer for trending terms
       # Keeping small helps model adapt to rapid language changes
       10_000: "Social media (emojis, slang)",  
       
       # News/articles - 30k words because:
       # - ~15k common English words
       # - ~10k topic-specific terms (politics, sports, etc)
       # - ~5k proper nouns (names, places)
       # Sweet spot between coverage (95%) and training speed
       30_000: "General purpose news/articles",
       
       # Technical docs - 50k words because:
       # - ~20k common technical terms
       # - ~20k field-specific jargon
       # - ~10k compound terms
       # Need larger vocab as splitting terms loses meaning
       50_000: "Technical documentation/papers",
       
       # Legal/Medical - 100k words because:
       # - ~40k medical terms (diseases, drugs, procedures)
       # - ~40k legal terms (statutes, precedents)
       # - ~20k common professional terms
       # Large vocab critical as terms have precise definitions
       100_000: "Legal/Medical (specialized terms)"
   }
   ```

3. **Batch Optimization Rules**:
   - Target 80-90% GPU utilization
   - Keep batch size * sequence length ≈ constant
   - Example: 512 tokens × 64 batch = 32,768 total tokens
   - Memory usage ≈ O(batch_size × seq_length²) for attention
   - Critical for transformer models where attention is quadratic

4. **Common Bottlenecks**:
   - CPU preprocessing becoming the bottleneck (Solution: Parallel DataLoader)
   - OOM during attention computation (Solution: Gradient checkpointing)
   - Vocab cache misses (Solution: Frequency-based vocab)