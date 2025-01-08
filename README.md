# Patent Phrase Similarity Classifier

A Natural Language Processing (NLP) project that analyzes and classifies similarity between phrases used in US patent descriptions.

## Overview
This project leverages the Hugging Face Transformers library and pretrained models to determine semantic similarity between patent phrases. The approach demonstrates practical NLP applications that can be extended to other domains like marketing, logistics, and medicine.

## Tech Stack
- Hugging Face Transformers
- Pretrained NLP models

## Use Cases
- Patent analysis and classification
- Semantic similarity detection
- Document comparison
- Cross-domain text analysis


## Practical Lessons

This notebook demonstrates critical practical lessons for NLP, with detailed explanations of not just what to do, but why each approach matters:

1. [Data preprocessing pipeline](./Data%20preprocessing%20pipeline.md):
   - Tokenization requires careful language-specific handling because language is inherently complex and contextual:
     - Contractions (don't -> do n't): Split to help model understand negation patterns while preserving semantic meaning
     - Special regex patterns: URLs/emails must stay intact since breaking them loses their meaning
     - Language rules: "Mr." vs "." matters because sentence boundaries affect context windows
     - Subword tokenization: Critical for handling unknown words by breaking them into meaningful pieces (e.g. "unhappily" -> "un-happi-ly")
   - Numericalization decisions directly impact model performance:
     - Vocab size tradeoff: Larger vocab = better coverage but more parameters and slower training
     - Unknown word threshold: Too low = loss of rare but important terms, too high = wasted parameters
     - Special tokens: <bos> provides crucial context clues, <eos> helps model learn natural endings
   - Batching affects both speed and learning quality:
     - Length sorting: Reduces wasted computation on padding by up to 70%
     - Dynamic batching: Adapts to sequence length for optimal GPU utilization
     - State handling: Critical for maintaining coherent context between batches

2. Transfer learning workflow (see [what is Transfer Learning in NLP](./What%20is%20Transfer%20Learning%20in%20NLP.md))
   - Language model fine-tuning:
     1. First, update only the last layer since it handles task-specific features
     2. Gradually unfreeze earlier layers one at a time to avoid damaging pretrained knowledge
     3. Use smaller learning rates for earlier layers to preserve fundamental patterns
   - Classification fine-tuning:
     1. Keep most layers frozen initially to maintain general language understanding
     2. Use different learning rates - higher for new layers, lower for pretrained
     3. Watch training curves to catch problems early and avoid wasting time

Example of this intuition:

    Layer 1 (Bottom): Basic patterns
    - Recognizes simple word patterns
    - "the" usually comes before a noun
    - Basic grammar rules
    [LEARNS VERY GENERAL PATTERNS]
    ↓
    Layer 2-3 (Middle): Intermediate concepts  
    - Understands phrases and clauses
    - Gets basic context and relationships
    - Recognizes common expressions
    [LEARNS LANGUAGE STRUCTURE]
    ↓
    Layer 4-5 (Top): Task-specific features
    - Makes final decisions for your task
    - Specializes in your domain (patents, medical, etc)
    [LEARNS YOUR SPECIFIC TASK]


3. Architecture choices and their impact:
   - AWD-LSTM benefits:
     - Weight-dropped LSTM: Prevents overfitting on sequential patterns
     - ASGD: Smoother convergence in language tasks vs standard SGD
     - Tied embeddings: Enforces consistency between input and output representations
   - Embedding optimizations:
     - Pretrained initialization: Jumpstarts learning with existing semantic knowledge
     - Adaptive sizing: Matches representation capacity to word importance
     - Tied weights: Reduces overfitting by sharing parameters

4. [Training optimizations and their rationale](./Why%20Training%20Optimizations%20Matter.md)
   - Progressive resizing is helpful:
     - Shorter sequences: Initial rapid learning of basic patterns
     - Gradual length increase: Builds up to handling long-range dependencies
     - Learning rate maintenance: Preserves fine-tuned knowledge through transitions
   - Mixed precision benefits include:
     - Memory savings: Enables larger batches for better gradient estimates
     - Speed improvement: Faster computation with minimal accuracy loss
     - Stability requirements: Why fp32 master weights matter

5. [Production considerations](./Production%20Guide%3A%20From%20Training%20to%20Deployment.md)

   Export Requirements 📦
   - Complete vocabulary must be exported with model
     → Missing terms = broken tokenization in production
     → Include special tokens like <unk>, <pad>, <bos>
   - Configuration must be preserved exactly
     → Ensures preprocessing stays identical to training
     → Includes tokenizer settings, max sequence length
   - Quantization decisions impact deployment
     → int8 = 75% less memory but 0.5% accuracy drop
     → fp16 = good balance for most use cases

   Maintenance Strategy 🔄
   - Regular vocabulary updates keep model current
     → Add emerging terms quarterly
     → Remove obsolete terms to save memory
   - Distribution monitoring catches problems early
     → Alert if input patterns shift >15%
     → Track accuracy on key metrics weekly
   - Strategic retraining maintains performance
     → Retrain if accuracy drops >2%
     → Maximum model age: 6 months

Key Takeaways on Why These Matter:
- Careful preprocessing directly impacts model understanding
- Transfer learning steps preserve critical knowledge while enabling adaptation
- Architecture choices affect both performance and training stability
- Training optimizations enable better results with limited resources
- Production considerations ensure reliable real-world performance

The notebook demonstrates these techniques using fastai while emphasizing responsible technology use.

## From Fast.ai Lesson 4

https://course.fast.ai/Lessons/lesson4.html

## Core NLP Models Explained

### LSTM (Long Short-Term Memory)
Think of LSTM as a smart reader with perfect short-term memory. It processes text sequentially (one word at a time) using special "gates" to control information flow:
- **Forget Gate**: Decides what to remove from memory
- **Input Gate**: Decides what new info to store
- **Output Gate**: Decides what to output
Best for: Sequential data processing, memory-efficient tasks

### ULMFit (Universal Language Model Fine-tuning)
ULMFit follows a three-stage learning process:
1. General language understanding (pretrained)
2. Domain-specific adaptation (like learning industry terms)
3. Task-specific fine-tuning (like learning to classify)
Best for: Transfer learning with limited data, controlled domain adaptation

### Transformers
Transformers process entire sequences simultaneously using "self-attention" - imagine having a photographic memory of the whole text at once. They excel at understanding relationships between all words in a sequence, regardless of distance.
Best for: Large-scale tasks, parallel processing, understanding long-range relationships

### Text Processing Pipeline

#### Tokenization Approaches
1. **Word Tokenization**
   - Splits on spaces and punctuation
   - Handles special cases (Mr., URLs, emails)
   - Uses special tokens (xxbos for beginning, xxup for uppercase)

2. **Subword Tokenization**
   - Analyzes common letter groups
   - Better for languages without spaces (Chinese, Japanese)
   - Handles compound words and unknown terms

### Quick Comparison

| Model     | Processing Style | Memory Usage | Best Use Case |
|-----------|-----------------|--------------|---------------|
| LSTM      | Sequential      | Linear (O(n))| Long sequences, limited resources |
| ULMFit    | Sequential+     | Linear+      | Domain adaptation, small datasets |
| Transformer| Parallel        | Quadratic (O(n²))| Large datasets, complex relationships |

### Ethical Considerations
- Language models can generate highly convincing text
- Potential for automated disinformation campaigns
- Challenge of detecting machine-generated content
- Need for responsible development and deployment


## Advanced NLP Architecture Guide

## Choosing Between LSTM, ULMFit, and Transformers

### Core Architectures

#### LSTM (Long Short-Term Memory)
A specialized RNN that handles sequential data using gates to control information flow. AWD-LSTM adds regularization through weight-dropped LSTM layers.

**Best For:**
- Sequential data where order matters deeply
- Limited computational resources
- Smaller datasets (< 100k examples)

#### ULMFit (Universal Language Model Fine-tuning)
A transfer learning method using a 3-stage approach: pretraining, domain adaptation, and task-specific fine-tuning.

**Best For:**
- Transfer learning with limited labeled data
- Domain adaptation (e.g., Wikipedia → domain-specific text)
- When you need interpretable intermediate steps

#### Transformers
Architecture using self-attention to process all input tokens simultaneously.

**Best For:**
- Large-scale tasks with substantial data
- When parallel processing is available (GPU/TPU)
- Tasks requiring long-range dependencies
- When pretraining from scratch isn't needed

### Decision Framework

1. **Resource Constraints?**
   - Limited GPU/memory → LSTM/ULMFit
   - Strong compute available → Transformers

2. **Data Size?**
   - Small dataset (< 100k examples) → ULMFit
   - Large dataset (> 1M examples) → Transformers
   - Medium dataset → Either, depending on compute

3. **Domain Specificity?**
   - Highly domain specific → ULMFit for controlled adaptation
   - General domain → Transformer-based models

4. **Interpretability Needs?**
   - Need to understand model decisions → LSTM/ULMFit
   - Pure performance matters most → Transformers

### Key Technical Differences

1. **Memory Usage**:
   - LSTM: O(n) - Linear memory usage
   - Transformer: O(n²) - Quadratic memory usage
   - ULMFit: Similar to LSTM but with additional overhead for fine-tuning

2. **Processing Speed**:
   - LSTM: Sequential processing
   - Transformer: Parallel processing
   - ULMFit: Sequential but with optimized training stages

3. **Context Window**:
   - LSTM: Theoretically unlimited but practically limited by vanishing gradients
   - Transformer: Fixed context window (e.g., 512 tokens for BERT)
   - ULMFit: Typically 70-100 tokens per sequence

### Implementation Example

```python
# ULMFit approach with AWD-LSTM
learn = text_classifier_learner(
    dls, 
    AWD_LSTM,
    drop_mult=0.5,
    config={
        'emb_sz': 400,
        'n_hid': 1150,
        'n_layers': 3
    },
    metrics=accuracy
).to_fp16()

# Gradual unfreezing for better transfer learning
learn.freeze()
learn.fit_one_cycle(1, 2e-2)
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4), 1e-2))
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4), 1e-3))
```

Sources:
- ULMFit paper (Howard & Ruder, 2018)
- BERT paper (Devlin et al., 2019)
- AWD-LSTM paper (Merity et al., 2017)
- Fast.ai documentation


