## What is Transfer Learning in NLP?

Transfer learning is a technique where a model trained on one task is repurposed for a second related task. In NLP, this typically involves pretrained language models, with ULMFiT (Universal Language Model Fine-tuning) being one notable approach:

1. **Pretraining**: Model learns general language patterns from large corpus (e.g. Wikipedia, Common Crawl)
   - Develops foundational understanding of syntax, grammar, and semantics
   - Uses self-supervised learning (typically masked language modeling or next token prediction)
   - Creates contextual language representations

2. **Domain Adaptation**: Fine-tune pretrained model on target domain text (optional step)
   - Adapts to domain-specific vocabulary and writing styles
   - Preserves general language knowledge while learning new patterns
   - May use techniques like discriminative fine-tuning

3. **Task-specific Fine-tuning**: Final adaptation for target task
   - Adds task-specific layers for the downstream task
   - Often uses lower learning rates for pretrained layers
   - May employ techniques like gradual unfreezing

### Why It Matters

Transfer learning enables:
- Strong performance with limited labeled data
- Faster training and convergence
- Better generalization
- Reduced computational and data requirements

### Key Techniques

- **Discriminative Fine-tuning**: Using different learning rates per layer
- **Gradual Unfreezing**: Incrementally unfreezing layers from last to first
- **Slanted Triangular Learning Rates**: Learning rate scheduling strategy
- **Masked Language Modeling**: Predicting masked tokens (BERT-style)
- **Autoregressive Pretraining**: Next token prediction (GPT-style)

Modern transfer learning approaches include BERT, GPT, T5, and their variants, which have significantly advanced the state of NLP.