
## Tokenization Special Case Example

Let me explain how these tokenization rules work in practice with concrete examples:

```python
import re
from transformers import AutoTokenizer

# Commonregex patterns for URLs and emails
URL_PATTERN = r'https?://\S+|www\.\S+'
EMAIL_PATTERN = r'\S+@\S+\.\S+'

# Example text
text = """
Check out https://fast.ai for ML courses. Contact support@fast.ai
Mr. Smith works at Apple Inc. The company is great.
"""

# 1. Preserve URLs and emails
def preserve_special_tokens(text):
    # Replace URLs and emails with special tokens
    urls = re.findall(URL_PATTERN, text)
    emails = re.findall(EMAIL_PATTERN, text)
    
    text = re.sub(URL_PATTERN, ' <URL> ', text)
    text = re.sub(EMAIL_PATTERN, ' <EMAIL> ', text)
    return text, urls, emails

# 2. Handle sentence boundaries
def handle_boundaries(text):
    # Common abbreviations that shouldn't split sentences
    abbreviations = ['Mr.', 'Mrs.', 'Dr.', 'Inc.', 'Ltd.']
    
    for abbr in abbreviations:
        text = text.replace(abbr, abbr.replace('.', '<PERIOD>'))
    
    # Now safe to split on periods
    text = text.replace('.', ' . ')
    
    # Restore abbreviation periods
    text = text.replace('<PERIOD>', '.')
    return text

# Example usage with a modern transformer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Process text
processed_text, urls, emails = preserve_special_tokens(text)
processed_text = handle_boundaries(processed_text)

# Tokenize
tokens = tokenizer.tokenize(processed_text)
print(tokens)
```

This code shows:
1. URLs/emails are preserved as special tokens instead of being split into meaningless pieces
2. Sentence boundaries are properly handled by distinguishing between periods in abbreviations vs. end of sentences
3. The resulting tokens maintain semantic meaning while being model-friendly

When run, you'll see the text is tokenized while maintaining important structural elements that would otherwise be lost with naive tokenization.

Sources:
- HuggingFace Tokenizers documentation
- spaCy's tokenization rules
- fastai's text preprocessing pipeline
