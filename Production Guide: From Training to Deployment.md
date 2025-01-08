## Production Guide: From Training to Deployment

### Why Production Considerations Matter 🎯
ML models often work in research but fail in production due to:
- Inconsistent preprocessing between training and serving
- Memory leaks from improper tensor handling
- Performance degradation from data drift
- Resource constraints in production

### Core Components & Examples 💻

#### 1. Model Export
Why: Training artifacts must be perfectly reproduced in production
```python
def export_production_model(model, config):
    """
    Export model with ALL required components for reproducible inference
    - Vocabulary: Required for identical tokenization
    - Config: Ensures same preprocessing
    - Model weights: Actual parameters
    """
    artifacts = {
        'model': model.state_dict(),
        'vocab': model.vocab,
        'config': {
            'max_len': 72,  # Must match training
            'tokenizer': 'spacy',  # Must be available in prod
            'special_tokens': ['<unk>', '<pad>']  # Required for edge cases
        }
    }
    torch.save(artifacts, 'model.pth')
```

#### 2. Production Inference Pipeline
Why: Must handle high throughput while maintaining low latency
```python
class ProductionInferenceService:
    """
    Production-grade inference with:
    - Batching: Optimize GPU/CPU utilization
    - Error handling: Graceful failure modes
    - Monitoring: Track performance metrics
    """
    def __init__(self, model_path):
        self.artifacts = torch.load(model_path)
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        
    @torch.inference_mode()  # Critical for memory efficiency
    def predict_batch(self, texts: List[str], batch_size=32) -> Dict:
        try:
            batches = [texts[i:i+batch_size] 
                      for i in range(0, len(texts), batch_size)]
            results = []
            
            for batch in batches:
                tokens = self.tokenizer(batch)
                with torch.cuda.amp.autocast():  # Mixed precision for speed
                    output = self.model(tokens)
                results.extend(output)
                
            return {'predictions': results}
            
        except Exception as e:
            log_error(e)  # Log for monitoring
            return {'error': str(e)}
```

#### 3. Health Monitoring
Why: Catch issues before they affect users
```python
class ModelHealthMonitor:
    """
    Monitor critical metrics:
    - Accuracy: Detect performance drops
    - Latency: Ensure SLA compliance
    - OOV rate: Catch vocabulary drift
    """
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.metrics_history = []

    def check_health(self, current_metrics: Dict) -> Dict:
        checks = {
            'accuracy': current_metrics['accuracy'] > self.thresholds['min_accuracy'],
            'latency': current_metrics['p95_latency'] < self.thresholds['max_latency'],
            'oov_rate': current_metrics['oov_rate'] < self.thresholds['max_oov']
        }
        
        if not all(checks.values()):
            self.trigger_alert(checks)
            
        return checks
```

### Resource Planning 📊

#### Memory Requirements
Why: Prevent OOM errors and optimize costs
```python
def calculate_memory_needs(model_config):
    """
    Calculate production memory requirements:
    - Model size: Parameters + gradients
    - Batch memory: Input tensors + intermediate activations
    - Buffer: Safety margin for spikes
    """
    model_params = model_config['hidden_size'] * model_config['n_layers']
    batch_memory = model_config['batch_size'] * model_config['seq_length'] * 4  # float32
    buffer = 1.5  # 50% safety margin
    
    return {
        'min_memory': (model_params + batch_memory) * buffer,
        'recommended_memory': (model_params + batch_memory) * 2
    }
```

#### Scaling Strategy
Why: Handle varying load while controlling costs
```python
class AutoscalingConfig:
    """
    Define scaling behavior:
    - Min replicas: Ensure baseline availability
    - Max replicas: Control costs
    - Scale triggers: When to add/remove capacity
    """
    def __init__(self):
        self.config = {
            'min_replicas': 2,  # High availability
            'max_replicas': 10,  # Cost control
            'target_latency': 100,  # ms
            'scale_up_threshold': 0.75,  # CPU utilization
            'scale_down_threshold': 0.25
        }
```

### Common Production Issues & Solutions 🔧

1. **Memory Leaks**
```python
# Wrong ❌
def predict(text):
    tokens = tokenize(text).cuda()  # Stays in GPU memory
    return model(tokens)

# Right ✅
def predict(text):
    with torch.inference_mode():
        tokens = tokenize(text).cuda()
        result = model(tokens)
        del tokens  # Explicit cleanup
        torch.cuda.empty_cache()  # Clear GPU memory
        return result
```

2. **Inconsistent Preprocessing**
```python
# Wrong ❌
def process_text(text):
    return text.lower()  # Missing steps from training

# Right ✅
def process_text(text):
    """
    Exactly match training preprocessing:
    - Lowercase
    - Special token handling
    - Sequence length checks
    """
    text = standardize_text(text)  # Same as training
    text = add_special_tokens(text)  # Same as training
    return truncate_sequence(text, max_len=72)  # Same as training
```

### Deployment Checklist ✅

1. Model Artifacts
   - [ ] Full vocabulary exported
   - [ ] All configs saved
   - [ ] Preprocessing matches training

2. Performance
   - [ ] Batch inference implemented
   - [ ] Memory usage optimized
   - [ ] Latency requirements met

3. Monitoring
   - [ ] Health checks configured
   - [ ] Alerts set up
   - [ ] Logging implemented