# Code Standards

## Overview

Our code standards are designed to ensure:
- Maintainability
- Readability
- Type safety
- Performance
- Testability

## Python Standards

### Type Hints
```python
from typing import Optional, Sequence, TypeVar
from pathlib import Path
from torch import Tensor

T = TypeVar('T')

def process_batch(
images: Sequence[Tensor],
labels: Optional[Sequence[int]] = None,
*,
augment: bool = True
) -> tuple[Tensor, Optional[Tensor]]:
"""Process a batch of images and optional labels."""
...
```

### Documentation
```python
class ModelTrainer:
"""Handles model training and validation.

This class encapsulates the training loop logic, including:
- Gradient updates
- Metric tracking
- Checkpoint management
- Early stopping

Attributes:
model: The neural network model
optimizer: The optimization algorithm
scheduler: Learning rate scheduler
device: Computing device (CPU/GPU)
"""

def train_epoch(
self,
dataloader: DataLoader,
*,
grad_clip: float = 1.0
) -> dict[str, float]:
"""Trains the model for one epoch.

Args:
dataloader: Iterator over training data
grad_clip: Maximum gradient norm

Returns:
Dictionary of metrics for the epoch

Raises:
RuntimeError: If loss becomes NaN
"""
...
```

### Code Organization

#### File Structure
```
src/
rxvision/
__init__.py
models/
__init__.py
backbone.py
heads.py
data/
__init__.py
dataset.py
transforms.py
utils/
__init__.py
metrics.py
```

#### Import Order
```python
# Standard library
from pathlib import Path
from typing import Optional

# Third-party packages
import numpy as np
import torch
from torch import nn

# Local imports
from rxvision.models import Backbone
from rxvision.utils import compute_metrics
```

## Testing Standards

### Unit Tests
```python
def test_model_forward():
"""Test model forward pass with various input sizes."""
model = create_model()
batch_sizes = [1, 8, 32]

for batch_size in batch_sizes:
inputs = torch.randn(batch_size, 3, 224, 224)
outputs = model(inputs)

assert outputs.shape == (batch_size, NUM_CLASSES)
assert not torch.isnan(outputs).any()
```

### Integration Tests
```python
@pytest.mark.integration
def test_training_pipeline(tmp_path: Path):
"""Test end-to-end training pipeline."""
config = load_config()
trainer = setup_trainer(config)

# Train for one epoch
metrics = trainer.train_epoch()

assert metrics['loss'] > 0
assert metrics['accuracy'] > 0.1
assert Path(tmp_path / 'checkpoint.pt').exists()
```

## Git Practices

### Commit Messages
```
feat(model): Add attention mechanism to backbone

- Implement self-attention layer
- Add skip connections
- Update forward pass logic

Performance improved by 2.3% on validation set.
Breaking changes:
- Model forward() now returns attention maps
- Config requires attention parameters

Closes #123
```

### Branch Naming
- Feature: `feature/add-attention`
- Bugfix: `fix/nan-gradients`
- Release: `release/v1.2.0`

## Code Review

### Review Checklist
1. Type safety
2. Test coverage
3. Documentation
4. Performance impact
5. Security implications

### Performance Guidelines
- Profile critical paths
- Optimize hot loops
- Use vectorized operations
- Minimize memory allocations

## Monitoring & Logging

### Log Levels
```python
logger = logging.getLogger(__name__)

logger.debug("Processing batch %d", batch_idx)
logger.info("Epoch %d completed", epoch)
logger.warning("Learning rate below threshold")
logger.error("Failed to load checkpoint: %s", err)
```

### Metrics
```python
def log_metrics(metrics: dict[str, float]) -> None:
"""Log metrics to monitoring system."""
for name, value in metrics.items():
if not np.isfinite(value):
logger.warning("Non-finite metric: %s=%f", name, value)
mlflow.log_metric(name, value)
```

## Related Documentation
- [[Development Setup]]
- [[Local Development]]
- [[Testing Strategy]] 