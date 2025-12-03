# D-J: Experiment Framework for Modular Arithmetic Learning

Comprehensive pipeline for training and visualizing neural networks on modular arithmetic tasks with reproducibility, device abstraction, and automated result tracking.

## Quick Start

```python
from main import Experiment

config = {
    'p': 17,
    'c': [4, 1, 0],
    'd': [1, 2, 3],
    'embedding_dim': 128,
    'hidden': 500,
    'learning_rate': 0.005,
    'max_steps': 1000,
    'batch_size': 1024,
    'weight_decay': 1e-4,
    'split': 0.99999,
    'negs_per_ex': 20,
    'latex_title': r'$(4x + y^2)^3$ mod p',
}

exp = Experiment(config)
exp.run(animate=False)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p` | int | - | Prime modulus (>= 2) |
| `c` | list | - | Polynomial coefficients |
| `d` | list | - | Polynomial degrees |
| `embedding_dim` | int | - | Embedding dimension |
| `hidden` | int | - | Hidden layer size |
| `learning_rate` | float | - | Learning rate |
| `max_steps` | int | - | Training iterations |
| `batch_size` | int | - | Batch size |
| `weight_decay` | float | - | L2 regularization |
| `split` | float | - | Train/test ratio (0-1) |
| `negs_per_ex` | int | - | Negative samples per example |
| `latex_title` | str | - | Experiment title |

### Methods

```python
exp.run(animate=False)           # Full pipeline
exp.setup_experiment_config()    # Initialize model/dataset
exp.train_model()                # Train only
exp.visualize_experiment()       # Visualize only
exp.calculate_parameters()       # Compute model stats
```


## Visualizations

Generated plots:
- Training and test loss curves
- Overall accuracy curves
- Positive/negative example accuracies
- (If `animate=True` and single test example): probability evolution animation

## Device Support

Automatically detects and uses:
- **MPS** (Apple Silicon)
- **CUDA** (NVIDIA GPU)
- **CPU** (fallback)

## Requirements

```
PyTorch >= 1.9
NumPy
Matplotlib
IPython
BadData_AppC
NegSamplingMath_Unlearn_AppC_FullDataSet
```