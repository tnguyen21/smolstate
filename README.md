# SmolState

A minimal reimplementation of the state perturbation model training loop.

## Overview

SmolState provides a simplified, standalone training pipeline for the StateTransitionPerturbationModel, inspired by and adapted from `state/src/state/tx/models/state_transition.py`. This implementation extracts the core training logic from the larger state framework while maintaining compatibility with existing data and model configurations.

## Features

- **Minimal Dependencies**: Only essential packages for training
- **Config Compatibility**: Uses existing state configuration files
- **Data Integration**: Compatible with state's PerturbationDataModule
- **Checkpointing**: Automatic checkpoint saving and resuming
- **Logging**: Console logging support
- **CLI Interface**: Simple command-line interface

## Quick Start

### Basic Training
```bash
# Option 1: Direct execution
python main.py train

# Option 2: Module execution  
python -m smolstate train

# Option 3: Installed package (after pip install -e .)
smolstate train
```

### With Configuration Overrides
```bash
python main.py train \
    data.kwargs.toml_config_path="../state/starter.toml" \
    training.max_steps=400 \
    training.batch_size=8 \
    model.kwargs.hidden_dim=512
```

### Resume Training
```bash
python main.py train --resume out/smolstate_run/checkpoints/last.ckpt
```

## Configuration

SmolState uses the same configuration system as state:

- **starter.toml**: Dataset paths and training specifications
- **model configs**: From `state/src/state/configs/model/`
- **training configs**: From `state/src/state/configs/training/`

### Command Line Overrides

Use dot notation to override any configuration parameter:

```bash
# Model parameters
model.kwargs.hidden_dim=672
model.kwargs.cell_set_len=128

# Training parameters  
training.max_steps=40000
training.lr=1e-4
training.batch_size=16

# Data parameters
data.kwargs.batch_col=gem_group
data.kwargs.pert_col=gene
```

## Architecture

### Core Components

1. **config.py**: Configuration loading and parsing
2. **data.py**: Data module wrapper and utilities
3. **model.py**: StateTransitionPerturbationModel implementation
4. **train.py**: Training loop and metrics
5. **checkpoint.py**: Model checkpointing and resuming
6. **main.py**: CLI interface

### Model Architecture

The StateTransitionPerturbationModel (adapted from `state/src/state/tx/models/state_transition.py`) features:
- Transformer backbone (LLaMA-based by default, with bidirectional attention)
- Perturbation and basal expression encoders  
- Distributional losses (Energy, Sinkhorn) via geomloss
- Optional gene decoder for multi-task learning
- Optional confidence token for uncertainty estimation
- Residual connections and layer normalization

## Output Structure

```
out/
└── smolstate_run/
    ├── checkpoints/
    │   ├── last.ckpt
    │   ├── final.ckpt
    │   └── step_*.ckpt
    ├── config.json
    ├── cell_type_onehot_map.pkl
    ├── pert_onehot_map.pt
    ├── batch_onehot_map.pkl
    ├── var_dims.pkl
    └── data_module.torch
```

## System Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU training)

## Dependencies

Core dependencies:
- torch
- geomloss  
- pyyaml
- toml

Additional dependencies:
- h5py (for reading H5AD files)
- transformers (for transformer backbones)
- tqdm (for progress bars)

Note: This implementation is standalone and does not require the full state framework installation.

## Examples

### Minimal Training Run
```python
from smolstate import create_config, create_data_module, StateTransitionPerturbationModel, create_trainer

# Load configuration
config = create_config()

# Setup data
data_module = create_data_module(config.config)
train_dl, val_dl = data_module.get_dataloaders()

# Create model
model_dims = data_module.get_model_dims()
model_kwargs = config.get_model_kwargs()
model_kwargs.update(model_dims)

model = StateTransitionPerturbationModel(**model_kwargs)

# Train
trainer = create_trainer(model, train_dl, val_dl, config.config, "output")
trainer.train()
```

### Custom Configuration
```python
from smolstate import create_config

# Custom overrides
overrides = {
    "training.max_steps": 1000,
    "training.lr": 5e-5,  
    "model.kwargs.hidden_dim": 512,
}

config = create_config(overrides=overrides)
```

## Development

SmolState is designed to be minimal and hackable. The codebase is small and focused, making it easy to modify for specific research needs.

### Key Design Principles

1. **Simplicity**: Minimal abstraction, clear data flow
2. **Compatibility**: Works with existing state configs and data
3. **Modularity**: Each component can be used independently
4. **Transparency**: Clear logging and checkpointing

## Troubleshooting

### Common Issues

1. **Tensor Shape Errors**: Ensure data loading returns correctly shaped tensors for the model
2. **Data Loading**: Check starter.toml paths are correct and H5AD files are accessible
3. **Memory Issues**: Reduce batch_size or cell_set_len for large datasets
4. **CUDA Errors**: Verify GPU memory and CUDA compatibility, use CPU fallback if needed

### Debug Mode

For debugging, enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

Same as parent state project.