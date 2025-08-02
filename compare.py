"""
Simple comparison script between PyTorch Lightning and vanilla PyTorch implementations.
Verifies that both implementations produce identical outputs with the same random weights.
"""

import torch
import numpy as np
from typing import Dict, Any
import warnings

warnings.filterwarnings("ignore")

from state.tx.models.state_transition import (
    StateTransitionPerturbationModel as LightningModel,
)

from smolstate.model import StateTransitionPerturbationModel as VanillaModel


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_test_batch(
    batch_size: int = 2,
    seq_len: int = 64,
    input_dim: int = 2000,
    pert_dim: int = 256,
    gene_dim: int = 5000,
) -> Dict[str, torch.Tensor]:
    """Create a test batch with random data."""

    total_cells = batch_size * seq_len

    batch = {
        "pert_emb": torch.randn(total_cells, pert_dim),
        "ctrl_cell_emb": torch.randn(total_cells, input_dim),
        "pert_cell_emb": torch.randn(total_cells, input_dim),  # Target in latent space
        "pert_cell_counts": torch.randn(total_cells, gene_dim),  # Target in gene space
        "batch": torch.randint(0, 3, (total_cells,)),
        "pert_name": ["test_pert"] * total_cells,
        "cell_type": ["test_cell"] * total_cells,
    }

    return batch


def create_model_config() -> Dict[str, Any]:
    """Create model configuration that works for both implementations."""
    return {
        "input_dim": 2000,
        "hidden_dim": 768,
        "output_dim": 2000,
        "pert_dim": 256,
        "batch_dim": 3,
        "gene_dim": 5000,
        "cell_set_len": 64,
        "n_encoder_layers": 2,
        "n_decoder_layers": 2,
        "dropout": 0.1,
        "predict_residual": True,
        "distributional_loss": "energy",
        "transformer_backbone_key": "GPT2",
        "transformer_backbone_kwargs": {"n_embd": 768, "n_positions": 64},
        "output_space": "gene",
        "embed_key": "X_hvg",
        "use_basal_projection": True,
        "batch_encoder": True,
        "confidence_token": True,
        "decoder_cfg": {
            "latent_dim": 2000,
            "gene_dim": 5000,
            "hidden_dims": [1024, 512],
            "dropout": 0.1,
        },
        "gene_decoder_bool": True,
        "loss": "energy",
        "blur": 0.05,
        "embed_key": "X_hvg",
    }


def compare_implementations():
    """Compare the Lightning and vanilla PyTorch implementations."""

    print("Comparing PyTorch Lightning vs Vanilla PyTorch Implementations")
    print("=" * 70)

    config = create_model_config()
    print("Model Configuration:")
    key_params = ["input_dim", "hidden_dim", "output_dim", "pert_dim", "cell_set_len"]
    for key in key_params:
        print(f"   {key}: {config[key]}")
    print(f"   confidence_token: {config['confidence_token']}")
    print(f"   batch_encoder: {config['batch_encoder']}")
    print()

    print("Creating test batch...")
    batch = create_test_batch(
        batch_size=2,
        seq_len=config["cell_set_len"],
        input_dim=config["input_dim"],
        pert_dim=config["pert_dim"],
        gene_dim=config["gene_dim"],
    )
    print("   Input shapes:")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"     {key}: {value.shape}")
    print()

    print("Creating models with identical weights...")

    # Lightning model
    set_seed(42)
    lightning_model = LightningModel(**config)
    lightning_model.eval()

    # Vanilla model
    set_seed(42)  # Same seed = same weights
    vanilla_model = VanillaModel(**config)
    vanilla_model.eval()

    # Verify parameter counts match
    lightning_params = sum(p.numel() for p in lightning_model.parameters())
    vanilla_params = sum(p.numel() for p in vanilla_model.parameters())

    print(f"   Lightning model parameters: {lightning_params:,}")
    print(f"   Vanilla model parameters: {vanilla_params:,}")

    if lightning_params != vanilla_params:
        print("   WARNING: Parameter counts don't match!")
        return False
    print()

    # Forward pass through both models
    print("Running forward passes...")

    with torch.no_grad():
        # Lightning model forward pass
        try:
            lightning_output = lightning_model.forward(batch, padded=True)
            if isinstance(lightning_output, tuple):
                lightning_main, lightning_confidence = lightning_output
            else:
                lightning_main, lightning_confidence = lightning_output, None
        except Exception as e:
            print(f"Error in Lightning model: {e}")
            return False

        # Vanilla model forward pass
        try:
            vanilla_output = vanilla_model.forward(batch, padded=True)
            if isinstance(vanilla_output, tuple):
                vanilla_main, vanilla_confidence = vanilla_output
            else:
                vanilla_main, vanilla_confidence = vanilla_output, None
        except Exception as e:
            print(f"Error in vanilla model: {e}")
            return False

    print(f"   Lightning output shape: {lightning_main.shape}")
    print(f"   Vanilla output shape: {vanilla_main.shape}")

    if lightning_confidence is not None:
        print(f"   Lightning confidence shape: {lightning_confidence.shape}")
        print(f"   Vanilla confidence shape: {vanilla_confidence.shape}")
    print()

    # Compare outputs
    print("Comparing outputs...")

    # Main output comparison
    main_diff = torch.abs(lightning_main - vanilla_main)
    max_main_diff = main_diff.max().item()
    mean_main_diff = main_diff.mean().item()

    print("   Main output:")
    print(f"     Max difference: {max_main_diff:.2e}")
    print(f"     Mean difference: {mean_main_diff:.2e}")
    print(f"     Lightning range: [{lightning_main.min():.4f}, {lightning_main.max():.4f}]")
    print(f"     Vanilla range: [{vanilla_main.min():.4f}, {vanilla_main.max():.4f}]")

    # Confidence comparison (if present)
    if lightning_confidence is not None and vanilla_confidence is not None:
        conf_diff = torch.abs(lightning_confidence - vanilla_confidence)
        max_conf_diff = conf_diff.max().item()
        mean_conf_diff = conf_diff.mean().item()

        print("   Confidence output:")
        print(f"     Max difference: {max_conf_diff:.2e}")
        print(f"     Mean difference: {mean_conf_diff:.2e}")

    # Determine success
    tolerance = 1e-4
    main_match = max_main_diff < tolerance

    confidence_match = True
    if lightning_confidence is not None and vanilla_confidence is not None:
        confidence_match = conf_diff.max().item() < tolerance

    success = main_match and confidence_match

    print(f"\n{'PASSED' if success else 'FAILED'} Comparison Result:")
    if success:
        print("   PASSED - Both implementations produce nearly identical outputs")
        print(f"   Maximum difference: {max_main_diff:.2e} (tolerance: {tolerance:.2e})")
    else:
        print("   FAILED - Implementations differ significantly")
        print(f"   Maximum difference: {max_main_diff:.2e} (tolerance: {tolerance:.2e})")

        # Additional debugging info
        print("\nDebugging Info:")
        print(f"   First 5 Lightning values: {lightning_main.flatten()[:5].tolist()}")
        print(f"   First 5 Vanilla values: {vanilla_main.flatten()[:5].tolist()}")
        print(f"   First 5 differences: {main_diff.flatten()[:5].tolist()}")

    return success


def run_quick_tests():
    """Run quick tests with different configurations."""

    print("\nQuick Configuration Tests")
    print("=" * 40)

    base_config = create_model_config()
    test_configs = [
        {"name": "No Confidence Token", "confidence_token": False},
        {"name": "No Batch Encoder", "batch_encoder": False},
        {"name": "No Gene Decoder", "gene_decoder_bool": False},
        {
            "name": "Minimal Config",
            "confidence_token": False,
            "batch_encoder": False,
            "gene_decoder_bool": False,
        },
    ]

    results = []

    for i, test_config in enumerate(test_configs, 1):
        print(f"\nTest {i}: {test_config['name']}")

        config = base_config.copy()
        config.update({k: v for k, v in test_config.items() if k != "name"})

        try:
            set_seed(42)
            lightning_model = LightningModel(**config)
            lightning_model.eval()

            set_seed(42)
            vanilla_model = VanillaModel(**config)
            vanilla_model.eval()

            batch = create_test_batch(
                batch_size=1,
                seq_len=config["cell_set_len"],
                input_dim=config["input_dim"],
                pert_dim=config["pert_dim"],
                gene_dim=config["gene_dim"],
            )

            # Forward pass
            with torch.no_grad():
                lightning_out = lightning_model.forward(batch, padded=True)
                vanilla_out = vanilla_model.forward(batch, padded=True)

            # Extract main outputs
            if isinstance(lightning_out, tuple):
                lightning_main = lightning_out[0]
            else:
                lightning_main = lightning_out

            if isinstance(vanilla_out, tuple):
                vanilla_main = vanilla_out[0]
            else:
                vanilla_main = vanilla_out

            # Compare
            max_diff = torch.abs(lightning_main - vanilla_main).max().item()
            success = max_diff < 1e-4

            print(f"   {'PASSED' if success else 'FAILED'} Max difference: {max_diff:.2e}")
            results.append(success)

        except Exception as e:
            print(f"   Error: {e}")
            results.append(False)

    success_rate = sum(results) / len(results)
    print(f"\n📊 Quick Tests: {success_rate:.1%} passed ({sum(results)}/{len(results)})")

    return success_rate == 1.0


if __name__ == "__main__":
    print("Model Implementation Comparison")
    print("=" * 50)

    try:
        # Run main comparison
        main_success = compare_implementations()

        # Run quick tests
        quick_success = run_quick_tests()

        print("\nFinal Results:")
        print(f"   Main comparison: {'PASSED' if main_success else 'FAILED'}")
        print(f"   Quick tests: {'PASSED' if quick_success else 'FAILED'}")

        if main_success and quick_success:
            print("\nSUCCESS: Vanilla PyTorch implementation matches Lightning implementation!")
        else:
            print("\nISSUES DETECTED: Please check implementation differences.")

    except ImportError as e:
        print(f"Import Error: {e}")
        print("Make sure both implementations are available in your Python path.")

    except Exception as e:
        print(f"Unexpected Error: {e}")
        import traceback

        traceback.print_exc()
