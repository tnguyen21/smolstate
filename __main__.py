#!/usr/bin/env python3
"""
SmolState - Minimal reimplementation of state training loop.

Usage:
    python -m smolstate train [options]
    python main.py train [options]
    smolstate train [options]
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Ensure we can import from current directory
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from .config import create_config, parse_cli_overrides
from .data import create_data_module, DataConfig
from .model import StateTransitionPerturbationModel
from .train import create_trainer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_command(args):
    """Run training with the given arguments."""
    logger.info("Starting smolstate training")

    # Parse command line overrides
    overrides = parse_cli_overrides(args.overrides)
    logger.info(f"Configuration overrides: {overrides}")

    # Create configuration
    config = create_config(
        toml_config_path=args.toml_config_path, model_name=args.model, overrides=overrides, base_path=args.state_path
    )

    logger.info("Configuration loaded successfully")

    # Setup output directory
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory {output_dir} exists and is not empty. Use --overwrite to proceed.")

    logger.info(f"Output directory: {output_dir}")

    # Set random seed
    torch.manual_seed(config.get_training_kwargs().get("train_seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.get_training_kwargs().get("train_seed", 42))

    # Create data module
    logger.info("Setting up data module...")
    data_module = create_data_module(config.config)

    # Save data module metadata
    data_module.save_metadata(str(output_dir))

    # Get model dimensions from data
    model_dims = data_module.get_model_dims()
    logger.info(f"Model dimensions: {model_dims}")

    # Update model config with data dimensions
    updated_model_config = DataConfig.update_model_config_with_data_dims(config.config["model"], model_dims)

    # Create decoder config if needed
    model_kwargs = updated_model_config["kwargs"]
    if model_kwargs.get("gene_decoder_bool", False):
        decoder_config = DataConfig.create_decoder_config(model_dims, model_kwargs)
        model_kwargs["decoder_cfg"] = decoder_config

    # Create model
    logger.info("Creating model...")
    model = StateTransitionPerturbationModel(**model_kwargs)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)

    logger.info("Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Parameter size: {param_size_gb:.2f} GB")

    # Get dataloaders
    train_dataloader, val_dataloader = data_module.get_dataloaders()
    logger.info(f"Train batches: {len(train_dataloader)}")
    if val_dataloader:
        logger.info(f"Validation batches: {len(val_dataloader)}")

    # Update config with final model configuration
    config.config["model"] = updated_model_config

    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config.config,
        output_dir=str(output_dir),
    )

    # Save final configuration
    trainer.checkpoint_manager.save_config(config.config)

    # Resume from checkpoint if available
    if args.resume:
        checkpoint_path = args.resume
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        trainer.resume_from_checkpoint(checkpoint_path)
    elif not args.no_resume:
        # Auto-resume from latest checkpoint
        latest_checkpoint = trainer.checkpoint_manager.find_latest_checkpoint()
        if latest_checkpoint:
            logger.info(f"Auto-resuming from: {latest_checkpoint}")
            trainer.resume_from_checkpoint(latest_checkpoint)

    # Start training
    try:
        trainer.train()

        # Print training summary
        summary = trainer.get_training_summary()
        logger.info("Training completed successfully!")
        logger.info(f"Training summary: {summary}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save current state
        trainer.checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            step=trainer.global_step,
            epoch=trainer.current_epoch,
            loss=0.0,
            config=config.config,
            is_final=False,
        )
        logger.info("Saved checkpoint before exit")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SmolState Training")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--toml-config-path", default="../state/starter.toml", help="Path to TOML configuration file"
    )
    train_parser.add_argument("--model", default="state_sm", help="Model configuration name")
    train_parser.add_argument("--output-dir", default="out", help="Output directory for training artifacts")
    train_parser.add_argument("--name", default="smolstate_run", help="Experiment name")
    train_parser.add_argument("--state-path", default="../state", help="Path to state repository")
    train_parser.add_argument("--resume", help="Resume from specific checkpoint path")
    train_parser.add_argument("--no-resume", action="store_true", help="Don't auto-resume from latest checkpoint")
    train_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    train_parser.add_argument("overrides", nargs="*", help="Configuration overrides (e.g., training.max_steps=1000)")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system info")

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "info":
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
