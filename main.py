#!/usr/bin/env python3
"""
SmolState - Minimal reimplementation of state training loop.

Usage:
    python main.py train [options]
    python main.py train data.kwargs.toml_config_path=../state/starter.toml training.max_steps=400
    python main.py infer --adata data.h5ad --model-dir out/smolstate_run
"""

from smolstate.config import create_config, parse_cli_overrides
from smolstate.data import create_data_module, DataConfig
from smolstate.model import StateTransitionPerturbationModel
from smolstate.train import create_trainer
from smolstate.infer import run_inference

import argparse
import logging
import sys
from pathlib import Path
import yaml

import torch

# Add current directory to path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def train_command(args):
    """Run training with the given arguments."""
    logger.info("Starting smolstate training")

    # Parse command line overrides
    overrides = parse_cli_overrides(args.overrides)
    logger.info(f"Configuration overrides: {overrides}")

    config = create_config(
        toml_config_path=args.toml_config_path, model_name=args.model, overrides=overrides, base_path=args.state_path
    )

    logger.info("Configuration loaded successfully")

    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory {output_dir} exists and is not empty. Use --overwrite to proceed.")

    logger.info(f"Output directory: {output_dir}")

    torch.manual_seed(config.get_training_kwargs().get("train_seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.get_training_kwargs().get("train_seed", 42))

    logger.info("Setting up data module...")
    data_module = create_data_module(config.config)

    data_module.save_metadata(str(output_dir))

    # Get model dimensions from data
    model_dims = data_module.get_model_dims()
    logger.info(f"Model dimensions: {model_dims}")

    updated_model_config = DataConfig.update_model_config_with_data_dims(config.config["model"], model_dims)

    model_kwargs = updated_model_config["kwargs"]
    if model_kwargs.get("gene_decoder_bool", False):
        decoder_config = DataConfig.create_decoder_config(model_dims, model_kwargs)
        model_kwargs["decoder_cfg"] = decoder_config

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

    train_dataloader, val_dataloader = data_module.get_dataloaders()
    logger.info(f"Train batches: {len(train_dataloader)}")
    if val_dataloader:
        logger.info(f"Validation batches: {len(val_dataloader)}")

    config.config["model"] = updated_model_config

    logger.info("Creating trainer...")
    trainer = create_trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config.config,
        output_dir=str(output_dir),
    )

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

    try:
        trainer.train()

        summary = trainer.get_training_summary()
        logger.info("Training completed successfully!")
        logger.info(f"Training summary: {summary}")

        # Save state-compatible config.yaml to output directory
        state_config = config.to_state_compatible_config()
        config_yaml_path = output_dir / "config.yaml"
        with open(config_yaml_path, "w") as f:
            yaml.dump(state_config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved config.yaml to {config_yaml_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
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


def infer_command(args):
    """Run inference with the given arguments."""
    logger.info("Starting smolstate inference")

    try:
        # Validate input file exists
        if not Path(args.adata).exists():
            raise FileNotFoundError(f"Input AnnData file not found: {args.adata}")

        # Validate model directory exists
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

        # Determine checkpoint path
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        else:
            # Default to final.ckpt in model_dir
            checkpoint_path = model_dir / "checkpoints" / "final.ckpt"
            if not checkpoint_path.exists():
                # Try last.ckpt as fallback
                last_checkpoint = model_dir / "checkpoints" / "last.ckpt"
                if last_checkpoint.exists():
                    checkpoint_path = last_checkpoint
                    logger.info(f"Using last checkpoint: {checkpoint_path}")
                else:
                    raise FileNotFoundError(f"No checkpoint found. Tried: {checkpoint_path} and {last_checkpoint}")
            else:
                logger.info(f"Using default checkpoint: {checkpoint_path}")

        # Validate required metadata files exist
        required_files = ["config.yaml", "var_dims.pkl", "pert_onehot_map.pt"]
        missing_files = []
        for file_name in required_files:
            if not (model_dir / file_name).exists():
                missing_files.append(file_name)

        if missing_files:
            raise FileNotFoundError(f"Missing required metadata files in {model_dir}: {missing_files}")

        # Run inference
        output_path = run_inference(
            checkpoint_path=str(checkpoint_path),
            model_dir=args.model_dir,
            adata_path=args.adata,
            output_path=args.output,
            pert_col=args.pert_col,
            celltype_col=args.celltype_col,
            celltypes=args.celltypes,
            batch_size=args.batch_size,
            embed_key=args.embed_key,
        )

        logger.info(f"Inference completed successfully! Output saved to: {output_path}")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SmolState Training")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--toml-config-path", default="starter.toml", help="Path to TOML configuration file")
    train_parser.add_argument("--model", default="state_sm", help="Model configuration name")
    train_parser.add_argument("--output-dir", default="out", help="Output directory for training artifacts")
    train_parser.add_argument("--name", default="smolstate_run", help="Experiment name")
    train_parser.add_argument("--state-path", default="../state", help="Path to state repository")
    train_parser.add_argument("--resume", help="Resume from specific checkpoint path")
    train_parser.add_argument("--no-resume", action="store_true", help="Don't auto-resume from latest checkpoint")
    train_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    train_parser.add_argument("overrides", nargs="*", help="Configuration overrides (e.g., training.max_steps=1000)")

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference on AnnData file")
    infer_parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (.ckpt). If not provided, uses model_dir/checkpoints/final.ckpt",
    )
    infer_parser.add_argument("--adata", type=str, required=True, help="Path to input AnnData file (.h5ad)")
    infer_parser.add_argument("--output", type=str, help="Path to output AnnData file (.h5ad)")
    infer_parser.add_argument(
        "--model-dir", type=str, required=True, help="Path to model directory containing config.yaml and metadata files"
    )
    infer_parser.add_argument(
        "--pert-col", type=str, default="target_gene", help="Column in adata.obs for perturbation labels"
    )
    infer_parser.add_argument("--celltype-col", type=str, help="Column in adata.obs for cell type labels (optional)")
    infer_parser.add_argument("--celltypes", type=str, help="Comma-separated list of cell types to include (optional)")
    infer_parser.add_argument("--batch-size", type=int, help="Batch size for inference (optional)")
    infer_parser.add_argument("--embed-key", type=str, help="Key in adata.obsm for input features (optional)")

    # Info command
    subparsers.add_parser("info", help="Show system info")

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
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
