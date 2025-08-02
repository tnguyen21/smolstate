import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import json
import shutil

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpointing for smolstate training."""

    def __init__(self, output_dir: str, save_freq: int = 2000, keep_n_checkpoints: int = 3):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_freq = save_freq
        self.keep_n_checkpoints = keep_n_checkpoints

        self.saved_checkpoints = []

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        step: int = 0,
        epoch: int = 0,
        loss: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
        is_final: bool = False,
        is_best: bool = False,
    ) -> str:
        """Save a checkpoint."""

        # Determine checkpoint filename
        if is_final:
            checkpoint_name = "final.ckpt"
        elif is_best:
            checkpoint_name = "best.ckpt"
        else:
            checkpoint_name = f"step_{step}.ckpt"

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Prepare checkpoint data
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "config": config,
        }

        # Add scheduler state if available
        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

        # Add model hyperparameters if available
        if hasattr(model, "hparams"):
            checkpoint_data["hyper_parameters"] = model.hparams

        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Also save as "last.ckpt" for easy resuming
        if not is_final and not is_best:
            last_checkpoint_path = self.checkpoint_dir / "last.ckpt"
            shutil.copy2(checkpoint_path, last_checkpoint_path)

        # Track saved checkpoints for cleanup
        if not is_final and not is_best:
            self.saved_checkpoints.append(checkpoint_path)
            self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if len(self.saved_checkpoints) > self.keep_n_checkpoints:
            # Sort by modification time
            self.saved_checkpoints.sort(key=lambda x: x.stat().st_mtime)

            # Remove oldest checkpoints
            while len(self.saved_checkpoints) > self.keep_n_checkpoints:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        logger.info(f"  Step: {checkpoint.get('step', 'unknown')}")
        logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"  Loss: {checkpoint.get('loss', 'unknown')}")

        return checkpoint

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the checkpoint directory."""
        last_checkpoint = self.checkpoint_dir / "last.ckpt"
        if last_checkpoint.exists():
            return str(last_checkpoint)

        # Look for step checkpoints
        step_checkpoints = list(self.checkpoint_dir.glob("step_*.ckpt"))
        if step_checkpoints:
            # Sort by step number
            step_checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))
            return str(step_checkpoints[-1])

        return None

    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """Get information about a checkpoint without loading the full model."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            return {}

        # Load only metadata
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        return {
            "step": checkpoint.get("step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "loss": checkpoint.get("loss", 0.0),
            "config": checkpoint.get("config", {}),
            "hyper_parameters": checkpoint.get("hyper_parameters", {}),
            "file_size": checkpoint_path.stat().st_size,
            "modified_time": checkpoint_path.stat().st_mtime,
        }

    def list_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """List all available checkpoints with their info."""
        checkpoints = {}

        for checkpoint_file in self.checkpoint_dir.glob("*.ckpt"):
            checkpoint_name = checkpoint_file.name
            checkpoints[checkpoint_name] = self.get_checkpoint_info(str(checkpoint_file))

        return checkpoints

    def save_config(self, config: Dict[str, Any], filename: str = "config.json"):
        """Save configuration to output directory."""
        config_path = self.output_dir / filename
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"Saved config: {config_path}")

    def load_config(self, filename: str = "config.json") -> Dict[str, Any]:
        """Load configuration from output directory."""
        config_path = self.output_dir / filename
        if not config_path.exists():
            return {}

        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded config: {config_path}")
        return config


def load_model_from_checkpoint(checkpoint_path: str, model_class: type, map_location: str = "auto") -> tuple:
    """Load model and training state from checkpoint."""
    if map_location == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(map_location)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model hyperparameters
    if "hyper_parameters" in checkpoint:
        model_kwargs = checkpoint["hyper_parameters"]
    elif "config" in checkpoint and "model" in checkpoint["config"]:
        model_kwargs = checkpoint["config"]["model"]["kwargs"]
    else:
        raise ValueError("No model hyperparameters found in checkpoint")

    model = model_class(**model_kwargs)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Return model and checkpoint info
    checkpoint_info = {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", 0.0),
        "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
        "scheduler_state_dict": checkpoint.get("scheduler_state_dict"),
        "config": checkpoint.get("config", {}),
    }

    return model, checkpoint_info


def save_model_for_inference(model: torch.nn.Module, output_path: str, config: Optional[Dict[str, Any]] = None):
    """Save model in inference-ready format."""
    output_path = Path(output_path)

    # Save model state and config
    save_data = {
        "model_state_dict": model.state_dict(),
        "hyper_parameters": getattr(model, "hparams", {}),
        "config": config,
    }

    torch.save(save_data, output_path)
    logger.info(f"Saved inference model: {output_path}")


if __name__ == "__main__":
    # Test checkpoint manager
    checkpoint_manager = CheckpointManager("test_output")

    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())

    # Save a test checkpoint
    checkpoint_path = checkpoint_manager.save_checkpoint(
        model=model, optimizer=optimizer, step=1000, epoch=5, loss=0.5, config={"test": True}
    )

    # Load and check
    checkpoint_info = checkpoint_manager.get_checkpoint_info(checkpoint_path)
    print("Checkpoint info:", checkpoint_info)

    # List all checkpoints
    all_checkpoints = checkpoint_manager.list_checkpoints()
    print("All checkpoints:", all_checkpoints)
