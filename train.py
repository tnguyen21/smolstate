import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import StateTransitionPerturbationModel
from checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class SmolTrainer:
    """Simple trainer for StateTransitionPerturbationModel."""

    def __init__(
        self,
        model: StateTransitionPerturbationModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "output",
        device: str = "auto",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Training config
        training_config = self.config.get("training", {})
        self.max_steps = training_config.get("max_steps", 40000)
        self.val_freq = training_config.get("val_freq", 2000)
        self.ckpt_freq = training_config.get("ckpt_every_n_steps", 2000)
        self.gradient_clip_val = training_config.get("gradient_clip_val", 10.0)
        self.train_seed = training_config.get("train_seed", 42)

        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()

        # Setup logging
        self._setup_logging()

        # Setup checkpointing
        self.checkpoint_manager = CheckpointManager(output_dir=self.output_dir, save_freq=self.ckpt_freq)

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.training_stats = {
            "train_losses": [],
            "val_losses": [],
            "learning_rates": [],
            "step_times": [],
        }

        # Set seed
        torch.manual_seed(self.train_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.train_seed)

    def _setup_optimizer(self):
        """Setup optimizer with configuration."""
        training_config = self.config.get("training", {})

        lr = float(training_config.get("lr", 1e-4))
        weight_decay = float(training_config.get("weight_decay", 0.0005))

        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)

        logger.info(f"Optimizer setup: lr={lr}, weight_decay={weight_decay}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        # Simple cosine annealing scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_steps, eta_min=1e-6)

    def _setup_logging(self):
        """Setup logging."""
        logger.info("Logging initialized")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step."""
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Forward pass and compute loss
        losses = self.model.compute_loss(batch, padded=True)
        total_loss = losses["total_loss"]

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        # Convert losses to float
        step_losses = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        step_losses["lr"] = self.scheduler.get_last_lr()[0]

        return step_losses

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single validation step."""
        self.model.eval()

        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        with torch.no_grad():
            losses = self.model.compute_loss(batch, padded=True)

        # Convert losses to float
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}

    def validate(self) -> Dict[str, float]:
        """Run full validation loop."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                step_losses = self.validation_step(batch)
                val_losses.append(step_losses)

        # Average validation losses
        avg_losses = {}
        if val_losses:
            for key in val_losses[0].keys():
                avg_losses[f"val_{key}"] = sum(d[key] for d in val_losses) / len(val_losses)

        return avg_losses

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to console."""
        # Console logging
        metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        logger.info(f"Step {step}: {metrics_str}")

        # Store for later analysis
        if "total_loss" in metrics:
            self.training_stats["train_losses"].append(metrics["total_loss"])
        if "val_total_loss" in metrics:
            self.training_stats["val_losses"].append(metrics["val_total_loss"])
        if "lr" in metrics:
            self.training_stats["learning_rates"].append(metrics["lr"])

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.max_steps} steps")
        logger.info(f"Model device: {next(self.model.parameters()).device}")
        logger.info(f"Training on device: {self.device}")

        start_time = time.time()
        train_metrics = {"total_loss": 0.0}  # Initialize to avoid UnboundLocalError

        # Training loop
        while self.global_step < self.max_steps:
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(self.train_dataloader):
                step_start_time = time.time()

                # Training step
                train_metrics = self.train_step(batch)

                # Track step time
                step_time = time.time() - step_start_time
                train_metrics["step_time"] = step_time
                self.training_stats["step_times"].append(step_time)

                # Log training metrics
                if self.global_step % 100 == 0:  # Log every 100 steps
                    self.log_metrics(train_metrics, self.global_step)

                # Validation
                if self.global_step % self.val_freq == 0 and self.global_step > 0:
                    val_metrics = self.validate()
                    if val_metrics:
                        combined_metrics = {**train_metrics, **val_metrics}
                        self.log_metrics(combined_metrics, self.global_step)

                        # Track best validation loss
                        if "val_total_loss" in val_metrics:
                            if val_metrics["val_total_loss"] < self.best_val_loss:
                                self.best_val_loss = val_metrics["val_total_loss"]
                                logger.info(f"New best validation loss: {self.best_val_loss:.6f}")

                # Checkpointing
                if self.global_step % self.ckpt_freq == 0 and self.global_step > 0:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=self.global_step,
                        epoch=self.current_epoch,
                        loss=train_metrics.get("total_loss", 0.0),
                        config=self.config,
                    )

                self.global_step += 1

                # Check if we've reached max steps
                if self.global_step >= self.max_steps:
                    break

            self.current_epoch += 1
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {self.current_epoch} completed in {epoch_time:.2f}s")

        # Final checkpoint
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            epoch=self.current_epoch,
            loss=train_metrics.get("total_loss", 0.0),
            config=self.config,
            is_final=True,
        )

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s ({self.global_step} steps)")

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["step"]
        self.current_epoch = checkpoint["epoch"]

        logger.info(f"Resumed training from step {self.global_step}, epoch {self.current_epoch}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        return {
            "total_steps": self.global_step,
            "total_epochs": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.training_stats["train_losses"][-1]
            if self.training_stats["train_losses"]
            else None,
            "avg_step_time": sum(self.training_stats["step_times"]) / len(self.training_stats["step_times"])
            if self.training_stats["step_times"]
            else 0,
            "final_lr": self.training_stats["learning_rates"][-1] if self.training_stats["learning_rates"] else None,
        }


def create_trainer(
    model: StateTransitionPerturbationModel,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    config: Dict[str, Any],
    output_dir: str,
) -> SmolTrainer:
    """Create a trainer instance."""
    return SmolTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        output_dir=output_dir,
    )
