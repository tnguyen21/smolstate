import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from geomloss import SamplesLoss

logger = logging.getLogger(__name__)


class LatentToGeneDecoder(nn.Module):
    """Decoder module to transform latent embeddings back to gene expression space."""

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims: List[int] = [512, 1024],
        dropout: float = 0.1,
        residual_decoder: bool = False,
    ):
        super().__init__()
        self.residual_decoder = residual_decoder

        if residual_decoder:
            self.blocks = nn.ModuleList()
            input_dim = latent_dim

            for hidden_dim in hidden_dims:
                block = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.blocks.append(block)
                input_dim = hidden_dim

            self.final_layer = nn.Sequential(nn.Linear(input_dim, gene_dim), nn.ReLU())
        else:
            layers = []
            input_dim = latent_dim

            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(input_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
                input_dim = hidden_dim

            layers.extend([nn.Linear(input_dim, gene_dim), nn.ReLU()])
            self.decoder = nn.Sequential(*layers)

    def gene_dim(self):
        if self.residual_decoder:
            return self.final_layer[0].out_features
        else:
            for module in reversed(self.decoder):
                if isinstance(module, nn.Linear):
                    return module.out_features
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_decoder:
            block_outputs = []
            current = x

            for i, block in enumerate(self.blocks):
                output = block(current)
                if i >= 1 and i % 2 == 1:
                    residual_idx = i - 1
                    output = output + block_outputs[residual_idx]

                block_outputs.append(output)
                current = output

            return self.final_layer(current)
        else:
            return self.decoder(x)


class CombinedLoss(nn.Module):
    """Combined Sinkhorn + Energy loss"""

    def __init__(self, sinkhorn_weight=0.001, energy_weight=1.0, blur=0.05):
        super().__init__()
        self.sinkhorn_weight = sinkhorn_weight
        self.energy_weight = energy_weight
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur)
        self.energy_loss = SamplesLoss(loss="energy", blur=blur)

    def forward(self, pred, target):
        sinkhorn_val = self.sinkhorn_loss(pred, target)
        energy_val = self.energy_loss(pred, target)
        return self.sinkhorn_weight * sinkhorn_val + self.energy_weight * energy_val


class ConfidenceToken(nn.Module):
    """Learnable confidence token that predicts expected loss value."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.confidence_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.confidence_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU(),
        )

    def append_confidence_token(self, seq_input: torch.Tensor) -> torch.Tensor:
        batch_size = seq_input.size(0)
        confidence_tokens = self.confidence_token.expand(batch_size, -1, -1)
        return torch.cat([seq_input, confidence_tokens], dim=1)

    def extract_confidence_prediction(
        self, transformer_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        main_output = transformer_output[:, :-1, :]
        confidence_output = transformer_output[:, -1:, :]
        confidence_pred = self.confidence_projection(confidence_output).squeeze(-1)
        return main_output, confidence_pred


class PerturbationModel(nn.Module, ABC):
    """Base class for perturbation models."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: Optional[int] = None,
        dropout: float = 0.1,
        output_space: str = "gene",
        gene_dim: int = 5000,
        **kwargs,
    ):
        super().__init__()

        # Core architecture settings
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pert_dim = pert_dim
        self.batch_dim = batch_dim
        self.gene_dim = gene_dim
        self.dropout = dropout
        self.output_space = output_space

        # Store hyperparameters for compatibility
        self.hparams = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "pert_dim": pert_dim,
            "batch_dim": batch_dim,
            "dropout": dropout,
            "output_space": output_space,
            "gene_dim": gene_dim,
            **kwargs,
        }

        # Build decoder if specified
        decoder_cfg = kwargs.get("decoder_cfg")
        if decoder_cfg and kwargs.get("gene_decoder_bool", True):
            self.gene_decoder = LatentToGeneDecoder(**decoder_cfg)
        else:
            self.gene_decoder = None

        # Build the main networks
        self._build_networks()

    @abstractmethod
    def _build_networks(self):
        """Build the core neural network components."""
        pass


class StateTransitionPerturbationModel(PerturbationModel):
    """
    State transition model using transformer backbone with distributional losses.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: Optional[int] = None,
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        transformer_backbone_key: str = "GPT2",
        transformer_backbone_kwargs: Optional[dict] = None,
        output_space: str = "gene",
        gene_dim: Optional[int] = None,
        **kwargs,
    ):
        # Store configuration
        self.predict_residual = predict_residual
        self.distributional_loss = distributional_loss
        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs or {}

        # Architecture parameters
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.cell_sentence_len = kwargs.get("cell_set_len", 256)
        self.use_basal_projection = kwargs.get("use_basal_projection", True)

        # Loss parameters
        self.decoder_loss_weight = kwargs.get("decoder_weight", 1.0)
        self.regularization = kwargs.get("regularization", 0.0)
        self.detach_decoder = kwargs.get("detach_decoder", False)

        # Update transformer kwargs
        self.transformer_backbone_kwargs["n_positions"] = (
            self.cell_sentence_len + kwargs.get("extra_tokens", 0)
        )

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            batch_dim=batch_dim,
            output_space=output_space,
            gene_dim=gene_dim,
            **kwargs,
        )

        # Build loss function
        self._build_loss_function(kwargs)

        # Add batch encoder if needed
        if kwargs.get("batch_encoder", False) and batch_dim is not None:
            self.batch_encoder = nn.Embedding(
                num_embeddings=batch_dim, embedding_dim=hidden_dim
            )
        else:
            self.batch_encoder = None

        # Add confidence token if requested
        if kwargs.get("confidence_token", False):
            self.confidence_token = ConfidenceToken(
                hidden_dim=hidden_dim, dropout=self.dropout
            )
            self.confidence_loss_fn = nn.MSELoss()
        else:
            self.confidence_token = None
            self.confidence_loss_fn = None

        # ReLU for gene space outputs
        embed_key = kwargs.get("embed_key")
        is_gene_space = embed_key == "X_hvg" or embed_key is None
        if is_gene_space or self.gene_decoder is None:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.Identity()

        # Freeze perturbation backbone if requested
        if kwargs.get("freeze_pert_backbone", False):
            for param in self.transformer_backbone.parameters():
                param.requires_grad = False
            for param in self.project_out.parameters():
                param.requires_grad = False

    def _build_loss_function(self, kwargs):
        """Build the distributional loss function."""
        blur = kwargs.get("blur", 0.05)
        loss_name = kwargs.get("loss", "energy")

        if loss_name == "energy":
            self.loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
        elif loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_name == "se":
            sinkhorn_weight = kwargs.get("sinkhorn_weight", 0.01)
            energy_weight = kwargs.get("energy_weight", 1.0)
            self.loss_fn = CombinedLoss(
                sinkhorn_weight=sinkhorn_weight, energy_weight=energy_weight, blur=blur
            )
        elif loss_name == "sinkhorn":
            self.loss_fn = SamplesLoss(loss="sinkhorn", blur=blur)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    def _build_networks(self):
        """Build the neural network components."""
        from state.tx.models.utils import (
            build_mlp,
            get_activation_class,
            get_transformer_backbone,
        )

        activation_class = get_activation_class(self.hparams.get("activation", "gelu"))

        # Perturbation encoder
        self.pert_encoder = build_mlp(
            in_dim=self.pert_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=activation_class,
        )

        # Basal expression encoder
        if self.use_basal_projection:
            self.basal_encoder = build_mlp(
                in_dim=self.input_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=activation_class,
            )
        else:
            self.basal_encoder = nn.Linear(self.input_dim, self.hidden_dim)

        # Transformer backbone
        self.transformer_backbone, self.transformer_model_dim = (
            get_transformer_backbone(
                self.transformer_backbone_key,
                self.transformer_backbone_kwargs,
            )
        )

        # Output projection
        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=activation_class,
        )

        # Final processing for 'all' output space
        if self.output_space == "all":
            self.final_down_then_up = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 8),
                nn.GELU(),
                nn.Linear(self.output_dim // 8, self.output_dim),
            )

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """Embed the perturbation input."""
        return self.pert_encoder(pert)

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Embed the basal expression input."""
        return self.basal_encoder(expr)

    def forward(self, batch: dict, padded: bool = True) -> torch.Tensor:
        """
        Main forward pass.

        Args:
            batch: Dictionary containing input tensors
            padded: Whether inputs are padded to fixed length

        Returns:
            Output tensor (and confidence prediction if confidence token is used)
        """
        # Reshape inputs based on padding
        if padded:
            pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(
                -1, self.cell_sentence_len, self.input_dim
            )
        else:
            pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)

        # Encode inputs
        pert_embedding = self.encode_perturbation(pert)
        control_cells = self.encode_basal_expression(basal)

        # Combine encodings
        seq_input = pert_embedding + control_cells

        # Add batch embeddings if available
        if self.batch_encoder is not None:
            batch_indices = batch["batch"]
            if batch_indices.dim() > 1 and batch_indices.size(-1) == self.batch_dim:
                batch_indices = batch_indices.argmax(-1)

            if padded:
                batch_indices = batch_indices.reshape(-1, self.cell_sentence_len)
            else:
                batch_indices = batch_indices.reshape(1, -1)

            batch_embeddings = self.batch_encoder(batch_indices.long())
            seq_input = seq_input + batch_embeddings

        # Add confidence token if present
        confidence_pred = None
        if self.confidence_token is not None:
            seq_input = self.confidence_token.append_confidence_token(seq_input)

        # Forward pass through transformer
        if self.hparams.get("mask_attn", False):
            batch_size, seq_length, _ = seq_input.shape
            device = seq_input.device

            base = torch.eye(seq_length, device=device).view(1, seq_length, seq_length)
            attn_mask = base.repeat(batch_size, 1, 1)

            outputs = self.transformer_backbone(
                inputs_embeds=seq_input, attention_mask=attn_mask
            )
            transformer_output = outputs.last_hidden_state
        else:
            transformer_output = self.transformer_backbone(
                inputs_embeds=seq_input
            ).last_hidden_state

        # Extract confidence prediction if used
        if self.confidence_token is not None:
            res_pred, confidence_pred = (
                self.confidence_token.extract_confidence_prediction(transformer_output)
            )
        else:
            res_pred = transformer_output

        # Generate output predictions
        if self.predict_residual and self.output_space == "all":
            out_pred = self.project_out(res_pred) + basal
            out_pred = self.final_down_then_up(out_pred)
        elif self.predict_residual:
            out_pred = self.project_out(res_pred + control_cells)
        else:
            out_pred = self.project_out(res_pred)

        # Apply ReLU if needed
        out_pred = self.relu(out_pred)

        # Reshape output
        output = out_pred.reshape(-1, self.output_dim)

        if confidence_pred is not None:
            return output, confidence_pred
        else:
            return output

    def compute_loss(
        self, batch: Dict[str, torch.Tensor], padded: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss for training.

        Args:
            batch: Input batch dictionary
            padded: Whether inputs are padded

        Returns:
            Dictionary containing loss components
        """
        losses = {}

        # Forward pass
        if self.confidence_token is not None:
            pred, confidence_pred = self.forward(batch, padded=padded)
        else:
            pred = self.forward(batch, padded=padded)
            confidence_pred = None

        target = batch["pert_cell_emb"]

        # Reshape for loss computation
        if padded:
            pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
            target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        else:
            pred = pred.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)

        # Main loss
        main_loss = self.loss_fn(pred, target).nanmean()
        losses["main_loss"] = main_loss
        total_loss = main_loss

        # Decoder loss
        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]

            if self.detach_decoder:
                if np.random.rand() < 0.1:
                    latent_preds = target.reshape_as(pred).detach()
                else:
                    latent_preds = pred.detach()
            else:
                latent_preds = pred

            # Compute decoder predictions and loss
            pert_cell_counts_preds = self.gene_decoder(latent_preds)
            if padded:
                gene_targets = gene_targets.reshape(
                    -1, self.cell_sentence_len, self.gene_decoder.gene_dim()
                )
            else:
                gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

            decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()
            losses["decoder_loss"] = decoder_loss
            total_loss = total_loss + self.decoder_loss_weight * decoder_loss

        # Confidence loss
        if confidence_pred is not None:
            loss_target = total_loss.detach().clone().unsqueeze(0) * 10

            if confidence_pred.dim() == 2:
                loss_target = loss_target.unsqueeze(0).expand(
                    confidence_pred.size(0), 1
                )
            else:
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))

            confidence_loss = self.confidence_loss_fn(
                confidence_pred.squeeze(), loss_target.squeeze()
            )
            losses["confidence_loss"] = confidence_loss
            total_loss = total_loss + 0.1 * confidence_loss

        # Regularization loss
        if self.regularization > 0.0:
            ctrl_cell_emb = batch["ctrl_cell_emb"].reshape_as(pred)
            delta = pred - ctrl_cell_emb
            l1_loss = torch.abs(delta).mean()
            losses["l1_regularization"] = l1_loss
            total_loss = total_loss + self.regularization * l1_loss

        losses["total_loss"] = total_loss
        return losses

    def predict(self, batch: dict, padded: bool = True) -> dict:
        """
        Generate predictions for inference.

        Args:
            batch: Input batch dictionary
            padded: Whether inputs are padded

        Returns:
            Dictionary containing predictions and metadata
        """
        with torch.no_grad():
            if self.confidence_token is not None:
                latent_output, confidence_pred = self.forward(batch, padded=padded)
            else:
                latent_output = self.forward(batch, padded=padded)
                confidence_pred = None

            output_dict = {
                "preds": latent_output,
                "pert_cell_emb": batch.get("pert_cell_emb", None),
                "pert_cell_counts": batch.get("pert_cell_counts", None),
                "pert_name": batch.get("pert_name", None),
                "celltype_name": batch.get("cell_type", None),
                "batch": batch.get("batch", None),
                "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
                "pert_cell_barcode": batch.get("pert_cell_barcode", None),
                "ctrl_cell_barcode": batch.get("ctrl_cell_barcode", None),
            }

            if confidence_pred is not None:
                output_dict["confidence_pred"] = confidence_pred

            if self.gene_decoder is not None:
                pert_cell_counts_preds = self.gene_decoder(latent_output)
                output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

            return output_dict


# Simple training utilities
class ModelTrainer:
    """Simple trainer for the perturbation model."""

    def __init__(
        self, model: StateTransitionPerturbationModel, optimizer: torch.optim.Optimizer
    ):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, batch: dict, padded: bool = True) -> dict:
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        losses = self.model.compute_loss(batch, padded=padded)
        total_loss = losses["total_loss"]

        total_loss.backward()
        self.optimizer.step()

        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}

    def eval_step(self, batch: dict, padded: bool = True) -> dict:
        """Perform a single evaluation step."""
        self.model.eval()

        with torch.no_grad():
            losses = self.model.compute_loss(batch, padded=padded)

        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
