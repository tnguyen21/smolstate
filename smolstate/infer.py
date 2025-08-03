import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import torch
import yaml
from tqdm import tqdm

from .checkpoint import load_model_from_checkpoint
from .model import StateTransitionPerturbationModel

logger = logging.getLogger(__name__)


class SmolStateInference:
    """Inference engine for smolstate models."""

    def __init__(
        self,
        checkpoint_path: str,
        model_dir: str,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = None
        self.config = None
        self.var_dims = None
        self.pert_onehot_map = None

        self._load_model_and_metadata()

    def _load_model_and_metadata(self):
        """Load model checkpoint and associated metadata."""
        logger.info(f"Loading model from checkpoint: {self.checkpoint_path}")

        # Load model using smolstate's checkpoint manager
        self.model, checkpoint_info = load_model_from_checkpoint(
            self.checkpoint_path, StateTransitionPerturbationModel, map_location=str(self.device)
        )
        self.model.eval()

        logger.info(f"Model loaded successfully. Step: {checkpoint_info.get('step', 'unknown')}")
        logger.info(
            f"Model architecture: {checkpoint_info.get('config', {}).get('model', {}).get('kwargs', {}).get('transformer_backbone_key', 'unknown')}"
        )

        # Load configuration
        config_path = self.model_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.warning(f"Config file not found at {config_path}")
            self.config = {}

        # Load variable dimensions
        var_dims_path = self.model_dir / "var_dims.pkl"
        if var_dims_path.exists():
            with open(var_dims_path, "rb") as f:
                self.var_dims = pickle.load(f)
        else:
            raise FileNotFoundError(f"Variable dimensions file not found: {var_dims_path}")

        # Load perturbation mapping
        pert_map_path = self.model_dir / "pert_onehot_map.pt"
        if pert_map_path.exists():
            self.pert_onehot_map = torch.load(pert_map_path, weights_only=False)
        else:
            raise FileNotFoundError(f"Perturbation mapping file not found: {pert_map_path}")

        logger.info(f"Loaded {len(self.pert_onehot_map)} perturbations in mapping")

    def run_inference(
        self,
        adata_path: str,
        output_path: Optional[str] = None,
        pert_col: str = "target_gene",
        celltype_col: Optional[str] = None,
        celltypes: Optional[str] = None,
        batch_size: Optional[int] = None,
        embed_key: Optional[str] = None,
    ) -> str:
        """
        Run inference on AnnData file.

        Args:
            adata_path: Path to input AnnData file
            output_path: Path for output file (optional)
            pert_col: Column name for perturbations
            celltype_col: Column name for cell types (optional)
            celltypes: Comma-separated cell types to include (optional)
            batch_size: Batch size override (optional)
            embed_key: Key in adata.obsm for input features (optional)

        Returns:
            Path to output file
        """
        logger.info(f"Loading AnnData from: {adata_path}")
        adata = sc.read_h5ad(adata_path)

        # Filter by cell type if requested
        if celltype_col and celltypes:
            celltypes_list = [ct.strip() for ct in celltypes.split(",")]
            if celltype_col not in adata.obs:
                available_cols = list(adata.obs.columns)
                raise ValueError(f"Column '{celltype_col}' not found in adata.obs. Available columns: {available_cols}")

            initial_n = adata.n_obs
            adata = adata[adata.obs[celltype_col].isin(celltypes_list)].copy()
            logger.info(f"Filtered to {adata.n_obs} cells of types {celltypes_list} (from {initial_n})")

            if adata.n_obs == 0:
                raise ValueError(f"No cells found with specified cell types: {celltypes_list}")

        # Get input features
        if embed_key and embed_key in adata.obsm:
            X = adata.obsm[embed_key]
            logger.info(f"Using adata.obsm['{embed_key}'] as input: shape {X.shape}")
        elif embed_key:
            available_keys = list(adata.obsm.keys()) if hasattr(adata, "obsm") else []
            raise ValueError(f"Embedding key '{embed_key}' not found in adata.obsm. Available keys: {available_keys}")
        else:
            try:
                X = adata.X.toarray()
            except AttributeError:
                X = adata.X
            logger.info(f"Using adata.X as input: shape {X.shape}")

        # Validate input dimensions
        if X.shape[0] != adata.n_obs:
            raise ValueError(f"Input feature matrix has {X.shape[0]} samples but adata has {adata.n_obs} observations")

        expected_input_dim = self.var_dims.get("input_dim", self.var_dims.get("gene_dim", 18080))
        if X.shape[1] != expected_input_dim:
            logger.warning(f"Input feature dimension mismatch: got {X.shape[1]}, expected {expected_input_dim}")
            if X.shape[1] < expected_input_dim:
                logger.info("Padding input features with zeros")
                import numpy as np

                padding = np.zeros((X.shape[0], expected_input_dim - X.shape[1]))
                X = np.concatenate([X, padding], axis=1)
            else:
                logger.info(f"Truncating input features to first {expected_input_dim} dimensions")
                X = X[:, :expected_input_dim]

        # Check perturbation column exists
        if pert_col not in adata.obs:
            available_cols = list(adata.obs.columns)
            raise ValueError(
                f"Perturbation column '{pert_col}' not found in adata.obs. Available columns: {available_cols}"
            )

        # Prepare perturbation tensors
        pert_names = adata.obs[pert_col].values
        predictions = self._process_batches(X, pert_names, batch_size)

        # Save predictions
        adata.X = predictions
        output_path = output_path or adata_path.replace(".h5ad", "_predictions.h5ad")

        try:
            adata.write_h5ad(output_path)
            logger.info(f"Saved predictions to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save predictions to {output_path}: {e}")
            raise

        return output_path

    def _process_batches(self, X: np.ndarray, pert_names: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Process data in batches and generate predictions."""

        # Use model's cell_sentence_len as batch size if not specified
        batch_size = batch_size or self.model.cell_sentence_len
        n_samples = len(pert_names)

        logger.info(f"Processing {n_samples} samples in batches of {batch_size}")

        # Prepare perturbation tensor
        pert_dim = self.var_dims["pert_dim"]
        pert_tensor = torch.zeros((n_samples, pert_dim), device="cpu")

        # Get control perturbation for fallback
        control_pert = self.config.get("data", {}).get("kwargs", {}).get("control_pert", "non-targeting")

        # Map perturbations to tensor
        matched_count = 0
        for idx, name in enumerate(pert_names):
            if name in self.pert_onehot_map:
                pert_tensor[idx] = self.pert_onehot_map[name]
                matched_count += 1
            else:
                # Use control perturbation as fallback
                if control_pert in self.pert_onehot_map:
                    pert_tensor[idx] = self.pert_onehot_map[control_pert]
                else:
                    # Use first available perturbation
                    first_pert = list(self.pert_onehot_map.keys())[0]
                    pert_tensor[idx] = self.pert_onehot_map[first_pert]

        logger.info(f"Matched {matched_count}/{n_samples} perturbations")
        if matched_count < n_samples:
            missing = set(pert_names) - set(self.pert_onehot_map.keys())
            logger.warning(f"Missing perturbations (showing first 10): {list(missing)[:10]}")

        # Process in batches
        all_predictions = []
        n_batches = (n_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            progress_bar = tqdm(total=n_samples, desc="Processing samples", unit="samples")

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                current_batch_size = end_idx - start_idx

                # Get batch data
                X_batch = torch.tensor(X[start_idx:end_idx], dtype=torch.float32).to(self.device)
                pert_batch = pert_tensor[start_idx:end_idx].to(self.device)
                pert_names_batch = pert_names[start_idx:end_idx].tolist()

                # Pad batch if needed (for last incomplete batch)
                if current_batch_size < batch_size:
                    padding_size = batch_size - current_batch_size

                    # Pad input features
                    X_pad = torch.zeros((padding_size, X_batch.shape[1]), device=self.device)
                    X_batch = torch.cat([X_batch, X_pad], dim=0)

                    # Pad perturbation tensor
                    pert_pad = torch.zeros((padding_size, pert_batch.shape[1]), device=self.device)
                    if control_pert in self.pert_onehot_map:
                        pert_pad[:] = self.pert_onehot_map[control_pert].to(self.device)
                    else:
                        pert_pad[:, 0] = 1  # Default
                    pert_batch = torch.cat([pert_batch, pert_pad], dim=0)

                    # Extend names
                    pert_names_batch.extend([control_pert] * padding_size)

                # Prepare batch dictionary
                batch = {
                    "ctrl_cell_emb": X_batch,
                    "pert_emb": pert_batch,
                    "pert_name": pert_names_batch,
                    "batch": torch.zeros((1, batch_size), device=self.device),
                }

                # Run inference
                batch_preds = self.model.predict(batch, padded=False)

                # Extract predictions (use gene decoder output if available)
                if "pert_cell_counts_preds" in batch_preds and batch_preds["pert_cell_counts_preds"] is not None:
                    pred_tensor = batch_preds["pert_cell_counts_preds"]
                else:
                    pred_tensor = batch_preds["preds"]

                # Only keep predictions for actual samples (not padding)
                actual_preds = pred_tensor[:current_batch_size]
                all_predictions.append(actual_preds.cpu().numpy())

                progress_bar.update(current_batch_size)

            progress_bar.close()

        # Concatenate all predictions
        return np.concatenate(all_predictions, axis=0)


def run_inference(
    checkpoint_path: str,
    model_dir: str,
    adata_path: str,
    output_path: Optional[str] = None,
    pert_col: str = "target_gene",
    celltype_col: Optional[str] = None,
    celltypes: Optional[str] = None,
    batch_size: Optional[int] = None,
    embed_key: Optional[str] = None,
) -> str:
    """
    Convenience function to run inference.

    Args:
        checkpoint_path: Path to model checkpoint
        model_dir: Directory containing model metadata
        adata_path: Path to input AnnData file
        output_path: Path for output file (optional)
        pert_col: Column name for perturbations
        celltype_col: Column name for cell types (optional)
        celltypes: Comma-separated cell types to include (optional)
        batch_size: Batch size override (optional)
        embed_key: Key in adata.obsm for input features (optional)

    Returns:
        Path to output file
    """
    inference_engine = SmolStateInference(checkpoint_path, model_dir)

    return inference_engine.run_inference(
        adata_path=adata_path,
        output_path=output_path,
        pert_col=pert_col,
        celltype_col=celltype_col,
        celltypes=celltypes,
        batch_size=batch_size,
        embed_key=embed_key,
    )
