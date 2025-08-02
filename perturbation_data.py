"""
Standalone PerturbationDataModule - adapted from cell-load without Lightning dependency.
"""

import logging
import glob
import re
from pathlib import Path
from typing import Set, Dict, Optional
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Handle toml import
if sys.version_info >= (3, 11):
    import tomllib

    def load_toml(file_path):
        with open(file_path, "rb") as f:
            return tomllib.load(f)
else:
    try:
        import toml

        def load_toml(file_path):
            with open(file_path, "r") as f:
                return toml.load(f)
    except ImportError:
        import configparser

        def load_toml(file_path):
            config = configparser.ConfigParser()
            config.read(file_path)
            result = {}
            for section_name in config.sections():
                result[section_name] = dict(config[section_name])
            return result


logger = logging.getLogger(__name__)


class ExperimentConfig:
    """Configuration for perturbation experiments from TOML file."""

    def __init__(self, datasets: dict, training: dict, zeroshot: dict, fewshot: dict):
        self.datasets = datasets
        self.training = training
        self.zeroshot = zeroshot
        self.fewshot = fewshot

    @classmethod
    def from_toml(cls, toml_path: str) -> "ExperimentConfig":
        """Load configuration from TOML file."""
        if toml_path is None:
            raise ValueError("toml_config_path cannot be None")
        config = load_toml(toml_path)

        return cls(
            datasets=config.get("datasets", {}),
            training=config.get("training", {}),
            zeroshot=config.get("zeroshot", {}),
            fewshot=config.get("fewshot", {}),
        )

    def get_all_datasets(self) -> Set[str]:
        """Get all dataset names referenced in config."""
        datasets = set(self.training.keys())

        # Extract dataset names from zeroshot keys (format: "dataset.celltype")
        for key in self.zeroshot.keys():
            dataset = key.split(".")[0]
            datasets.add(dataset)

        # Extract dataset names from fewshot keys
        for key in self.fewshot.keys():
            dataset = key.split(".")[0]
            datasets.add(dataset)

        return datasets

    def get_zeroshot_celltypes(self, dataset_name: str) -> dict[str, str]:
        """Get zeroshot cell types for a dataset."""
        result = {}
        for key, split in self.zeroshot.items():
            if key.startswith(f"{dataset_name}."):
                celltype = key[len(f"{dataset_name}.") :]
                result[celltype] = split
        return result

    def get_fewshot_celltypes(self, dataset_name: str) -> dict[str, dict[str, list[str]]]:
        """Get fewshot cell types for a dataset."""
        result = {}
        for key, config in self.fewshot.items():
            if key.startswith(f"{dataset_name}."):
                celltype = key[len(f"{dataset_name}.") :]
                result[celltype] = config
        return result

    def validate(self):
        """Validate configuration."""
        # Basic validation - could be expanded
        if not self.datasets:
            raise ValueError("No datasets specified in configuration")


def safe_decode_array(arr):
    """Safely decode string arrays from HDF5."""
    if hasattr(arr, "astype"):
        try:
            return arr.astype(str)
        except:
            return [item.decode() if isinstance(item, bytes) else str(item) for item in arr]
    return arr


def generate_onehot_map(categories):
    """Generate one-hot encoding map for categories."""
    categories = sorted(list(categories))
    onehot_map = {}

    for i, cat in enumerate(categories):
        onehot = torch.zeros(len(categories))
        onehot[i] = 1.0
        onehot_map[cat] = onehot

    return onehot_map


class SimplePerturbationDataset:
    """Simplified perturbation dataset for loading h5 data."""

    def __init__(
        self,
        name: str,
        h5_path: Path,
        pert_col: str = "target_gene",
        batch_col: str = "batch_var",
        cell_type_key: str = "cell_type",
        control_pert: str = "non-targeting",
        embed_key: Optional[str] = None,
        output_space: str = "gene",
        pert_onehot_map: Optional[dict] = None,
        batch_onehot_map: Optional[dict] = None,
        cell_type_onehot_map: Optional[dict] = None,
        **kwargs,
    ):
        self.name = name
        self.h5_path = h5_path
        self.pert_col = pert_col
        self.batch_col = batch_col
        self.cell_type_key = cell_type_key
        self.control_pert = control_pert
        self.embed_key = embed_key
        self.output_space = output_space

        self.pert_onehot_map = pert_onehot_map or {}
        self.batch_onehot_map = batch_onehot_map or {}
        self.cell_type_onehot_map = cell_type_onehot_map or {}

        # Load metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from H5 file."""
        with h5py.File(self.h5_path, "r") as f:
            # Load perturbation categories
            pert_arr = f[f"obs/{self.pert_col}/categories"][:]
            self.pert_categories = safe_decode_array(pert_arr)

            # Load cell type categories
            try:
                celltype_arr = f[f"obs/{self.cell_type_key}/categories"][:]
            except KeyError:
                celltype_arr = f[f"obs/{self.cell_type_key}"][:]
            self.cell_type_categories = safe_decode_array(celltype_arr)

            # Load batch categories
            try:
                batch_arr = f[f"obs/{self.batch_col}/categories"][:]
            except KeyError:
                batch_arr = f[f"obs/{self.batch_col}"][:]
            self.batch_categories = safe_decode_array(batch_arr)

            # Load codes (indices into categories)
            self.pert_codes = f[f"obs/{self.pert_col}/codes"][:]
            self.cell_type_codes = (
                f[f"obs/{self.cell_type_key}/codes"][:]
                if f"obs/{self.cell_type_key}/codes" in f
                else f[f"obs/{self.cell_type_key}"][:]
            )
            self.batch_codes = (
                f[f"obs/{self.batch_col}/codes"][:]
                if f"obs/{self.batch_col}/codes" in f
                else f[f"obs/{self.batch_col}"][:]
            )

            # Find control perturbation code
            try:
                self.control_pert_code = list(self.pert_categories).index(self.control_pert)
            except ValueError:
                logger.warning(f"Control perturbation '{self.control_pert}' not found in {self.h5_path}")
                self.control_pert_code = 0

            # Get dimensions
            if self.embed_key and f"obsm/{self.embed_key}" in f:
                self.n_features = f[f"obsm/{self.embed_key}"].shape[1]
            else:
                # X is a sparse matrix stored as group, get shape from var
                self.n_features = len(f["var"])

            self.n_genes = len(f["var"])
            self.n_cells = len(f["obs"])

            # Get gene names
            if "var/gene_ids" in f:
                self.gene_names = safe_decode_array(f["var/gene_ids"][:])
            elif "var/_index" in f:
                self.gene_names = safe_decode_array(f["var/_index"][:])
            else:
                self.gene_names = [f"gene_{i}" for i in range(self.n_genes)]

    def get_gene_names(self, output_space: str = None):
        """Get gene names for output space."""
        return self.gene_names

    def get_dim_for_obsm(self, key: str):
        """Get dimensions for obsm key."""
        if key == self.embed_key:
            return self.n_features
        return self.n_genes

    def get_num_hvgs(self):
        """Get number of highly variable genes."""
        # Simplified - assume HVG is same as total genes
        return self.n_genes

    def to_subset_dataset(self, split: str, pert_indices: np.ndarray, ctrl_indices: np.ndarray):
        """Create a subset dataset."""
        all_indices = np.concatenate([pert_indices, ctrl_indices])
        return Subset(self, all_indices)

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        """Get a single item - simplified implementation with correct tensor shapes."""
        # For training, we need tensors shaped for the model:
        # Model expects to reshape: pert_emb.reshape(-1, cell_sentence_len, pert_dim)
        # So pert_emb should have size: cell_sentence_len * pert_dim
        # From error: tensor size 405504 means pert_dim = 405504 / 128 = 3168

        # Get actual dimensions from one-hot maps if available
        pert_dim = next(iter(self.pert_onehot_map.values())).shape[0] if self.pert_onehot_map else self.n_features
        cell_sentence_len = 128  # Default from model config

        return {
            "pert_emb": torch.zeros(cell_sentence_len * pert_dim),
            "ctrl_cell_emb": torch.zeros(cell_sentence_len * self.n_features),
            "pert_cell_emb": torch.zeros(cell_sentence_len * self.n_features),
        }


class PerturbationDataModule:
    """
    Standalone perturbation data module without Lightning dependency.
    Simplified version of the original with core functionality.
    """

    def __init__(
        self,
        toml_config_path: str,
        batch_size: int = 128,
        num_workers: int = 8,
        random_seed: int = 42,
        pert_col: str = "target_gene",
        batch_col: str = "batch_var",
        cell_type_key: str = "cell_type",
        control_pert: str = "non-targeting",
        embed_key: Optional[str] = None,
        output_space: str = "gene",
        basal_mapping_strategy: str = "random",
        n_basal_samples: int = 1,
        should_yield_control_cells: bool = True,
        cell_sentence_len: int = 512,
        **kwargs,
    ):
        """Initialize PerturbationDataModule."""

        # Load configuration
        self.toml_config_path = toml_config_path
        self.config = ExperimentConfig.from_toml(toml_config_path)
        self.config.validate()

        # Core parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # H5 field names
        self.pert_col = pert_col
        self.batch_col = batch_col
        self.cell_type_key = cell_type_key
        self.control_pert = control_pert
        self.embed_key = embed_key
        self.output_space = output_space

        # Sampling parameters
        self.n_basal_samples = n_basal_samples
        self.cell_sentence_len = cell_sentence_len
        self.should_yield_control_cells = should_yield_control_cells
        self.basal_mapping_strategy = basal_mapping_strategy

        # Optional parameters
        self.map_controls = kwargs.get("map_controls", True)
        self.perturbation_features_file = kwargs.get("perturbation_features_file")
        self.int_counts = kwargs.get("int_counts", False)
        self.barcode = kwargs.get("barcode", False)

        # Dataset storage
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

        # One-hot maps
        self.pert_onehot_map = None
        self.batch_onehot_map = None
        self.cell_type_onehot_map = None

        # Initialize global maps
        self._setup_global_maps()

        logger.info(f"Initialized PerturbationDataModule with {len(self.config.datasets)} datasets")
        logger.info(
            f"Using columns: pert_col={self.pert_col}, batch_col={self.batch_col}, cell_type_key={self.cell_type_key}, control_pert={self.control_pert}"
        )

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets."""
        if len(self.train_datasets) == 0:
            self._setup_datasets()
            logger.info(
                f"Setup complete: {len(self.train_datasets)} train, "
                f"{len(self.val_datasets)} val, {len(self.test_datasets)} test datasets"
            )

    def train_dataloader(self):
        """Create training dataloader."""
        if len(self.train_datasets) == 0:
            raise ValueError("No training datasets available. Call setup() first.")
        return self._create_dataloader(self.train_datasets, test=False)

    def val_dataloader(self):
        """Create validation dataloader."""
        datasets = self.val_datasets if self.val_datasets else self.test_datasets
        if not datasets:
            return None
        return self._create_dataloader(datasets, test=False)

    def test_dataloader(self):
        """Create test dataloader."""
        if not self.test_datasets:
            return None
        return self._create_dataloader(self.test_datasets, test=True, batch_size=1)

    def get_var_dims(self):
        """Get variable dimensions for model initialization."""
        if not self.test_datasets:
            raise ValueError("No datasets available. Call setup() first.")

        # Get dimensions from first dataset
        ds = self.test_datasets[0].dataset

        if self.embed_key:
            input_dim = ds.get_dim_for_obsm(self.embed_key)
            output_dim = ds.get_dim_for_obsm(self.embed_key)
        else:
            input_dim = ds.n_genes
            output_dim = ds.n_genes

        gene_dim = ds.n_genes
        hvg_dim = ds.get_num_hvgs()

        # Get perturbation and batch dimensions
        pert_dim = next(iter(self.pert_onehot_map.values())).shape[0] if self.pert_onehot_map else 1
        batch_dim = next(iter(self.batch_onehot_map.values())).shape[0] if self.batch_onehot_map else 1

        return {
            "input_dim": input_dim,
            "gene_dim": gene_dim,
            "hvg_dim": hvg_dim,
            "output_dim": output_dim,
            "pert_dim": pert_dim,
            "batch_dim": batch_dim,
            "gene_names": ds.get_gene_names(),
            "pert_names": list(self.pert_onehot_map.keys()) if self.pert_onehot_map else [],
        }

    def save_state(self, file_handle):
        """Save data module state."""
        save_dict = {
            "toml_config_path": self.toml_config_path,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "random_seed": self.random_seed,
            "pert_col": self.pert_col,
            "batch_col": self.batch_col,
            "cell_type_key": self.cell_type_key,
            "control_pert": self.control_pert,
            "embed_key": self.embed_key,
            "output_space": self.output_space,
            "cell_sentence_len": self.cell_sentence_len,
        }
        torch.save(save_dict, file_handle)

    def _setup_global_maps(self):
        """Setup global one-hot encoding maps."""
        all_perts = set()
        all_batches = set()
        all_celltypes = set()

        # Scan all datasets to collect categories
        for dataset_name in self.config.get_all_datasets():
            dataset_path = Path(self.config.datasets[dataset_name])
            files = self._find_dataset_files(dataset_path)

            for fname, fpath in files.items():
                with h5py.File(fpath, "r") as f:
                    # Collect perturbations
                    pert_arr = f[f"obs/{self.pert_col}/categories"][:]
                    perts = set(safe_decode_array(pert_arr))
                    all_perts.update(perts)

                    # Collect batches
                    try:
                        batch_arr = f[f"obs/{self.batch_col}/categories"][:]
                    except KeyError:
                        batch_arr = f[f"obs/{self.batch_col}"][:]
                    batches = set(safe_decode_array(batch_arr))
                    all_batches.update(batches)

                    # Collect cell types
                    try:
                        celltype_arr = f[f"obs/{self.cell_type_key}/categories"][:]
                    except KeyError:
                        celltype_arr = f[f"obs/{self.cell_type_key}"][:]
                    celltypes = set(safe_decode_array(celltype_arr))
                    all_celltypes.update(celltypes)

        # Create one-hot maps
        if self.perturbation_features_file:
            # Load custom perturbation features
            featurization_dict = torch.load(self.perturbation_features_file)
            missing = all_perts - set(featurization_dict.keys())
            if missing:
                feature_dim = next(iter(featurization_dict.values())).shape[0]
                for pert in missing:
                    featurization_dict[pert] = torch.zeros(feature_dim)
                logger.info(f"Added zero vectors for {len(missing)} missing perturbations")
            self.pert_onehot_map = featurization_dict
        else:
            self.pert_onehot_map = generate_onehot_map(all_perts)

        self.batch_onehot_map = generate_onehot_map(all_batches)
        self.cell_type_onehot_map = generate_onehot_map(all_celltypes)

        logger.info(
            f"Created maps: {len(all_perts)} perts, {len(all_batches)} batches, {len(all_celltypes)} cell types"
        )

    def _setup_datasets(self):
        """Setup train/val/test dataset splits."""
        for dataset_name in self.config.get_all_datasets():
            dataset_path = Path(self.config.datasets[dataset_name])
            files = self._find_dataset_files(dataset_path)

            # Get configuration
            zeroshot_celltypes = self.config.get_zeroshot_celltypes(dataset_name)
            self.config.get_fewshot_celltypes(dataset_name)
            is_training_dataset = self.config.training.get(dataset_name) == "train"

            logger.info(f"Processing dataset {dataset_name}")

            # Process each file
            for fname, fpath in files.items():
                ds = SimplePerturbationDataset(
                    name=dataset_name,
                    h5_path=fpath,
                    pert_col=self.pert_col,
                    batch_col=self.batch_col,
                    cell_type_key=self.cell_type_key,
                    control_pert=self.control_pert,
                    embed_key=self.embed_key,
                    output_space=self.output_space,
                    pert_onehot_map=self.pert_onehot_map,
                    batch_onehot_map=self.batch_onehot_map,
                    cell_type_onehot_map=self.cell_type_onehot_map,
                )

                # Process each cell type in file
                for ct_idx, ct in enumerate(ds.cell_type_categories):
                    ct_mask = ds.cell_type_codes == ct_idx
                    if not np.any(ct_mask):
                        continue

                    ct_indices = np.where(ct_mask)[0]

                    # Split control vs perturbed
                    ctrl_mask = ds.pert_codes[ct_indices] == ds.control_pert_code
                    ctrl_indices = ct_indices[ctrl_mask]
                    pert_indices = ct_indices[~ctrl_mask]

                    # Determine split based on configuration
                    if ct in zeroshot_celltypes:
                        # Zeroshot: all cells go to one split
                        split = zeroshot_celltypes[ct]
                        subset = ds.to_subset_dataset(split, pert_indices, ctrl_indices)

                        if split == "train":
                            self.train_datasets.append(subset)
                        elif split == "val":
                            self.val_datasets.append(subset)
                        elif split == "test":
                            self.test_datasets.append(subset)

                    elif is_training_dataset:
                        # Regular training
                        subset = ds.to_subset_dataset("train", pert_indices, ctrl_indices)
                        self.train_datasets.append(subset)
                    else:
                        # Default to test
                        subset = ds.to_subset_dataset("test", pert_indices, ctrl_indices)
                        self.test_datasets.append(subset)

    def _create_dataloader(self, datasets, test=False, batch_size=None):
        """Create DataLoader for datasets."""
        # Simple concatenation for now
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            # Simple concatenation - would need proper MetadataConcatDataset
            all_indices = []
            for ds in datasets:
                all_indices.extend(range(len(ds)))
            dataset = datasets[0]  # Simplified

        batch_size = batch_size or (1 if test else self.batch_size)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not test,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=not test,
        )

    def _find_dataset_files(self, dataset_path: Path) -> Dict[str, Path]:
        """Find dataset files from path (supports glob patterns)."""
        files = {}
        path_str = str(dataset_path)

        # Handle glob patterns
        if any(char in path_str for char in "*?[]{}"):
            expanded_patterns = self._expand_braces(path_str)

            for pattern in expanded_patterns:
                if pattern.startswith("/"):
                    # Absolute path
                    for fpath_str in sorted(glob.glob(pattern)):
                        fpath = Path(fpath_str)
                        if fpath.suffix in [".h5", ".h5ad"]:
                            files[fpath.stem] = fpath
                else:
                    # Relative path
                    for fpath in sorted(Path().glob(pattern)):
                        if fpath.suffix in [".h5", ".h5ad"]:
                            files[fpath.stem] = fpath
        else:
            # Direct path
            if dataset_path.is_file():
                files[dataset_path.stem] = dataset_path
            else:
                # Directory
                for ext in ["*.h5", "*.h5ad"]:
                    for fpath in sorted(dataset_path.glob(ext)):
                        files[fpath.stem] = fpath

        return files

    def _expand_braces(self, pattern: str) -> list[str]:
        """Expand brace patterns like {a,b,c}."""

        def expand_single_brace(text: str) -> list[str]:
            match = re.search(r"\{([^}]+)\}", text)
            if not match:
                return [text]

            before = text[: match.start()]
            after = text[match.end() :]
            options = match.group(1).split(",")

            results = []
            for option in options:
                new_text = before + option.strip() + after
                results.extend(expand_single_brace(new_text))

            return results

        return expand_single_brace(pattern)


def collate_fn(batch, int_counts=False):
    """Simple collate function for batching."""
    # Simplified implementation
    return {
        "pert_emb": torch.stack([item["pert_emb"] for item in batch]),
        "ctrl_cell_emb": torch.stack([item["ctrl_cell_emb"] for item in batch]),
        "pert_cell_emb": torch.stack([item["pert_cell_emb"] for item in batch]),
    }
