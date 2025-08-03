"""
Standalone PerturbationDataModule - adapted from cell-load without Lightning dependency.
"""

import tomllib
import logging
import glob
import re
from pathlib import Path
from typing import Set, Dict, Optional
from functools import partial

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


# Import the real PerturbationDataset from cell-load
from cell_load.dataset import PerturbationDataset, MetadataConcatDataset
from cell_load.mapping_strategies import RandomMappingStrategy, BatchMappingStrategy
from cell_load.data_modules.samplers import PerturbationBatchSampler
from cell_load.utils.data_utils import generate_onehot_map, safe_decode_array


def load_toml(file_path):
    with open(file_path, "rb") as f:
        return tomllib.load(f)


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


class PerturbationDataModule:
    """
    Standalone perturbation data module without Lightning dependency.
    Uses the real PerturbationDataset from cell-load for proper data loading.
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

        # Mapping strategy
        self.mapping_strategy_cls = {
            "batch": BatchMappingStrategy,
            "random": RandomMappingStrategy,
        }[basal_mapping_strategy]

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
        underlying_ds = self.test_datasets[0].dataset

        if self.embed_key:
            input_dim = underlying_ds.get_dim_for_obsm(self.embed_key)
            output_dim = underlying_ds.get_dim_for_obsm(self.embed_key)
        else:
            input_dim = underlying_ds.n_genes
            output_dim = underlying_ds.n_genes

        gene_dim = underlying_ds.n_genes
        try:
            hvg_dim = underlying_ds.get_num_hvgs()
        except AttributeError:
            hvg_dim = gene_dim

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
            "gene_names": underlying_ds.get_gene_names(output_space=self.output_space),
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

    def _create_base_dataset(self, dataset_name: str, fpath: Path) -> PerturbationDataset:
        """Create a base PerturbationDataset instance."""
        mapping_kwargs = {"map_controls": self.map_controls}

        return PerturbationDataset(
            name=dataset_name,
            h5_path=fpath,
            mapping_strategy=self.mapping_strategy_cls(
                random_state=self.random_seed,
                n_basal_samples=self.n_basal_samples,
                **mapping_kwargs,
            ),
            embed_key=self.embed_key,
            pert_onehot_map=self.pert_onehot_map,
            batch_onehot_map=self.batch_onehot_map,
            cell_type_onehot_map=self.cell_type_onehot_map,
            pert_col=self.pert_col,
            cell_type_key=self.cell_type_key,
            batch_col=self.batch_col,
            control_pert=self.control_pert,
            random_state=self.random_seed,
            should_yield_control_cells=self.should_yield_control_cells,
            store_raw_expression=False,
            output_space=self.output_space,
            store_raw_basal=False,
            barcode=self.barcode,
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
                ds = self._create_base_dataset(dataset_name, fpath)

                # Get metadata from the dataset
                cache = ds.metadata_cache

                # Process each cell type in file
                for ct_idx, ct in enumerate(cache.cell_type_categories):
                    ct_mask = cache.cell_type_codes == ct_idx
                    if not np.any(ct_mask):
                        continue

                    ct_indices = np.where(ct_mask)[0]

                    # Split control vs perturbed
                    ctrl_mask = cache.pert_codes[ct_indices] == cache.control_pert_code
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
        use_int_counts = "int_counts" in self.__dict__ and self.int_counts
        collate_fn = partial(PerturbationDataset.collate_fn, int_counts=use_int_counts)

        ds = MetadataConcatDataset(datasets)
        use_batch = self.basal_mapping_strategy == "batch"

        batch_size = batch_size or (1 if test else self.batch_size)

        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=batch_size,
            drop_last=False,
            cell_sentence_len=self.cell_sentence_len,
            test=test,
            use_batch=use_batch,
        )

        return DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=4 if not test else None,
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


def create_data_module(config: Dict) -> PerturbationDataModule:
    """Create and setup data module from configuration."""
    data_wrapper = PerturbationDataModule(**config)
    data_wrapper.setup()
    return data_wrapper


if __name__ == "__main__":
    """Test data loading functionality."""
    from .config import create_config

    config = create_config()
    data_module = create_data_module(config.config)

    train_dl, val_dl = data_module.train_dataloader(), data_module.val_dataloader()

    print(f"Train dataloader: {len(train_dl)} batches")
    if val_dl:
        print(f"Val dataloader: {len(val_dl)} batches")

    dims = data_module.get_var_dims()
    print(f"Model dimensions: {dims}")

    batch = next(iter(train_dl))
    print(f"Batch keys: {batch.keys()}")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)} (len={len(value) if hasattr(value, '__len__') else 'N/A'})")
