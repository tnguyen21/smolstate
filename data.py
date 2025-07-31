import logging
import pickle
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DataModuleWrapper:
    """Wrapper around state's PerturbationDataModule for smolstate."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_module = None
        self.var_dims = None
        self.onehot_maps = {}
        
    def setup(self) -> None:
        """Initialize the data module with configuration."""
        try:
            from .perturbation_data import PerturbationDataModule
        except ImportError:
            from perturbation_data import PerturbationDataModule
        
        # Get data configuration
        data_kwargs = self.config.get("kwargs", {})
        training_config = self.config.get("training", {})
        
        # Extract cell sentence length from model config
        try:
            cell_sentence_len = self.config["model"]["kwargs"]["cell_set_len"]
        except KeyError:
            cell_sentence_len = 256  # Default
            
        # Add batch_size and cell_sentence_len to kwargs
        data_kwargs = data_kwargs.copy()
        data_kwargs["batch_size"] = training_config.get("batch_size", 16)
        data_kwargs["cell_sentence_len"] = cell_sentence_len
        
        # CRITICAL: Set toml_config_path from the outer config
        if "toml_config_path" not in data_kwargs or data_kwargs["toml_config_path"] is None:
            # Get from the main config - find it in the config structure
            for key, value in self.config.items():
                if isinstance(value, str) and value.endswith(".toml"):
                    data_kwargs["toml_config_path"] = value
                    break
            else:
                # Fallback to default path
                data_kwargs["toml_config_path"] = "../state/starter.toml"
        
        # Override column names to match the actual H5 file format
        data_kwargs["pert_col"] = "target_gene"
        data_kwargs["batch_col"] = "batch_var" 
        data_kwargs["control_pert"] = "non-targeting"
        
        logger.info(f"Data kwargs toml_config_path: {data_kwargs.get('toml_config_path')}")
        
        # Initialize data module directly
        self.data_module = PerturbationDataModule(**data_kwargs)
        
        # Setup data module for training
        self.data_module.setup(stage="fit")
        
        # Extract dimensions and mappings
        self.var_dims = self.data_module.get_var_dims()
        self.onehot_maps = {
            "cell_type": getattr(self.data_module, "cell_type_onehot_map", {}),
            "pert": getattr(self.data_module, "pert_onehot_map", {}),
            "batch": getattr(self.data_module, "batch_onehot_map", {}),
        }
        
        logger.info(f"Data module setup complete. Var dims: {self.var_dims}")
        
    def get_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Get train and validation dataloaders."""
        if self.data_module is None:
            raise RuntimeError("Data module not setup. Call setup() first.")
            
        train_dl = self.data_module.train_dataloader()
        
        # Try to get validation dataloader
        try:
            val_dl = self.data_module.val_dataloader()
        except:
            val_dl = None
            logger.warning("No validation dataloader available")
            
        return train_dl, val_dl
        
    def get_model_dims(self) -> Dict[str, int]:
        """Get dimensions needed for model initialization."""
        if self.var_dims is None:
            raise RuntimeError("Data module not setup. Call setup() first.")
            
        # Determine gene dimension based on output space
        output_space = self.config.get("kwargs", {}).get("output_space", "all")
        if output_space == "gene":
            gene_dim = self.var_dims.get("hvg_dim", 2000)
        else:
            gene_dim = self.var_dims.get("gene_dim", 2000)
            
        return {
            "input_dim": self.var_dims.get("gene_dim", 2000),
            "output_dim": self.var_dims.get("output_dim", 2000), 
            "gene_dim": gene_dim,
            "pert_dim": self.var_dims.get("pert_dim", 5120),  # Use actual pert dimension from data
            "batch_dim": len(self.onehot_maps.get("batch", {})),
            "n_cell_types": len(self.onehot_maps.get("cell_type", {})),
            "n_perts": len(self.onehot_maps.get("pert", {})),
            "n_batches": len(self.onehot_maps.get("batch", {})),
        }
        
    def get_decoder_config(self) -> Dict[str, Any]:
        """Get decoder configuration."""
        dims = self.get_model_dims()
        model_kwargs = self.config.get("model", {}).get("kwargs", {})
        
        return {
            "latent_dim": dims["output_dim"],
            "gene_dim": dims["gene_dim"],
            "hidden_dims": model_kwargs.get("decoder_hidden_dims", [1024, 1024, 512]),
            "dropout": model_kwargs.get("decoder_dropout", 0.1),
            "residual_decoder": model_kwargs.get("residual_decoder", False),
        }
        
    def save_metadata(self, output_dir: str) -> None:
        """Save data module metadata for later use."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save onehot mappings
        with open(output_path / "cell_type_onehot_map.pkl", "wb") as f:
            pickle.dump(self.onehot_maps["cell_type"], f)
            
        torch.save(self.onehot_maps["pert"], output_path / "pert_onehot_map.pt")
        
        with open(output_path / "batch_onehot_map.pkl", "wb") as f:
            pickle.dump(self.onehot_maps["batch"], f)
            
        # Save variable dimensions
        with open(output_path / "var_dims.pkl", "wb") as f:
            pickle.dump(self.var_dims, f)
            
        # Save data module state
        with open(output_path / "data_module.torch", "wb") as f:
            self.data_module.save_state(f)
            
        logger.info(f"Saved data module metadata to {output_dir}")


class DataConfig:
    """Data configuration helper."""
    
    @staticmethod
    def update_model_config_with_data_dims(
        model_config: Dict[str, Any], 
        data_dims: Dict[str, int]
    ) -> Dict[str, Any]:
        """Update model configuration with dimensions from data."""
        model_config = model_config.copy()
        model_kwargs = model_config.get("kwargs", {})
        
        # Update core dimensions
        model_kwargs.update({
            "input_dim": data_dims["input_dim"],
            "output_dim": data_dims["output_dim"],
            "pert_dim": data_dims["pert_dim"],
            "gene_dim": data_dims["gene_dim"],
            "batch_dim": data_dims.get("batch_dim"),
        })
        
        # Add counts for categorical variables
        if "n_cell_types" in data_dims:
            model_kwargs["n_cell_types"] = data_dims["n_cell_types"]
        if "n_perts" in data_dims:
            model_kwargs["n_perts"] = data_dims["n_perts"] 
        if "n_batches" in data_dims:
            model_kwargs["n_batches"] = data_dims["n_batches"]
            
        model_config["kwargs"] = model_kwargs
        return model_config
        
    @staticmethod
    def create_decoder_config(
        data_dims: Dict[str, int],
        model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create decoder configuration from data dimensions."""
        return {
            "latent_dim": data_dims["output_dim"],
            "gene_dim": data_dims["gene_dim"],
            "hidden_dims": model_kwargs.get("decoder_hidden_dims", [1024, 1024, 512]),
            "dropout": model_kwargs.get("decoder_dropout", 0.1),
            "residual_decoder": model_kwargs.get("residual_decoder", False),
        }


def create_data_module(config: Dict[str, Any]) -> DataModuleWrapper:
    """Create and setup data module from configuration."""
    data_wrapper = DataModuleWrapper(config)
    data_wrapper.setup()
    return data_wrapper


def test_data_loading():
    """Test data loading functionality."""
    try:
        from .config import create_config
    except ImportError:
        from config import create_config
    
    # Create config
    config = create_config()
    
    # Create data module
    data_module = create_data_module(config.config)
    
    # Get dataloaders
    train_dl, val_dl = data_module.get_dataloaders()
    
    print(f"Train dataloader: {len(train_dl)} batches")
    if val_dl:
        print(f"Val dataloader: {len(val_dl)} batches")
    
    # Get model dimensions
    dims = data_module.get_model_dims()
    print(f"Model dimensions: {dims}")
    
    # Test a batch
    batch = next(iter(train_dl))
    print(f"Batch keys: {batch.keys()}")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)} (len={len(value) if hasattr(value, '__len__') else 'N/A'})")


if __name__ == "__main__":
    test_data_loading()