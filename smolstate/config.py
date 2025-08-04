import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import tomllib


def load_toml(file_path):
    with open(file_path, "rb") as f:
        return tomllib.load(f)


logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader for smolstate training."""

    def __init__(self, base_path: str = "../state"):
        self.base_path = Path(base_path)
        self.config_path = self.base_path / "src/state/configs"

        # Also look for configs in smolstate directory
        self.local_config_path = Path(__file__).parent.parent / "configs"

    def load_toml_config(self, toml_path: str) -> Dict[str, Any]:
        """Load TOML configuration file (like starter.toml)."""
        return load_toml(toml_path)

    def load_yaml_config(self, yaml_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    def load_model_config(self, model_name: str = "state_sm") -> Dict[str, Any]:
        """Load model configuration."""
        # Try local config first, then fall back to state config
        local_config_file = self.local_config_path / "model" / f"{model_name}.yaml"
        if local_config_file.exists():
            return self.load_yaml_config(str(local_config_file))

        config_file = self.config_path / "model" / f"{model_name}.yaml"
        return self.load_yaml_config(str(config_file))

    def load_training_config(self, config_name: str = "default") -> Dict[str, Any]:
        """Load training configuration."""
        # Try local config first, then fall back to state config
        local_config_file = self.local_config_path / "training" / f"{config_name}.yaml"
        if local_config_file.exists():
            return self.load_yaml_config(str(local_config_file))

        config_file = self.config_path / "training" / f"{config_name}.yaml"
        return self.load_yaml_config(str(config_file))

    def load_data_config(self, config_name: str = "perturbation") -> Dict[str, Any]:
        """Load data configuration."""
        # Try local config first, then fall back to state config
        local_config_file = self.local_config_path / "data" / f"{config_name}.yaml"
        if local_config_file.exists():
            return self.load_yaml_config(str(local_config_file))

        config_file = self.config_path / "data" / f"{config_name}.yaml"
        return self.load_yaml_config(str(config_file))

    def resolve_config_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve ${...} variable references in config."""

        def resolve_value(value, full_config):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Extract variable path like "model.kwargs.cell_set_len"
                var_path = value[2:-1]
                keys = var_path.split(".")

                # Navigate through config dictionary
                result = full_config
                for key in keys:
                    if isinstance(result, dict) and key in result:
                        result = result[key]
                    else:
                        logger.warning(f"Could not resolve variable: {var_path}")
                        return value
                return result
            elif isinstance(value, dict):
                return {k: resolve_value(v, full_config) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item, full_config) for item in value]
            else:
                return value

        return resolve_value(config, config)

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries."""
        result = {}
        for config in configs:
            result = self._deep_merge(result, config)
        return result

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def apply_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply command-line style overrides to config."""
        result = config.copy()

        for key, value in overrides.items():
            # Handle nested keys like "model.kwargs.hidden_dim"
            keys = key.split(".")
            current = result

            # Navigate to parent of target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            # Set the final value
            current[keys[-1]] = value

        return result


class SmolStateConfig:
    """Main configuration class for smolstate training."""

    def __init__(
        self,
        toml_config_path: str = "../state/starter.toml",
        model_name: str = "state_sm",
        training_config: str = "default",
        data_config: str = "perturbation",
        overrides: Optional[Dict[str, Any]] = None,
        base_path: str = "../state",
    ):
        self.loader = ConfigLoader(base_path)
        self.overrides = overrides or {}
        self.toml_config_path = toml_config_path

        # Load all configs
        self.toml_config = self.loader.load_toml_config(toml_config_path)
        self.model_config = self.loader.load_model_config(model_name)
        self.training_config = self.loader.load_training_config(training_config)
        self.data_config = self.loader.load_data_config(data_config)

        # Merge all configs
        self.config = self._build_full_config()

        # Apply overrides
        if self.overrides:
            self.config = self.loader.apply_overrides(self.config, self.overrides)

        # Resolve variable references
        self.config = self.loader.resolve_config_variables(self.config)

    def _build_full_config(self) -> Dict[str, Any]:
        """Build the complete configuration."""
        # Start with data config as base
        config = self.data_config.copy()

        # Add dataset config from TOML
        config["datasets"] = self.toml_config.get("datasets", {})
        config["training_spec"] = self.toml_config.get("training", {})
        config["zeroshot"] = self.toml_config.get("zeroshot", {})
        config["fewshot"] = self.toml_config.get("fewshot", {})

        # Add model config
        config["model"] = self.model_config

        # Add training config
        config["training"] = self.training_config

        # Set some defaults
        config.setdefault("output_dir", "out")
        config.setdefault("name", "smolstate_run")
        config.setdefault("overwrite", False)

        return config

    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get model initialization kwargs."""
        model_kwargs = self.config["model"]["kwargs"].copy()

        # Add dimensions that will be set by data module
        # These will be updated after data module initialization
        model_kwargs.setdefault("input_dim", 2000)  # Will be updated
        model_kwargs.setdefault("output_dim", 2000)  # Will be updated
        model_kwargs.setdefault("pert_dim", 5120)  # From ESM2 features
        model_kwargs.setdefault("gene_dim", 2000)  # Will be updated

        return model_kwargs

    def get_data_kwargs(self) -> Dict[str, Any]:
        """Get data module initialization kwargs."""
        data_kwargs = self.config["kwargs"].copy()

        # Set toml_config_path for the data module
        data_kwargs["toml_config_path"] = self.toml_config_path

        return data_kwargs

    def get_training_kwargs(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config["training"].copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with dot notation."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Dict-like access."""
        return self.config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.config

    def to_state_compatible_config(self) -> Dict[str, Any]:
        """Convert smolstate config to state-compatible format for config.yaml."""
        state_config = {
            "data": {
                "name": "PerturbationDataModule",
                "kwargs": self.get_data_kwargs(),
                "output_dir": None,
                "debug": True,
            },
            "model": self.config["model"],
            "training": self.get_training_kwargs(),
            "wandb": {
                "entity": "your_entity_name",
                "project": "smolstate",
                "local_wandb_dir": "./wandb_logs",
                "tags": [],
            },
            "name": self.config.get("name", "smolstate_run"),
            "output_dir": self.config.get("output_dir", "out"),
            "use_wandb": False,
            "overwrite": self.config.get("overwrite", False),
            "return_adatas": False,
            "pred_adata_path": None,
            "true_adata_path": None,
        }

        # Ensure model has required fields
        if "device" not in state_config["model"]:
            state_config["model"]["device"] = "cuda"
        if "checkpoint" not in state_config["model"]:
            state_config["model"]["checkpoint"] = None

        return state_config


def create_config(
    toml_config_path: str = "../state/starter.toml",
    model_name: str = "state_sm",
    overrides: Optional[Dict[str, Any]] = None,
    base_path: str = "../state",
) -> SmolStateConfig:
    """Create a complete configuration for training."""
    return SmolStateConfig(
        toml_config_path=toml_config_path, model_name=model_name, overrides=overrides, base_path=base_path
    )


def parse_cli_overrides(args: list) -> Dict[str, Any]:
    """Parse command-line overrides in Hydra format."""
    overrides = {}

    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)

            # Try to convert to appropriate type
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "").isdigit():
                value = float(value)
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]  # Remove quotes

            overrides[key] = value

    return overrides


if __name__ == "__main__":
    # Test the config loader
    config = create_config()
    print("Model kwargs:", config.get_model_kwargs())
    print("Data kwargs:", config.get_data_kwargs())
    print("Training kwargs:", config.get_training_kwargs())
