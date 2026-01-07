# utils/config.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import yaml
from pathlib import Path

from data.class_mapper import ClassMapping

__all__ = [
    "Config",
    "load_config"
]

from utils.logger import PipelineLogger, get_logger


@dataclass
class ModelConfig:
    """Model configuration"""
    do_train: bool = True
    backbone: str = "resnet50"

@dataclass
class DatasetConfig:
    """Dataset configuration"""
    id: str = ""
    num_classes: int = 0
    only_training_labels: bool = False

@dataclass
class DataLoaderConfig:
    """DataLoader settings"""
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    num_workers: int = 4

@dataclass
class TrainConfig:
    """Training configuration"""
    patience: int = 5
    delta: float = 0.0001
    epochs: int = 70
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    progressive_unfreeze: bool = True
    unfreeze_schedule: List[int] = field(default_factory=lambda: [5, 15, 25, 35])
    classifier_lr_multiplier: int = 10

@dataclass
class TestConfig:
    """Testing configuration"""
    custom_db: bool = True
    test_db_id: str = ""


@dataclass
class Config:
    """Main configuration class"""

    # General settings
    HF_token: str = ""
    random_state: int = 43
    working_directory: Path = field(default_factory=lambda: Path("/working"))

    # Preprocessing settings
    image_size: Tuple[int, int] = (224, 224)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Device
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    test: TestConfig = field(default_factory=TestConfig)

    # Class mapping
    class_mapping: Optional[ClassMapping] = None
    allow_new_class_outside_preload: bool = True

    # Logger
    logger: Optional[PipelineLogger] = None

    def __post_init__(self):
        """Initialize class mapping and logger after dataclass creation"""
        # Convert string paths to Path objects if needed
        if isinstance(self.working_directory, str):
            self.working_directory = Path(self.working_directory)

        # Ensure paths exist or create them
        self.working_directory.mkdir(parents=True, exist_ok=True)

        if self.logger is None:
            self.logger = get_logger()

        if self.class_mapping is None:
            self.class_mapping = ClassMapping(
                self,
                None,
                self.allow_new_class_outside_preload
            )

    def to_flat_dict(self, exclude_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Convert config to flat dictionary for logging

        Args:
            exclude_keys: Keys to exclude from the dict (e.g., ['logger', 'class_mapping'])

        Returns:
            Flattened dictionary with nested keys joined by dots
        """
        exclude_keys = exclude_keys or ['logger', 'class_mapping', 'transforms']

        def flatten(obj, prefix=''):
            items = {}

            if hasattr(obj, '__dataclass_fields__'):
                # It's a dataclass
                for field_name, field_obj in obj.__dataclass_fields__.items():
                    if field_name in exclude_keys:
                        continue

                    value = getattr(obj, field_name)
                    key = f"{prefix}.{field_name}" if prefix else field_name

                    # Recursively flatten nested dataclasses
                    if hasattr(value, '__dataclass_fields__'):
                        items.update(flatten(value, key))
                    elif isinstance(value, Path):
                        items[key] = str(value)
                    elif isinstance(value, (list, tuple)) and value:
                        # Convert lists/tuples to string for logging
                        items[key] = str(value)
                    else:
                        items[key] = value

            return items

        return flatten(self)

    def log_to_mlflow(self, mlflow):
        """
        Log all config parameters to MLflow

        Args:
            mlflow: MLflow module (import mlflow)
        """
        params = self.to_flat_dict()

        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                self.logger.warning(f"Failed to log parameter {key}: {e}")

        self.logger.info(f"Logged {len(params)} parameters to MLflow")

    def print_summary(self):
        """Print a formatted summary of the configuration"""
        params = self.to_flat_dict()

        print("\n" + "=" * 60)
        print("Configuration Summary".center(60))
        print("=" * 60)

        # Group by prefix
        grouped = {}
        for key, value in params.items():
            prefix = key.split('.')[0] if '.' in key else 'general'
            if prefix not in grouped:
                grouped[prefix] = []
            grouped[prefix].append((key, value))

        for group_name, group_params in grouped.items():
            print(f"\n[{group_name.upper()}]")
            for key, value in group_params:
                # Remove prefix for cleaner display
                display_key = key.replace(f"{group_name}.", "") if '.' in key else key
                print(f"  {display_key:<40} = {value}")

        print("=" * 60 + "\n")


def _convert_paths(config_dict: Dict[str, Any], path_keys: List[str]) -> Dict[str, Any]:
    """
    Convert string paths to Path objects in nested dictionaries

    Args:
        config_dict: Configuration dictionary
        path_keys: List of keys that should be converted to Path

    Returns:
        Updated configuration dictionary
    """
    for key in path_keys:
        if key in config_dict and config_dict[key]:
            config_dict[key] = Path(config_dict[key])
    return config_dict


def load_config(yaml_path: str, logger: Optional[PipelineLogger] = None) -> Config:
    """
    Load configuration from YAML file

    Args:
        yaml_path: Path to YAML configuration file
        logger: Optional logger instance

    Returns:
        Config object populated with YAML values
    """
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Convert paths in nested dataset config
    dataset_dict = config_dict.get('dataset', {})
    if 'path' in dataset_dict and dataset_dict['path']:
        dataset_dict['path'] = Path(dataset_dict['path'])

    # Create sub-configs
    model_config = ModelConfig(**config_dict.get('model', {}))
    dataset_config = DatasetConfig(**dataset_dict)
    dataloader_config = DataLoaderConfig(**config_dict.get('dataloader', {}))
    train_config = TrainConfig(**config_dict.get('train', {}))
    test_config = TestConfig(**config_dict.get('test', {}))

    # Extract top-level configs and convert paths
    top_level = {k: v for k, v in config_dict.items()
                 if k not in ['model', 'fit', 'dataset', 'dataloader', 'optimizer', 'test']}

    # Convert string paths to Path objects
    path_fields = ['working_directory']
    top_level = _convert_paths(top_level, path_fields)

    # Create main config
    config = Config(
        **top_level,
        model=model_config,
        dataset=dataset_config,
        dataloader=dataloader_config,
        train=train_config,
        test=test_config,
        logger=logger
    )

    return config
