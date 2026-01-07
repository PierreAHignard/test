# preprocessing/pipeline.py
from typing import Any, Tuple
from preprocessing.transforms import DataAugmentation, ModelSpecificPreprocessor
from utils.config import Config

__all__ = [
    "TransformPipeline"
]

class TransformPipeline:
    """Pipeline complète : combine les 3 étapes dans le bon ordre"""

    def __init__(
        self,
        config: Config,
        transforms = None
    ):
        self.config = config

        # Instancier dans l'ordre
        self.augmentor = DataAugmentation(config, transforms=transforms)
        self.model_processor = ModelSpecificPreprocessor(config)

    def __call__(self, image: Any, bbox: Tuple[float, float, float, float] = None) -> Tuple[str, Any]:

        # Data Augmentation
        image = self.augmentor(image)

        # Model-Specific Preprocessing
        image = self.model_processor(image)

        return image