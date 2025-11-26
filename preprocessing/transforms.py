import torch
import torchvision.transforms as T
from typing import Dict, Any, Tuple
from utils.config import Config
from PIL import Image

# ============ ÉTAPE 1: PRE_PREPROCESS ============

class PreProcessor:
    """Étape 1: Standardisation des données"""

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, image: Any) -> Any:
        # Traiter l'image
        # Convertir en PIL si nécessaire
        if isinstance(image, str):
            image = Image.open(image)

        # Convertir en RGB (supprimer RGBA si présent)
        if image.mode in ['RGBA', 'LA']:
            rgb = Image.new('RGB', image.size, (255, 255, 255))
            rgb.paste(image, mask=image.split()[-1] if 'A' in image.mode else None)
            image = rgb
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        # Redimensionner
        image = image.resize(self.config.image_size)

        # Convertir en tensor float32
        image = T.ToTensor()(image)

        return image


# ============ ÉTAPE 2: DATA AUGMENTATION ============

class DataAugmentation:
    """Étape 2: Augmentation des données (seulement train)"""

    def __init__(self, config: Config, is_train: bool = True):
        self.config = config
        self.is_train = is_train

        if is_train:
            transforms = self.config.transforms
        else:
            transforms = [] # Pas d'augmentation en eval

        self.transforms = T.Compose(transforms)

    def __call__(self, image: Any) -> Tuple[str, Any]:
        if self.is_train:
            # Convertir tensor -> PIL pour torchvision.transforms
            if isinstance(image, torch.Tensor):
                image = T.ToPILImage()(image)

            # Appliquer augmentations
            image = self.transforms(image)

            # Reconvertir en tensor
            image = T.ToTensor()(image)

        return image


# ============ ÉTAPE 3: MODEL-SPECIFIC PREPROCESS ============

class ModelSpecificPreprocessor:
    """Étape 3: Normalisation spécifique au modèle"""

    def __init__(self, config: Config, model_name: str = 'resnet50'):
        self.config = config
        self.model_name = model_name

        # Normalisation ImageNet standard pour plupart des modèles
        self.normalize = T.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std
        )

    def __call__(self, image: Any) -> Tuple[str, Any]:
        # S'assurer que c'est un tensor
        if not isinstance(image, torch.Tensor):
            image = T.ToTensor()(image)

        # S'assurer que c'est 3 canaux (C, H, W)
        if image.dim() == 2:  # Grayscale
            image = image.unsqueeze(0).repeat(3, 1, 1)

        # Normaliser
        image = self.normalize(image)

        return image
