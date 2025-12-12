__all__= [
    "unfreeze_last_layers",
    "FocalLoss"
]

import torch
from torch import nn
import torch.nn.functional as F


def unfreeze_last_layers(model, num_layers_to_unfreeze=5):
    """
    Gèle le backbone et dégèle les N dernières couches + classifier.

    Args:
        model: Modèle PyTorch (MobileNetV3, etc.)
        num_layers_to_unfreeze: Nombre de dernières couches à entraîner
    """
    # Geler tout le backbone par défaut
    for param in model.features.parameters():
        param.requires_grad = False

    # Dégeler les N dernières couches
    if num_layers_to_unfreeze > 0:
        for layer in model.features[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

    # Dégeler le classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Afficher les stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

class FocalLoss(nn.Module):
    """
    Focal Loss pour gérer les classes déséquilibrées.

    Args:
        alpha: Facteur de pondération pour chaque classe.
               - Si float/int : même poids pour toutes les classes
               - Si list/tensor : poids spécifique par classe (taille = num_classes)
        gamma: Facteur de focalisation (recommandé: 2.0)
        reduction: 'mean', 'sum' ou 'none'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Gérer alpha comme scalaire ou tensor
        if isinstance(alpha, (float, int)):
            self.alpha = alpha
            self.alpha_tensor = None
        else:
            # alpha est une liste ou un tensor de poids par classe
            self.alpha = None
            if isinstance(alpha, list):
                self.alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha_tensor = alpha.float()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits du modèle (batch_size, num_classes)
            targets: Labels (batch_size)
        """
        # Calcul de la cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Calcul de pt (probabilité de la classe correcte)
        pt = torch.exp(-ce_loss)

        # Calcul du terme focal
        focal_term = (1 - pt) ** self.gamma

        # Application de alpha
        if self.alpha_tensor is not None:
            # Alpha par classe : sélectionner le alpha correspondant à chaque target
            if self.alpha_tensor.device != targets.device:
                self.alpha_tensor = self.alpha_tensor.to(targets.device)
            alpha_t = self.alpha_tensor[targets]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            # Alpha scalaire
            focal_loss = self.alpha * focal_term * ce_loss

        # Réduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Loss avec pondération automatique basée sur les fréquences de classes.
    Utilise la formule: weight = (1 - beta) / (1 - beta^n)
    où n est le nombre d'échantillons par classe.
    """
    def __init__(self, samples_per_class, beta=0.9999, loss_type='focal', gamma=2.0):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.loss_type = loss_type
        self.gamma = gamma

        # Calcul des poids
        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)

        self.register_buffer('weights', weights)

    def forward(self, inputs, targets):
        if self.loss_type == 'focal':
            loss_fn = FocalLoss(alpha=self.weights, gamma=self.gamma, reduction='mean')
        else:
            loss_fn = nn.CrossEntropyLoss(weight=self.weights)

        return loss_fn(inputs, targets)
