"""
Model training script with progressive fine-tuning and multi-stage training.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from model.callbacks import EarlyStopping
from typing import Optional, List, Dict, Any

__all__ = [
    "Trainer",
    "TrainingStage"
]

class TrainingStage:
    """Configuration pour une étape d'entraînement"""
    def __init__(
        self,
        name: str,
        epochs: int,
        learning_rate: float,
        criterion: Optional[nn.Module] = None,
        weight_decay: float = 0.01,
        classifier_lr_multiplier: float = 10.0,
        train_loader = None,
        unfreeze_schedule: Optional[List[int]] = None,
        scheduler_type: str = "cosine",
        scheduler_params: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.weight_decay = weight_decay
        self.classifier_lr_multiplier = classifier_lr_multiplier
        self.train_loader = train_loader
        self.unfreeze_schedule = unfreeze_schedule or []
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}

class Trainer:
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            config,
            device,
            criterion = nn.CrossEntropyLoss(),
            training_stages: Optional[List[TrainingStage]] = None
    ):
        self.model = model
        self.default_train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.default_criterion = criterion

        # Configuration des étapes d'entraînement
        self.training_stages = training_stages
        self.current_stage_idx = 0
        self.current_stage = None

        # Configuration du fine-tuning progressif
        self.progressive_unfreeze = config.fit.progressive_unfreeze

        # Initialisation : geler le backbone, garder le classificateur entraînable
        if self.progressive_unfreeze:
            self._freeze_backbone()

        # Variables d'état
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None

        self.early_stopping = EarlyStopping(
            patience=config.optimizer.patience,
            delta=config.optimizer.delta,
            verbose=True
        )

        self.best_val_loss = float('inf')
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
            "stage": []
        }
        self.current_epoch = 0
        self.stage_epoch = 0  # Epoch dans le stage actuel

    def _freeze_backbone(self):
        """Gèle toutes les couches du backbone pré-entraîné"""
        for name, param in self.model.named_parameters():
            if 'fc' not in name and 'classifier' not in name and 'head' not in name:
                param.requires_grad = False
        print("✓ Backbone gelé (seul le classificateur est entraînable)")

    def _get_layer_groups(self):
        """
        Divise le modèle en groupes de couches pour le dégel progressif.
        Adapté pour ResNet, mais peut être modifié pour d'autres architectures.
        """
        # Pour ResNet
        if hasattr(self.model, 'layer4'):
            return [
                self.model.layer4,
                self.model.layer3,
                self.model.layer2,
                self.model.layer1,
                [self.model.conv1, self.model.bn1]
            ]
        # Pour EfficientNet
        elif hasattr(self.model, 'features'):
            features = list(self.model.features.children())
            n = len(features)
            return [
                features[int(n*0.75):],
                features[int(n*0.5):int(n*0.75)],
                features[int(n*0.25):int(n*0.5)],
                features[:int(n*0.25)]
            ]
        else:
            return [list(self.model.children())]

    def _unfreeze_layer_group(self, group_idx):
        """Dégèle un groupe de couches spécifique"""
        layer_groups = self._get_layer_groups()

        if group_idx < len(layer_groups):
            group = layer_groups[group_idx]
            if isinstance(group, list):
                for layer in group:
                    for param in layer.parameters():
                        param.requires_grad = True
            else:
                for param in group.parameters():
                    param.requires_grad = True

            print(f"✓ Groupe de couches {group_idx + 1} dégelé")
            self._update_optimizer()

    def _create_optimizer(self, learning_rate: float, weight_decay: float, classifier_lr_multiplier: float):
        """
        Crée l'optimiseur avec des learning rates différenciés.
        """
        classifier_params = []
        backbone_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'fc' in name or 'classifier' in name or 'head' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)

        param_groups = []

        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': learning_rate
            })

        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': learning_rate * classifier_lr_multiplier
            })

        return torch.optim.Adam(param_groups, weight_decay=weight_decay)

    def _create_scheduler(self, scheduler_type: str, epochs_remaining: int, scheduler_params: Dict[str, Any]):
        """Crée le scheduler en fonction du type spécifié"""
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs_remaining,
                **scheduler_params
            )
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 10),
                gamma=scheduler_params.get('gamma', 0.1)
            )
        elif scheduler_type == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_params.get('gamma', 0.95)
            )
        elif scheduler_type == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_params.get('factor', 0.1),
                patience=scheduler_params.get('patience', 5)
            )
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs_remaining
            )

    def _update_optimizer(self):
        """Met à jour l'optimiseur et le scheduler après changement de configuration"""
        if self.current_stage:
            self.optimizer = self._create_optimizer(
                self.current_stage.learning_rate,
                self.current_stage.weight_decay,
                self.current_stage.classifier_lr_multiplier
            )
            epochs_remaining = self.current_stage.epochs - self.stage_epoch
            self.scheduler = self._create_scheduler(
                self.current_stage.scheduler_type,
                epochs_remaining,
                self.current_stage.scheduler_params
            )

    def _initialize_stage(self, stage: TrainingStage):
        """Initialise une nouvelle étape d'entraînement"""
        print(f"\n{'='*60}")
        print(f"Démarrage de l'étape : {stage.name}")
        print(f"{'='*60}")
        print(f"Époques : {stage.epochs}")
        print(f"Learning rate : {stage.learning_rate:.2e}")
        print(f"Multiplicateur LR classificateur : {stage.classifier_lr_multiplier}")
        print(f"Weight decay : {stage.weight_decay}")

        self.current_stage = stage
        self.stage_epoch = 0

        # Mise à jour du criterion
        self.criterion = stage.criterion if stage.criterion else self.default_criterion

        # Mise à jour du train_loader
        self.train_loader = stage.train_loader if stage.train_loader else self.default_train_loader

        # Création de l'optimiseur et du scheduler
        self.optimizer = self._create_optimizer(
            stage.learning_rate,
            stage.weight_decay,
            stage.classifier_lr_multiplier
        )

        self.scheduler = self._create_scheduler(
            stage.scheduler_type,
            stage.epochs,
            stage.scheduler_params
        )

        # Réinitialiser l'early stopping pour cette phase
        self.early_stopping = EarlyStopping(
            patience=self.config.optimizer.patience,
            delta=self.config.optimizer.delta,
            verbose=True
        )

    def _check_unfreeze_schedule(self):
        """Vérifie si des couches doivent être dégelées à cette époque"""
        if not self.progressive_unfreeze or not self.current_stage:
            return

        for idx, epoch_threshold in enumerate(self.current_stage.unfreeze_schedule):
            if self.stage_epoch == epoch_threshold:
                self._unfreeze_layer_group(idx)
                print(f"  → Learning rate actuel du backbone: {self.optimizer.param_groups[0]['lr']:.2e}")
                if len(self.optimizer.param_groups) > 1:
                    print(f"  → Learning rate actuel du classificateur: {self.optimizer.param_groups[1]['lr']:.2e}")

    def train_epoch(self):
        """Une époque d'entraînement"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(self.train_loader, desc=f"Training ({self.current_stage.name if self.current_stage else 'Default'})"):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        train_acc = 100 * correct / total

        self.history["train_loss"].append(avg_loss)
        self.history["train_acc"].append(train_acc)
        self.history["stage"].append(self.current_stage.name if self.current_stage else "Default")

        return avg_loss, train_acc

    def validate(self):
        """Validation après chaque époque"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        self.history["val_loss"].append(avg_loss)
        self.history["val_acc"].append(accuracy)

        return avg_loss, accuracy

    def fit(self):
        """Boucle d'entraînement complète avec support multi-stage"""

        # Mode multi-stage
        if self.training_stages:
            for stage_idx, stage in enumerate(self.training_stages):
                self.current_stage_idx = stage_idx
                self._initialize_stage(stage)

                for epoch in range(stage.epochs):
                    self.stage_epoch = epoch
                    self.current_epoch += 1

                    self._check_unfreeze_schedule()

                    train_loss, train_acc = self.train_epoch()
                    val_loss, val_acc = self.validate()

                    # Mise à jour du scheduler
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.history["lr"].append(current_lr)

                    print(f"\nÉpoque globale {self.current_epoch} | Étape '{stage.name}' {epoch+1}/{stage.epochs}")
                    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                    print(f"  Learning Rate: {current_lr:.2e}")

                    # Sauvegarde du meilleur modèle
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        checkpoint = {
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'epoch': self.current_epoch,
                            'stage': stage.name,
                            'loss': val_loss,
                            'accuracy': val_acc
                        }
                        torch.save(checkpoint, self.config.working_directory / f'checkpoint_stage_{stage_idx}.tar')
                        print("  ✓ Checkpoint sauvegardé")

                    # Early stopping
                    if self.early_stopping(val_loss):
                        print(f"\n✓ Étape '{stage.name}' arrêtée à l'époque {epoch + 1}")
                        break

                print(f"\n✓ Étape '{stage.name}' terminée")

        # Mode classique (rétrocompatibilité)
        else:
            # Initialisation en mode classique
            self.criterion = self.default_criterion
            self.train_loader = self.default_train_loader
            self.optimizer = self._create_optimizer(
                self.config.fit.learning_rate,
                self.config.fit.weight_decay,
                self.config.fit.classifier_lr_multiplier
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.fit.epochs
            )

            unfreeze_schedule = self.config.fit.unfreeze_schedule if hasattr(self.config.fit, 'unfreeze_schedule') else []

            for epoch in range(self.config.fit.epochs):
                self.current_epoch = epoch

                # Vérifier si des couches doivent être dégelées
                if self.progressive_unfreeze:
                    for idx, epoch_threshold in enumerate(unfreeze_schedule):
                        if epoch == epoch_threshold:
                            self._unfreeze_layer_group(idx)

                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc = self.validate()

                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history["lr"].append(current_lr)

                print(f"Epoch {epoch+1}/{self.config.fit.epochs}")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                print(f"  Learning Rate: {current_lr:.2e}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    checkpoint = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': val_loss,
                        'accuracy': val_acc
                    }
                    torch.save(checkpoint, self.config.working_directory / 'checkpoint.tar')
                    print("  ✓ Checkpoint sauvegardé")

                if self.early_stopping(val_loss):
                    print(f"\n✓ Entraînement arrêté à l'époque {epoch + 1}")
                    break

        return self.history
