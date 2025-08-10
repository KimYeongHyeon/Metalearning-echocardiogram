import torch
import torch.nn as nn
import learn2learn as l2l
from .base import BaseMetaLearner

class ANIL(BaseMetaLearner):
    def __init__(self, 
                 network: str,
                 encoder_name: str,
                 in_channels: int,
                 classes: int,
                 activation: str,
                 fast_lr: float,
                 adaptation_steps: int,
                 std: float,
                 shot: int,
                 way: int,
                 device: str,
                 first_order: bool = True):
        self.first_order = first_order
        super().__init__(
            network=network,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            fast_lr=fast_lr,
            adaptation_steps=adaptation_steps,
            std=std,
            shot=shot,
            way=way,
            device=device
        )

    def _prepare_model(self):
        """Split model into features and classifier, wrap classifier with MAML"""
        # Split model into features and classifier
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        self.classifier = list(self.model.children())[-1]
        
        # Wrap classifier with MAML
        self.classifier = l2l.algorithms.MAML(
            self.classifier,
            lr=self.fast_lr,
            first_order=self.first_order
        )
        
        # Update optimizer for both feature extractor and classifier
        self.optimizer = torch.optim.Adam([
            {'params': self.features.parameters()},
            {'params': self.classifier.parameters()}
        ], lr=1e-3)

    def _clone_model(self):
        """Clone the classifier for adaptation"""
        return l2l.clone_module(self.classifier)

    def adapt(self, adaptation_data, adaptation_labels):
        """Adapt only the classifier to the new task"""
        # Extract features
        features = self.features(adaptation_data)
        
        # Clone and adapt classifier
        classifier = self._clone_model()
        adaptation_labels_heatmap = self._prepare_heatmap_labels(adaptation_labels, adaptation_data)
        
        for _ in range(self.adaptation_steps):
            predictions = classifier(features)
            adaptation_error = self.loss_fn(
                predictions,
                adaptation_labels_heatmap,
                target_weight=torch.tensor([[1,1,1,1,0.01]]).to(predictions.device)
            )
            classifier.adapt(adaptation_error)
            
        return classifier

    def train_step(self, batch):
        """Perform one training step"""
        self.optimizer.zero_grad()
        
        data, labels = batch['data'], batch['label']
        (adaptation_data, adaptation_labels), (evaluation_data, evaluation_labels) = self._split_data(data, labels)
        
        # Extract features for both adaptation and evaluation data
        adaptation_features = self.features(adaptation_data)
        evaluation_features = self.features(evaluation_data)
        
        # Adapt classifier
        classifier = self.adapt(adaptation_data, adaptation_labels)
        
        # Evaluate
        predictions = classifier(evaluation_features)
        evaluation_labels_heatmap = self._prepare_heatmap_labels(evaluation_labels, evaluation_data)
        
        evaluation_error = self.loss_fn(
            predictions,
            evaluation_labels_heatmap,
            target_weight=torch.tensor([[1,1,1,1,0.01]]).to(predictions.device)
        )
        
        # Backpropagate
        evaluation_error.backward()
        self.optimizer.step()
        
        return evaluation_error.item() 