import torch
import torch.nn as nn
import learn2learn as l2l
from learn2learn.optim.transforms.metacurvature_transform import MetaCurvatureTransform
from .base import BaseMetaLearner

class MetaCurvature(BaseMetaLearner):
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
        """Wrap model with MetaCurvature"""
        self.model = l2l.algorithms.GBML(
            self.model,
            lr=self.fast_lr,
            transform=MetaCurvatureTransform,
            adapt_transform=False,
            first_order=self.first_order
        )

    def _clone_model(self):
        """Clone the MetaCurvature model"""
        return l2l.clone_module(self.model)

    def adapt(self, adaptation_data, adaptation_labels):
        """Adapt the model to the new task"""
        learner = self._clone_model()
        adaptation_labels_heatmap = self._prepare_heatmap_labels(adaptation_labels, adaptation_data)
        
        for _ in range(self.adaptation_steps):
            predictions = learner(adaptation_data)
            adaptation_error = self.loss_fn(
                predictions,
                adaptation_labels_heatmap,
                target_weight=torch.tensor([[1,1,1,1,0.01]]).to(predictions.device)
            )
            learner.adapt(adaptation_error)
            
        return learner

    def train_step(self, batch):
        """Perform one training step"""
        self.optimizer.zero_grad()
        
        data, labels = batch['data'], batch['label']
        (adaptation_data, adaptation_labels), (evaluation_data, evaluation_labels) = self._split_data(data, labels)
        
        # Adapt model
        learner = self.adapt(adaptation_data, adaptation_labels)
        
        # Evaluate
        predictions = learner(evaluation_data)
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