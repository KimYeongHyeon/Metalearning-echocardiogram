import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from abc import abstractmethod
from utils.model import get_basemodel
from utils.utils import heatmap2coor
from utils.loss import HeatmapMSELoss
from utils.heatmaps import render_gaussian_dot_f
from utils.evaluation import AverageMeter, distance_error

class BaseMetaLearner:
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
                 device: str):
        self.network = network
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        self.activation = activation
        self.fast_lr = fast_lr
        self.adaptation_steps = adaptation_steps
        self.std = std
        self.shot = shot
        self.way = way
        self.device = device
        
        self.model = get_basemodel(
            network=network,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        ).to(device)
        
        self.loss_fn = HeatmapMSELoss(use_target_weight=True)
        self.img_size = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def _prepare_heatmap_labels(self, labels, data):
        if self.img_size is None:
            self.img_size = data[0].shape[1:]
            
        heatmap = render_gaussian_dot_f(
            labels.flip(dims=[2]),  # xy 2 yx 
            torch.tensor([self.std, self.std], dtype=torch.float32).to(data.device), 
            self.img_size,
        ).to(torch.float)
        
        background = 1 - heatmap.sum(dim=1).unsqueeze(1).clip(0,1)
        heatmap = torch.concat((heatmap, background), 1)
        return heatmap

    def _split_data(self, data, labels):
        adaptation_indices = torch.zeros(data.size(0), dtype=torch.bool)
        adaptation_indices[torch.arange(self.shot*self.way)*2] = True
        evaluation_indices = ~adaptation_indices
        
        adaptation_data = data[adaptation_indices]
        adaptation_labels = labels[adaptation_indices]
        evaluation_data = data[evaluation_indices]
        evaluation_labels = labels[evaluation_indices]
        
        return (adaptation_data, adaptation_labels), (evaluation_data, evaluation_labels)

    @abstractmethod
    def _prepare_model(self):
        """Prepare the model for meta-learning (e.g., wrap with MAML, MetaSGD, etc.)"""
        pass

    @abstractmethod
    def _clone_model(self):
        """Clone the model for adaptation"""
        pass

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

    def evaluate(self, batch):
        """Common evaluation logic"""
        data, labels = batch['data'], batch['label']
        (adaptation_data, adaptation_labels), (evaluation_data, evaluation_labels) = self._split_data(data, labels)
        
        # Adapt model
        adapted_model = self.adapt(adaptation_data, adaptation_labels)
        
        # Evaluate
        predictions = adapted_model(evaluation_data)
        evaluation_labels_heatmap = self._prepare_heatmap_labels(evaluation_labels, evaluation_data)
        
        evaluation_error = self.loss_fn(
            predictions, 
            evaluation_labels_heatmap,
            target_weight=torch.tensor([[1,1,1,1,0.01]]).to(predictions.device)
        )
        
        # Calculate distance error
        sample = {
            'data': evaluation_data,
            'label': evaluation_labels
        }
        mean_distance_error = distance_error(sample, heatmap2coor(predictions[:,:4,...]))
        
        return evaluation_error.item(), mean_distance_error.mean().item() 