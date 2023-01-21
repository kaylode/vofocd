import torch
from torch import nn

class ModelWithIntgratedLoss(nn.Module): # Faster-RCNN
    """Add utilitarian functions for module to work with pipeline

    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat

    """

    def __init__(self, model: nn.Module,  device: torch.device):
        super().__init__()
        self.model = model
        self.device = device

    def forward_batch(self, batch, metrics=None):
        """
        Forward the batch through models, losses and metrics
        If some parameters are needed, it's best to include in the batch
        """
        outputs, loss, loss_dict = self.model.forward_batch(batch, self.device, is_train=True)

        if metrics is not None:
            outputs, _, _ = self.model.forward_batch(batch, self.device, is_train=False)
            for metric in metrics:
                metric.update(output=outputs['outputs'], batch=batch)

        return {
            'loss': loss,
            'loss_dict': loss_dict,
            'model_outputs': outputs
        }

    def training_step(self, batch):
        return self.forward_batch(batch)

    def evaluate_step(self, batch, metrics=None):
        return self.forward_batch(batch, metrics)

    def state_dict(self):
        return self.model.state_dict()

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
