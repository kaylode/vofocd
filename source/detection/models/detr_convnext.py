from typing import Dict, List, Any, Optional
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from theseus.base.utilities.cuda import move_to, detach
from theseus.base.utilities.logits import logits2labels
from .detr_components.transformer import Transformer
from .detr_components.backbone import build_backbone
from .detr_components.detr import DETR, PostProcess

class DETRConvnext(nn.Module):
    """Convolution models from timm
    
    name: `str`
        timm model name
    num_classes: `int`
        number of classes
    from_pretrained: `bool` 
        whether to use timm pretrained
    classnames: `Optional[List]`
        list of classnames
    """

    def __init__(
        self,
        model_name: str,
        backbone_name: str = 'convnext_base',
        num_classes: int = 1000,
        num_queries: int = 100,
        classnames: Optional[List] = None,
        freeze: bool = False,
        **kwargs
    ):
        super().__init__()
        self.name = model_name

        self.classnames = classnames
        self.num_classes = num_classes
        self.freeze = freeze

        backbone = build_backbone(
            backbone_name, 
            hidden_dim=kwargs.get('hidden_dim', 256), 
            position_embedding=kwargs.get('position_embedding', 'sine'), 
            freeze_backbone=kwargs.get('freeze_backbone', False), 
            dilation=kwargs.get('dilation', True),
            return_interm_layers=False
        )
        

        transformer = Transformer(
            d_model=kwargs.get('hidden_dim', 256),
            dropout=kwargs.get('dropout', 0.1),
            nhead=kwargs.get('nheads', 8),
            dim_feedforward=kwargs.get('dim_feedforward', 2048),
            num_encoder_layers=kwargs.get('enc_layers', 6),
            num_decoder_layers=kwargs.get('dec_layers', 6),
            normalize_before=kwargs.get('pre_norm', True),
            return_intermediate_dec=True,
        )

        self.postprocessor = PostProcess()
        
        self.model = DETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=num_queries,
            aux_loss=False
        )

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.model

    def forward(self, batch: Dict, device: torch.device):
        x = move_to(batch['inputs'], device)
        outputs = self.model(x)
        return {
            'outputs': outputs,
        }

    def postprocess(self, outputs: Dict, batch: Dict):
        results = self.postprocessor(
            outputs = outputs['outputs'],
            target_sizes=batch['img_sizes']
        )

        denormalized_targets = batch['targets']
        denormalized_targets = self.postprocessor(
            outputs = denormalized_targets,
            target_sizes=batch['img_sizes']
        )

        batch['targets'] = denormalized_targets
        return results, batch

  