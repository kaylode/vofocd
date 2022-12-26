from typing import Dict, List, Any, Optional
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from theseus.base.utilities.cuda import move_to, detach
from theseus.base.utilities.logits import logits2labels
from .detr_components.models.transformer import Transformer
from .detr_components.models.backbone import build_position_encoding, Backbone_ConvNext, Joiner
from .detr_components.models.detr import DETR

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
        num_classes: int = 1000,
        classnames: Optional[List] = None,
        freeze: bool = False,
        **kwargs
    ):
        super().__init__()
        self.name = model_name

        self.classnames = classnames
        self.num_classes = num_classes
        self.freeze = freeze

        backbone = self.build_backbone(AttributeDict({
            'hidden_dim': 256,
            'position_embedding': 'sine', #, 'learned'
            'masks': False,
            'lr_backbone': 1e-5,
            'dilation': True,
            'backbone': 'resnet50'
        }))

        transformer = self.build_transformer(
            AttributeDict({
              'hidden_dim': 256,
              'dropout': 0.1,
              'nheads': 8,
              'dim_feedforward': 2048,
              'enc_layers': 6,
              'dec_layers': 6,
              'pre_norm': True
        }))

        self.model = DETR(
            backbone,
            transformer,
            num_classes=2,
            num_queries=100,
            aux_loss=False
        )

        
    def build_backbone(self, args):
        position_embedding = build_position_encoding(args)
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks
        backbone = Backbone_ConvNext(args.backbone, train_backbone, return_interm_layers, args.dilation)
        model = Joiner(backbone, position_embedding)
        model.num_channels = backbone.num_channels
        return model

    def build_transformer(self, args):
        return Transformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
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

  