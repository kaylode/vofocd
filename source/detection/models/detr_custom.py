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

class DETRCustomBackbone(nn.Module):
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
        min_conf: float = 0.25,
        aux_loss: bool = True,
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

        self.postprocessor = PostProcess(min_conf=min_conf)
        
        self.model = DETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=num_queries,
            aux_loss=aux_loss
        )

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.model

    def forward_batch(self, batch: Dict, device: torch.device):
        x = move_to(batch['inputs'], device)
        outputs = self.model(x)
        return {
            'outputs': outputs,
        }

    def postprocess(self, outputs: Dict, batch: Dict):
        batch_size = outputs['outputs']['pred_logits'].shape[0]
        target_sizes = torch.Tensor([batch['inputs'].shape[-2:]]).repeat(batch_size, 1)

        results = self.postprocessor(
            outputs = outputs['outputs'],
            target_sizes=target_sizes
        )

        denormalized_targets = batch['targets']
        denormalized_targets = self.postprocessor(
            outputs = denormalized_targets,
            target_sizes=target_sizes
        )

        batch['targets'] = denormalized_targets
        return results, batch
    
    @torch.no_grad()
    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        """
        Inference using the model.
        adict: `Dict[str, Any]`
            dictionary of inputs
        device: `torch.device`
            current device 
        """
        outputs = self.forward_batch(adict, device)

        batch_size = outputs['outputs']['pred_logits'].shape[0]
        target_sizes = torch.Tensor([adict['inputs'].shape[-2:]]).repeat(batch_size, 1)

        results = self.postprocessor(
            outputs = outputs['outputs'],
            target_sizes=target_sizes
        )
        
        scores = []
        bboxes = []
        classids = []
        classnames = []
        for result in results:
            score = move_to(detach(result['scores']), torch.device('cpu')).numpy().tolist()
            boxes = move_to(detach(result['boxes']), torch.device('cpu')).numpy().tolist()
            classid = move_to(detach(result['labels']), torch.device('cpu')).numpy().tolist()
            scores.append(score)
            bboxes.append(boxes)
            classids.append(classid)
            if self.classnames:
                classname = [self.classnames[int(clsid)] for clsid in classid]
                classnames.append(classname)

        return {
            'boxes': bboxes,
            'labels': classids,
            'confidences': scores, 
            'names': classnames,
        }


  