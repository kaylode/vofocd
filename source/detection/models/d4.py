from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from theseus.base.utilities.cuda import move_to, detach

from .detr_components.backbone import build_backbone
from .dn_dab_detr.DABDETR import DABDETR, PostProcess
from .dn_dab_detr.transformer import DyHeadTransformer

class D4(nn.Module):
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
        num_image_classes: int = None,
        **kwargs
    ):
        super().__init__()
        self.name = model_name

        self.classnames = classnames
        self.num_classes = num_classes
        self.freeze = freeze
        self.num_image_classes = num_image_classes

        
        backbone = build_backbone(
            backbone_name, 
            freeze_backbone=kwargs.get('freeze_backbone', False), 
            dilation=kwargs.get('dilation', True),
            return_interm_layers=True
        )
        transformer = DyHeadTransformer(
            d_model=kwargs.get('hidden_dim', 256),
            dropout=kwargs.get('dropout', 0.0),
            nhead=kwargs.get('nheads', 8),
            num_queries=num_queries,
            dim_feedforward=kwargs.get('dim_feedforward', 2048),
            num_decoder_layers=kwargs.get('dec_layers', 6),
            normalize_before=kwargs.get('pre_norm', True),
            return_intermediate_dec=True,
            query_dim=4,
            activation=kwargs.get('transformer_activation','prelu'),
            num_patterns=kwargs.get('num_patterns',0),
            bbox_mask_attn=kwargs.get('bbox_mask_attn', False)
        )

        self.model = DABDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_image_classes=num_image_classes,
            num_queries=num_queries,
            aux_loss=aux_loss,
            iter_update=True,
            query_dim=4,
            bbox_embed_diff_each_layer=False,
            random_refpoints_xy=kwargs.get('random_refpoints_xy', False),
        )
    
        self.postprocessor = PostProcess(num_select=kwargs.get('num_select', 300))

        self.scalar = kwargs.get('scalar', 5)
        self.label_noise_scale = kwargs.get('label_noise_scale', 0.2)
        self.box_noise_scale = kwargs.get('box_noise_scale', 0.4)
        self.num_patterns = kwargs.get('num_patterns', 3)

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.model

    def forward_batch(self, batch: Dict, device: torch.device):
        x = move_to(batch['inputs'], device)
        if self.training:

            if self.num_image_classes is not None:
                obj_targets = batch['obj_targets']['targets']
            else:
                obj_targets = batch['targets']
            dn_args = (
                obj_targets, 
                self.scalar,
                self.label_noise_scale,
                self.box_noise_scale,
                self.num_patterns
            )
        else:
            dn_args = self.num_patterns

        if 'bin_masks' in batch.keys():
            bin_masks =  move_to(batch['bin_masks'], device)
        else:
            bin_masks=None

        if self.num_image_classes is not None:
            local_outputs, mask_dict, global_outputs = self.model(x, dn_args=dn_args, bin_masks=bin_masks)
        else:
            local_outputs, mask_dict = self.model(x, dn_args=dn_args, bin_masks=bin_masks)
            global_outputs = None
        mask_dict = move_to(mask_dict, device)
        
        if global_outputs is not None:
            return {
                'obj_outputs': {
                    'outputs': local_outputs,
                    'mask_dict': mask_dict
                },
                'img_outputs': {
                    'outputs': global_outputs
                }
            }
        else:
            return {
                'outputs': local_outputs,
                'mask_dict': mask_dict
            }

    def postprocess(self, outputs: Dict, batch: Dict):

        if self.num_image_classes is not None:
            obj_outputs = outputs['obj_outputs']
            obj_targets = batch['obj_targets']
        else:
            obj_outputs = outputs
            obj_targets = batch

        batch_size = obj_outputs['outputs']['pred_logits'].shape[0]
        target_sizes = torch.Tensor([batch['inputs'].shape[-2:]]).repeat(batch_size, 1)

        results = self.postprocessor(
            outputs = obj_outputs['outputs'],
            target_sizes=target_sizes
        )

        denormalized_targets = obj_targets['targets']
        denormalized_targets = self.postprocessor(
            outputs = denormalized_targets,
            target_sizes=target_sizes
        )

        obj_targets['targets'] = denormalized_targets

        if self.num_image_classes is not None:

            return {
                'obj_outputs': results,
                'img_outputs': outputs['img_outputs']
            }, batch
        
        else:
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


  