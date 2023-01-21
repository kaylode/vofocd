from typing import Dict, List
from theseus.cv.classification.models.timm_models import BaseTimmModel
from timm.models.convnext import LayerNorm
from source.detection.models.detr_utils.misc import NestedTensor
import torch
import torch.nn as nn
import torch.nn.functional as F

def is_norm_layer(layer):
    for norm_layer in [LayerNorm, nn.BatchNorm2d]:
        if isinstance(layer, norm_layer):
            return True
    return False

class CustomBackbone(nn.Module):
    def __init__(self, backbone_name: str, return_interm_layers: bool, **kwargs):
        super().__init__()
        backbone = BaseTimmModel(
            model_name=backbone_name,
            from_pretrained=True
        )
        self.body = backbone.model
        self.return_interm_layers = return_interm_layers
        self.num_channels = backbone.feature_dim

    def freeze_norm_layers(self):
        for _ ,child in (self.body.named_children()):
            if is_norm_layer(child):
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True 
        
    def forward(self, tensor_list: NestedTensor):
        inter_features = self.body.forward_features(tensor_list.tensors)

        if len(inter_features.shape) == 5:
            xs = {
                str(k):v for k,v in enumerate(inter_features)
            }
        else:
            xs = {'0': inter_features}
  
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
