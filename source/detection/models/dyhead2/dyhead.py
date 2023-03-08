import torch.nn as nn
from .attention_layers import Scale_Aware_Layer, Spatial_Aware_Layer, Task_Aware_Layer
from collections import OrderedDict
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import numpy as np
import torch
import torch.nn.functional as F
from source.detection.models.detr_utils.misc import NestedTensor

class DyHead_Block(nn.Module):
    def __init__(self, L, S, C):
        super(DyHead_Block, self).__init__()
        # Saving all dimension sizes of F
        self.L_size = L
        self.S_size = S
        self.C_size = C

        # Inititalizing all attention layers
        self.scale_attention = Scale_Aware_Layer(s_size=self.S_size)
        self.spatial_attention = Spatial_Aware_Layer(L_size=self.L_size)
        self.task_attention = Task_Aware_Layer(num_channels=self.C_size)

    def forward(self, F_tensor):
        scale_output = self.scale_attention(F_tensor)
        spacial_output = self.spatial_attention(scale_output)
        task_output = self.task_attention(spacial_output)

        return task_output

class FPNDyHead(nn.Module):
    def __init__(self, S:int, backbone_name = 'resnet50', num_blocks=6, fpn_returned_layers=[1,2,3]):
        super(FPNDyHead, self).__init__()
        self.backbone = resnet_fpn_backbone(
            backbone_name=backbone_name, 
            pretrained=True,
            returned_layers=fpn_returned_layers
        )

        C = self.backbone.out_channels

        self.dy_blocks = nn.Sequential(OrderedDict([
            (
                'Block_{}'.format(i+1),
                DyHead_Block(len(fpn_returned_layers), S, C)
            ) for i in range(num_blocks)
        ]))

    def concat_feature_maps(self, fpn_output, masks=None):
        # Calculating median height to upsample or desample each fpn levels
        heights = []
        level_tensors = []
        for key, values in fpn_output.items():
            if key != 'pool':
                heights.append(values.shape[2])
                level_tensors.append(values)
        self.median_height = int(np.median(heights))

        # Upsample and Desampling tensors to median height and width
        for i in range(len(level_tensors)):
            level = level_tensors[i]
            # If level height is greater than median, then downsample with interpolate
            if level.shape[2] > self.median_height:
                level = F.interpolate(input=level, size=(self.median_height, self.median_height),mode='nearest')
            # If level height is less than median, then upsample
            else:
                level = F.interpolate(input=level, size=(self.median_height, self.median_height), mode='nearest')
            level_tensors[i] = level
        
        # Concating all levels with dimensions (batch_size, levels, C, H, W)
        concat_levels = torch.stack(level_tensors, dim=1)

        # Reshaping tensor from (batch_size, levels, C, H, W) to (batch_size, levels, HxW=S, C)
        concat_levels = concat_levels.flatten(start_dim=3).transpose(dim0=2, dim1=3)

        # Reshaping mask
        if masks is not None:
            masks = F.interpolate(masks.unsqueeze(1).float(), size=(self.median_height,self.median_height)).to(torch.bool)
            masks = masks.repeat(1,len(level_tensors), 1, 1)
            return concat_levels, masks

        return concat_levels
    
    def forward(self, samples: NestedTensor):
        device = samples.tensors.device
        masks = samples.mask
        fpn_outputs = self.backbone(samples.tensors)
        F_tensor, F_mask = self.concat_feature_maps(fpn_outputs, masks)
        F_tensor, F_mask = F_tensor.to(device), F_mask.to(device)
        dy_outputs = self.dy_blocks(F_tensor)
        

        bs, L, S, C = dy_outputs.shape
        w = h = self.median_height
        dy_outputs = [
            f for f in dy_outputs.permute(1, 0, 3, 2).view(L, bs, C, h, w)
        ]

        F_mask = F_mask.permute(1,0,2,3)

        return [NestedTensor(dy, mask) for dy, mask in zip(dy_outputs, F_mask)]