from typing import List
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from .dyrelu import h_sigmoid, DYReLU
from collections import OrderedDict
from source.detection.models.detr_utils.misc import NestedTensor

class Conv3x3Norm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv3x3Norm, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.GroupNorm(num_groups=16, num_channels=out_channels)

    def forward(self, input, **kwargs):
        x = self.conv(input.contiguous(), **kwargs)
        x = self.bn(x)
        return x


class DyConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, conv_func=Conv3x3Norm):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))

        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True))

        self.h_sigmoid = h_sigmoid()
        self.relu = DYReLU(in_channels, out_channels)
        self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.AttnConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        next_x = {}
        feature_names = list(x.keys())
        for level, name in enumerate(feature_names):

            feature = x[name]

            # offset_mask = self.offset(feature)
            # offset = offset_mask[:, :18, :, :]
            # mask = offset_mask[:, 18:, :, :].sigmoid()
            # conv_args = dict(offset=offset, mask=mask)

            temp_fea = [self.DyConv[1](feature)]
            if level > 0:
                temp_fea.append(self.DyConv[2](x[feature_names[level - 1]]))
            if level < len(x) - 1:
                temp_fea.append(F.upsample_bilinear(self.DyConv[0](x[feature_names[level + 1]]),
                                                    size=[feature.size(2), feature.size(3)]))
            attn_fea = []
            res_fea = []
            for fea in temp_fea:
                res_fea.append(fea)
                attn_fea.append(self.AttnConv(fea))

            res_fea = torch.stack(res_fea)
            spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))
            mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)
            next_x[name] = self.relu(mean_fea)

        return next_x


class DyHead(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, num_convs=6):
        super(DyHead, self).__init__()

        self.fpn = torchvision.ops.FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
            )
        
        dyhead_tower = []
        for i in range(num_convs):
            dyhead_tower.append(
                DyConv(
                    out_channels if i == 0 else out_channels,
                    out_channels,
                    conv_func=Conv3x3Norm,
                )
            )

        self.dyhead_tower = nn.Sequential(*dyhead_tower)

    def forward(self, features: List[NestedTensor]):

        """
        List of features
        """

        src_inputs = [i.tensors for i in features]
        masks = [i.mask for i in features]

        wrapped_features = OrderedDict()
        for i, f in enumerate(src_inputs):
            wrapped_features[f'feat_{i}'] = f

        x = self.fpn(wrapped_features)
        dyhead_tower = self.dyhead_tower(x)
        return [NestedTensor(i, mask) for i, mask in zip(dyhead_tower.values(), masks)]