from typing import List, Dict

from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision

class FPN(nn.Module):

    def __init__(self, in_channels_list=[256, 512, 1024], out_channels=512) -> None:
        super().__init__()

        assert len(in_channels_list) == 3

        self.model = torchvision.ops.FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
            )
        
        self.upsample_layer = nn.Upsample(scale_factor=2)
        self.downsample_layer = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3,3),
            stride=2,
            padding=1
        )
        
    def forward(self, features: List[torch.Tensor]):
        """
        List of features
        """

        wrapped_features = OrderedDict()
        for i, f in enumerate(features):
            wrapped_features[f'feat_{i}'] = f

        pyramids = self.model(wrapped_features)

        # # Downsampling and upsampling
        # pyd2 = self.upsample_layer(pyramids['feat_2'])
        # pyd1 = pyramids['feat_1']
        # pyd0 = self.downsample_layer(pyramids['feat_0'])


        # stacked_fmaps = torch.stack([
        #     pyd0, pyd1, pyd2
        # ], dim=1)

        # # flatten spatial dims
        # stacked_fmaps = torch.flatten(stacked_fmaps, start_dim=-2)

        return pyramids