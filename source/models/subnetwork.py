import torch
import torch.nn as nn

class Subnetwork(nn.Module):
    def __init__(self, in_feat_channels, out_feat_channels) -> None:
        super().__init__()
        self.in_feat_channels = in_feat_channels
        self.out_feat_channels = out_feat_channels

        self.conv_1x1s = nn.ModuleList() # using linear trick
        for in_dim, out_dim in zip(self.in_feat_channels, self.out_feat_channels):
            self.conv_1x1s.append(
                nn.Sequential(
                    nn.Linear(in_dim, in_dim*2),
                    nn.ReLU(inplace=False),
                    nn.Linear(in_dim*2, out_dim),
                )
            ) # pointwise/1x1 convs, implemented with linear layers    

    def forward(self, xs):
        xs_out = []
        for i, x in enumerate(xs):
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.conv_1x1s[i](x)
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
            xs_out.append(x)
        return xs_out