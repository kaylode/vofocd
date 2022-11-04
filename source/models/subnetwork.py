import torch
import torch.nn as nn

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=1, padding=0,
        bias=False, stride=stride)

class Subnetwork(nn.Module):
    """
    https://github.com/amzn/xfer/blob/master/var_info_distil/wide_residual_network.py
    """
    def __init__(self, in_feat_channels, out_feat_channels) -> None:
        super().__init__()
        self.in_feat_channels = in_feat_channels
        self.out_feat_channels = out_feat_channels


        self.conv_1x1s = nn.ModuleList() # using linear trick
        for in_dim, out_dim in zip(self.in_feat_channels, self.out_feat_channels):
            self.conv_1x1s.append(
                nn.Sequential(
                    conv1x1(in_dim, in_dim*2),
                    nn.ReLU(),
                    conv1x1(in_dim*2, in_dim*2),
                    nn.ReLU(),
                    conv1x1(in_dim*2, out_dim),
                )
            ) # pointwise/1x1 convs, implemented with linear layers    

        # variance are represented as a softplus function applied to "variance parameters".
        init_variance_param_value = self._variance_param_to_variance(torch.tensor(5.0))
        self.variance_params = []
        for out_feat_chnl in self.out_feat_channels:
            variance_param = nn.Parameter(
                torch.full((out_feat_chnl, 1, 1), init_variance_param_value)
            )
            self.variance_params.append(variance_param)
        self.variance_params = nn.ParameterList(self.variance_params)

    def _variance_to_variance_param(self, variance):
        """
        Convert variance to corresponding variance parameter by inverse of the softplus function.
        :param torch.FloatTensor variance: the target variance for obtaining the variance parameter
        """
        return torch.log(torch.exp(variance) - 1.0)

    def _variance_param_to_variance(self, variance_param):
        """
        Convert the variance parameter to corresponding variance by the softplus function.
        :param torch.FloatTensor variance_param: the target variance parameter for obtaining the variance
        """
        return torch.log(torch.exp(variance_param) + 1.0)

    def forward(self, xs):
        xs_out = []
        for i, x in enumerate(xs):
            x = self.conv_1x1s[i](x)
            xs_out.append(x)

        variance_outs = []
        for chnnl_idx in range(len(self.out_feat_channels)): 
            variance = self._variance_param_to_variance(self.variance_params[chnnl_idx])
            # Input has an additional dimension for mini-batch, resize the variance to match its dimension
            variance = variance.unsqueeze(0)
            variance_outs.append(variance)

        return xs_out, variance_outs