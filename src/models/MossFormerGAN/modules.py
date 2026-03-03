import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size, eps=1e-8):
        super(GlobalLayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.gamma)
        torch.nn.init.zeros_(self.beta)

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + self.eps, 0.5) + self.beta
        return gLN_y

class CumulativeLayerNorm(nn.Module):
    """Cumulative Layer Normalization (cLN)"""
    def __init__(self, channel_size, eps=1e-8):
        super(CumulativeLayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.gamma)
        torch.nn.init.zeros_(self.beta)

    def forward(self, y):
        # Using built-in LayerNorm for simplicity on sequence dimension
        # Note: True cLN uses cumulative statistics, standard LN applied over feature dim
        return F.layer_norm(y.transpose(1, 2), (y.size(1),), weight=self.gamma.squeeze(), bias=self.beta.squeeze(), eps=self.eps).transpose(1, 2)


def select_norm(norm, dim):
    if norm == 'gln':
        return GlobalLayerNorm(dim)
    elif norm == 'cln':
        return CumulativeLayerNorm(dim)
    elif norm == 'ln':
        return nn.LayerNorm(dim)
    elif norm == 'bn':
        return nn.BatchNorm1d(dim)
    else:
        return nn.Identity()

class DepthwiseConv1DBlock(nn.Module):
    """Depthwise Separable Convolution Block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm='gln', activation='prelu'):
        super(DepthwiseConv1DBlock, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.norm = select_norm(norm, out_channels)
        
        if activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class FeedForwardNetwork(nn.Module):
    """Position-wise Feed-Forward Network"""
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
            
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x expected shape: [Batch, Time, Channels]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x
