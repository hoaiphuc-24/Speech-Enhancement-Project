import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleDiscriminator(nn.Module):
    """Single-Scale Time-Domain Discriminator for Audio"""
    def __init__(self, use_spectral_norm=False):
        super(ScaleDiscriminator, self).__init__()
        # PyTorch Spectral Norm helps stabilize GAN training
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator typical for Audio GANs (e.g. MelGAN, HiFi-GAN).
    Reduced from 3 → 2 scales (~33% fewer discriminator params) while keeping
    full channel width in each ScaleDiscriminator for discrimination quality.
    """
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),  # Base scale — Spectral Norm
            ScaleDiscriminator(use_spectral_norm=False), # 2x downsampled — Weight Norm
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y, y_hat):
        # Allow [Batch, Time] inputs to be converted to [Batch, Channels, Time]
        if len(y.shape) == 2:
            y = y.unsqueeze(1)
        if len(y_hat.shape) == 2:
            y_hat = y_hat.unsqueeze(1)
            
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
