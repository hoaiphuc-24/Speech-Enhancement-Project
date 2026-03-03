import torch
import torch.nn as nn
from .mossformer import MossFormerBlock

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        # x is [B, C, T]
        x = self.conv(x)
        # norm expects [B, T, C]
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Decoder, self).__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            output_padding=stride - 1
        )

    def forward(self, x):
        # x is [B, C, T]
        return self.deconv(x)

class MossFormerGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=256, num_blocks=4):
        super(MossFormerGenerator, self).__init__()
        
        # Audio Encoder
        # Reduces sample rate by factor of 8 for internal processing processing
        self.encoder = Encoder(in_channels, hidden_channels, kernel_size=16, stride=8)
        
        # MossFormer Processing Blocks
        self.bottleneck = nn.Sequential(*[
            MossFormerBlock(d_model=hidden_channels, num_heads=8, ffn_expansion=4, conv_kernel_size=15)
            for _ in range(num_blocks)
        ])
        
        # Audio Decoder
        self.decoder = Decoder(hidden_channels, out_channels, kernel_size=16, stride=8)

    def forward(self, x):
        # Allow 2D inputs [Batch, Time]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        original_length = x.shape[-1]
        
        # Encode
        encoded = self.encoder(x)
        
        # MossFormer blocks expect shape [B, T, C]
        bottleneck_in = encoded.transpose(1, 2)
        
        # Process 
        bottleneck_out = self.bottleneck(bottleneck_in)
        
        # Convert back to [B, C, T] for decoder
        bottleneck_out = bottleneck_out.transpose(1, 2)
        
        # Masking / Decoding
        masked_encoded = encoded * torch.sigmoid(bottleneck_out)  # Masking approach
        enhanced = self.decoder(masked_encoded)
        
        # Trim padding if necessary due to stide/kernel size
        if enhanced.shape[-1] != original_length:
            diff = enhanced.shape[-1] - original_length
            if diff > 0:
                enhanced = enhanced[:, :, :-diff]
            elif diff < 0:
                enhanced = torch.nn.functional.pad(enhanced, (0, -diff))
                
        return enhanced.squeeze(1) if enhanced.shape[1] == 1 else enhanced
