import torch
import torch.nn as nn
import torch.nn.functional as F

class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, ipt):
        return self.main(ipt)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)

class WaveUnet(nn.Module):
    """
    Standard implementation of Wave-Unet for Speech Enhancement
    Input shape: (Batch, Channels=1, Time) or (Batch, Time)
    Output shape: (Batch, Channels=1, Time)
    """
    def __init__(self, in_channels=1, n_layers=10, channels_interval=20):
        super(WaveUnet, self).__init__()
        self.n_layers = n_layers
        self.channels_interval = channels_interval
        
        encoder_in_channels_list = [in_channels] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]

        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(in_channels + self.channels_interval, in_channels, kernel_size=1, stride=1),
            nn.Tanh()  # Output values bounded between [-1, 1], useful for waveform
        )

    def forward(self, x):
        # x shape should be: [Batch, Channels, Time Length]
        # If input is [Batch, Time Length], unsqueeze to add channel dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        original_x = x
        skips = []
        for i in range(self.n_layers):
            x = self.encoder[i](x)
            skips.append(x)
            # Decimate by factor of 2 (Drop every other sample)
            x = x[:, :, ::2]

        x = self.middle(x)

        for i in range(self.n_layers):
            # Upsample using linear interpolation
            x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=True)
            
            # Crop or pad to match the skip connection size if there are length mismatches due to decimation/interpolation
            skip_length = skips[-i - 1].shape[2]
            diff = skip_length - x.shape[2]
            if diff > 0:
                x = F.pad(x, (0, diff))
            elif diff < 0:
                x = x[:, :, :skip_length]
            
            # Concatenate skip connection
            x = torch.cat([x, skips[-i - 1]], dim=1)
            x = self.decoder[i](x)

        # Final concatenation with original input
        x = torch.cat([x, original_x], dim=1)
        x = self.out(x)
        
        # If input was [Batch, Time Length], return squeezed 
        if len(original_x.shape) == 2:
            x = x.squeeze(1)
            
        return x

if __name__ == '__main__':
    model = WaveUnet(in_channels=1, n_layers=10, channels_interval=20)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    
    # Dummy input representing batch of 4 audio clips, 1 channel, 16000 samples (~1 sec audio)
    dummy_input = torch.randn(4, 1, 16000)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
