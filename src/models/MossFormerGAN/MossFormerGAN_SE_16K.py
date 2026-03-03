import torch
from torch import nn
from .generator import MossFormerGenerator
from .discriminator import MultiScaleDiscriminator

class MossFormerGAN_SE_16K(nn.Module):
    """
    Main entry point for MossFormerGAN_SE_16K architectures.
    Combines the Generator (MossFormer) and the MultiScaleDiscriminator
    for End-to-End time-domain speech enhancement.
    """
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 hidden_channels=256, 
                 num_blocks=4):
        super(MossFormerGAN_SE_16K, self).__init__()
        
        # Generator for Speech Enhancement
        self.generator = MossFormerGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks
        )
        
        # Discriminator for GAN objectives (Feature Matching, Real/Fake)
        self.discriminator = MultiScaleDiscriminator()
        
    def forward(self, x):
        """
        By default, the forward method acts as the Generator mapping
        from noisy audio (x) to enhanced audio.
        During training, you can extract `self.generator` and `self.discriminator` independently.
        """
        # x expected shape: [Batch, Time] or [Batch, 1, Time]
        return self.generator(x)

if __name__ == "__main__":
    # Test Network construction
    model = MossFormerGAN_SE_16K(in_channels=1, out_channels=1, hidden_channels=256, num_blocks=4)
    print("MossFormerGAN Generator Parameters:", sum(p.numel() for p in model.generator.parameters() if p.requires_grad))
    print("MossFormerGAN Discriminator Parameters:", sum(p.numel() for p in model.discriminator.parameters() if p.requires_grad))
    
    # Dummy forward pass
    x = torch.randn(4, 16000) # [Batch, Time]
    y_hat = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_hat.shape}")
    
    # Test Discriminator
    scores_real, scores_fake, fmaps_real, fmaps_fake = model.discriminator(x, y_hat)
    print(f"Discriminator generated {len(scores_real)} multi-scale scores.")
