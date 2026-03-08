"""
src/loss/mossformergan_loss.py
Loss functions for MossFormerGAN (GAN-based speech enhancement model).

Losses:
  - si_snr_loss          : Scale-Invariant SNR (shared with WaveUnet)
  - discriminator_loss   : LSGAN discriminator loss
  - generator_loss       : LSGAN generator adversarial loss
  - mossformergan_g_total: Combined generator loss used in training loop
"""

import torch


def si_snr_loss(estimated: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Noise Ratio Loss.

    Args:
        estimated : [B, T] enhanced waveform
        target    : [B, T] clean reference waveform
    Returns:
        Scalar loss (negative SI-SNR, minimise to maximise SI-SNR)
    """
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target    = target    - target.mean(dim=-1, keepdim=True)
    dot          = (estimated * target).sum(dim=-1, keepdim=True)
    target_power = (target ** 2).sum(dim=-1, keepdim=True) + eps
    s_target     = dot * target / target_power
    e_noise      = estimated - s_target
    si_snr = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps
    )
    return -si_snr.mean()


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    LSGAN Discriminator Loss.
    Teaches the discriminator to output 1 for real audio, 0 for fake.

    Args:
        disc_real_outputs      : list of discriminator outputs on real audio
        disc_generated_outputs : list of discriminator outputs on generated audio
    Returns:
        loss     : total discriminator loss (scalar)
        r_losses : per-discriminator real losses (list)
        g_losses : per-discriminator fake losses (list)
    """
    loss = 0
    r_losses, g_losses = [], []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss  += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """
    LSGAN Generator Adversarial Loss.
    Teaches the generator to fool the discriminator (output close to 1 for fakes).

    Args:
        disc_outputs : list of discriminator outputs on generated audio
    Returns:
        loss       : total generator adversarial loss (scalar)
        gen_losses : per-discriminator losses (list)
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses


def mossformergan_g_total(
    enhanced_audio  : torch.Tensor,
    clean_audio     : torch.Tensor,
    disc_outputs,
    l1_loss_fn,
    l1_weight       : float,
    si_snr_weight   : float,
):
    """
    Combined Generator loss for MossFormerGAN:
        loss_g = Adversarial + l1_weight * L1 + si_snr_weight * SI-SNR

    Args:
        enhanced_audio : [B, T] generator output
        clean_audio    : [B, T] clean reference
        disc_outputs   : discriminator outputs on enhanced audio (for adv loss)
        l1_loss_fn     : nn.L1Loss instance
        l1_weight      : weight for L1 reconstruction loss
        si_snr_weight  : weight for SI-SNR loss
    Returns:
        loss_g_total : combined generator loss
        loss_g_fake  : adversarial component
        loss_l1      : L1 reconstruction component
        loss_si_snr  : SI-SNR component
    """
    loss_g_fake, _ = generator_loss(disc_outputs)
    loss_l1        = l1_loss_fn(enhanced_audio, clean_audio)
    loss_si_snr    = si_snr_loss(enhanced_audio, clean_audio)
    loss_g_total   = (loss_g_fake
                      + l1_weight     * loss_l1
                      + si_snr_weight * loss_si_snr)
    return loss_g_total, loss_g_fake, loss_l1, loss_si_snr
