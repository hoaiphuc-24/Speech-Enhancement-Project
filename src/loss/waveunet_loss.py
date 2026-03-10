"""
src/loss/waveunet_loss.py
Loss functions for WaveUnet (time-domain regression model).

Losses:
  - si_snr_loss    : Scale-Invariant SNR (primary metric for speech enhancement)
  - ms_stft_loss   : Multi-Scale STFT (spectral convergence + log magnitude)
  - waveunet_total : Combined loss used in training loop
"""

import torch
import torch.nn.functional as F


def si_snr_loss(estimated: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Noise Ratio Loss.
    Standard metric for speech enhancement — directly optimises perceptual quality.

    Args:
        estimated : [B, T] enhanced waveform
        target    : [B, T] clean reference waveform
        eps       : numerical stability epsilon
    Returns:
        Scalar loss (negative SI-SNR, minimise to maximise SI-SNR)
    """
    # Remove DC offset
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target    = target    - target.mean(dim=-1, keepdim=True)
    # Project estimated onto target
    dot          = (estimated * target).sum(dim=-1, keepdim=True)
    target_power = (target ** 2).sum(dim=-1, keepdim=True) + eps
    s_target     = dot * target / target_power   # scaled clean signal
    e_noise      = estimated - s_target          # noise residual
    si_snr = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps
    )
    return -si_snr.mean()   # minimise negative SI-SNR


def ms_stft_loss(
    estimated : torch.Tensor,
    target    : torch.Tensor,
    fft_sizes : tuple = (512, 1024, 2048),
    hop_sizes : tuple = (128, 256, 512),
    win_sizes : tuple = (512, 1024, 2048),
) -> torch.Tensor:
    """
    Multi-Scale STFT Loss.
    Evaluates reconstruction quality across multiple frequency resolutions.
    Combines spectral convergence loss and log-magnitude L1 loss.

    Args:
        estimated : [B, T] enhanced waveform
        target    : [B, T] clean reference waveform
        fft_sizes : tuple of FFT sizes for each STFT scale
        hop_sizes : tuple of hop lengths for each STFT scale
        win_sizes : tuple of window lengths for each STFT scale
    Returns:
        Scalar averaged loss across all scales
    """
    loss = 0.0
    x = estimated.reshape(-1, estimated.shape[-1])  # flatten batch
    y = target.reshape(-1, target.shape[-1])

    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        window = torch.hann_window(win_size, device=estimated.device)
        S_est = torch.stft(x, n_fft=fft_size, hop_length=hop_size,
                           win_length=win_size, window=window, return_complex=True)
        S_tgt = torch.stft(y, n_fft=fft_size, hop_length=hop_size,
                           win_length=win_size, window=window, return_complex=True)
        mag_est = S_est.abs() + 1e-8
        mag_tgt = S_tgt.abs() + 1e-8
        # Spectral convergence
        sc_loss  = torch.norm(mag_tgt - mag_est, 'fro') / (torch.norm(mag_tgt, 'fro') + 1e-8)
        # Log-magnitude L1
        log_loss = F.l1_loss(torch.log(mag_est), torch.log(mag_tgt))
        loss += sc_loss + log_loss

    return loss / len(fft_sizes)


def waveunet_total(
    estimated    : torch.Tensor,
    target       : torch.Tensor,
    si_snr_weight: float,
    stft_weight  : float,
):
    """
    Combined training loss for WaveUnet (SI-SNR + Multi-Scale STFT).

    Args:
        estimated    : [B, T] enhanced waveform
        target       : [B, T] clean reference waveform
        si_snr_weight: weight for SI-SNR loss
        stft_weight  : weight for Multi-Scale STFT loss
    Returns:
        loss_total  : weighted sum
        loss_si_snr : SI-SNR loss component
        loss_stft   : Multi-Scale STFT loss component
    """
    loss_si_snr = si_snr_loss(estimated, target)
    loss_stft   = ms_stft_loss(estimated, target)
    loss_total  = si_snr_weight * loss_si_snr + stft_weight * loss_stft
    return loss_total, loss_si_snr, loss_stft
