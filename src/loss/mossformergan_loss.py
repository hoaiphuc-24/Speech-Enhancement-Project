"""
src/loss/mossformergan_loss.py
Loss functions for MossFormerGAN (spectral-domain, STFT-based GAN speech enhancement).

Main entry point:
  - loss_mossformergan_se_16k : Combined generator + discriminator loss in spectral domain

Helpers:
  - stft / istft             : Short-Time Fourier Transform wrappers
  - power_compress / uncompress : Power-law compression on spectrograms
  - batch_pesq               : Batch PESQ scoring for discriminator training target
"""

import torch
import torch.nn.functional as F
import numpy as np

try:
    from pesq import pesq
except ImportError:
    pesq = None
    print("[mossformergan_loss] WARNING: pesq package not found. PESQ scoring disabled.")


# ---------------------------------------------------------------------------
# STFT / iSTFT helpers
# ---------------------------------------------------------------------------

def stft(x, args, center=True):
    """
    Compute STFT of a batch of waveforms.

    Args:
        x    : [B, T] waveform tensor
        args : namespace with attributes n_fft, hop_length, win_length
        center : whether to pad signal at center (matches librosa convention)
    Returns:
        spec : [B, 2, F, T] real/imag stacked spectrogram
    """
    n_fft     = getattr(args, 'n_fft',      512)
    hop_length = getattr(args, 'hop_length', 100)
    win_length = getattr(args, 'win_length', 400)

    window = torch.hann_window(win_length).to(x.device)

    if center:
        x = F.pad(x, (n_fft // 2, n_fft // 2))

    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )  # [B, F, T]

    real = spec.real.unsqueeze(1)  # [B, 1, F, T]
    imag = spec.imag.unsqueeze(1)  # [B, 1, F, T]
    return torch.cat([real, imag], dim=1)  # [B, 2, F, T]


def istft(spec, args):
    """
    Compute iSTFT from [B, F, T] complex tensor.

    Args:
        spec : [B, F, T] complex tensor
        args : namespace with attributes n_fft, hop_length, win_length
    Returns:
        x    : [B, T] waveform tensor
    """
    n_fft      = getattr(args, 'n_fft',      512)
    hop_length = getattr(args, 'hop_length', 100)
    win_length = getattr(args, 'win_length', 400)

    window = torch.hann_window(win_length).to(spec.device)
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )


# ---------------------------------------------------------------------------
# Power compression
# ---------------------------------------------------------------------------

def power_compress(spec, c=0.3):
    """
    Power-law compression on magnitude of [B, 2, F, T] spectrogram.
    Compresses loud components to improve perceptual quality of loss.
    """
    real = spec[:, 0, :, :]
    imag = spec[:, 1, :, :]
    mag  = torch.clamp(torch.sqrt(real**2 + imag**2 + 1e-9), min=1e-9)
    mag_compressed = mag ** c
    phase = torch.atan2(imag, real)
    real_c = mag_compressed * torch.cos(phase)
    imag_c = mag_compressed * torch.sin(phase)
    return torch.stack([real_c, imag_c], dim=1)  # [B, 2, F, T]


def power_uncompress(real, imag, c=0.3):
    """
    Invert power compression to reconstruct complex spectrogram.

    Args:
        real : [B, 1, F, T]
        imag : [B, 1, F, T]
    Returns:
        spec : [B, F, T] complex tensor
    """
    mag_c = torch.clamp(torch.sqrt(real**2 + imag**2 + 1e-9), min=1e-9)
    mag   = mag_c ** (1.0 / c)
    phase = torch.atan2(imag, real)
    real_u = mag * torch.cos(phase)
    imag_u = mag * torch.sin(phase)
    # squeeze channel dim → [B, F, T] complex
    return torch.complex(real_u.squeeze(1), imag_u.squeeze(1))


# ---------------------------------------------------------------------------
# Batch PESQ
# ---------------------------------------------------------------------------

def batch_pesq(clean_list, enhanced_list, sr=16000):
    """
    Compute mean PESQ score for a batch.

    Args:
        clean_list    : list of np.ndarray clean waveforms
        enhanced_list : list of np.ndarray enhanced waveforms
        sr            : sample rate (default 16000)
    Returns:
        pesq_tensor : [B] float32 tensor of PESQ scores, or None on failure
    """
    if pesq is None:
        return None

    scores = []
    for ref, deg in zip(clean_list, enhanced_list):
        try:
            score = pesq(sr, ref, deg, 'wb')
            # Normalize to [0, 1]: PESQ WB range is [-0.5, 4.5]
            scores.append((score + 0.5) / 5.0)
        except Exception:
            scores.append(None)

    if any(s is None for s in scores):
        return None

    return torch.FloatTensor(scores)


# ---------------------------------------------------------------------------
# Main loss function
# ---------------------------------------------------------------------------

def loss_mossformergan_se_16k(args, inputs, labels, out_list, c, discriminator, device):
    """
    Combined generator + discriminator loss for MossFormerGAN_SE_16K.
    Operates in the spectral domain using STFT magnitude and RI components.

    Args:
        args          : namespace with n_fft, hop_length, win_length, batch_size
        inputs        : noisy waveform (unused directly, kept for API consistency)
        labels        : [B, T] clean reference waveforms
        out_list      : [pred_real, pred_imag] each [B, 1, T, F] from generator
        c             : amplitude scaling tensor [B] or scalar
        discriminator : metric discriminator (labels_mag, pred_mag) → scalar score
        device        : torch device
    Returns:
        (loss, discrim_loss_metric) or (None, None) on bad batch
    """
    one_labels = torch.ones(args.batch_size).to(device)

    # Scale labels
    labels = torch.transpose(labels, 0, 1)
    labels = torch.transpose(labels * c, 0, 1)

    # Generator output: permute [B,1,T,F] → [B,1,F,T]
    pred_real = out_list[0].permute(0, 1, 3, 2)
    pred_imag = out_list[1].permute(0, 1, 3, 2)
    pred_mag  = torch.sqrt(pred_real**2 + pred_imag**2)

    # Labels STFT → power compress
    labels_spec = stft(labels, args, center=True).to(torch.float32)
    labels_spec = power_compress(labels_spec)

    labels_real = labels_spec[:, 0, :, :].unsqueeze(1)
    labels_imag = labels_spec[:, 1, :, :].unsqueeze(1)
    labels_mag  = torch.sqrt(labels_real**2 + labels_imag**2)

    # Generator adversarial loss (fool discriminator → predict 1)
    predict_fake_metric = discriminator(labels_mag, pred_mag)
    gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())

    # Spectral reconstruction losses
    loss_mag = F.mse_loss(pred_mag,  labels_mag)
    loss_ri  = F.mse_loss(pred_real, labels_real) + F.mse_loss(pred_imag, labels_imag)

    # Time-domain loss via iSTFT
    pred_spec_uncompress = power_uncompress(pred_real, pred_imag).squeeze(0) \
        if pred_real.dim() == 4 else power_uncompress(pred_real, pred_imag)
    # Reconstruct waveform
    pred_spec_uncompress_full = power_uncompress(pred_real, pred_imag)  # [B, F, T] complex
    pred_audio = istft(pred_spec_uncompress_full, args)

    length     = min(pred_audio.size(-1), labels.size(-1))
    pred_audio = pred_audio[..., :length]
    labels     = labels[..., :length]
    time_loss  = torch.mean(torch.abs(pred_audio - labels))

    # Total generator loss
    loss = 0.1 * loss_ri + 0.9 * loss_mag + 0.2 * time_loss + 0.05 * gen_loss_GAN

    if torch.isnan(loss):
        print('train loss is nan, skip this batch!')
        return None, None

    # PESQ scoring for discriminator target
    pred_audio_list  = list(pred_audio.detach().cpu().numpy())
    labels_audio_list = list(labels.cpu().numpy())
    pesq_score = batch_pesq(labels_audio_list, pred_audio_list)

    if pesq_score is not None:
        pesq_score = pesq_score.to(device)
        predict_enhance_metric = discriminator(labels_mag, pred_mag.detach())
        predict_max_metric     = discriminator(labels_mag, labels_mag)
        discrim_loss_metric = (
            F.mse_loss(predict_max_metric.flatten(), one_labels) +
            F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
        )
    else:
        print('train pesq score is None, skip this batch!')
        return None, None

    return loss, discrim_loss_metric
