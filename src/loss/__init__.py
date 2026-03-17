"""
src/loss/__init__.py
Exposes loss modules for each model.
"""

from .waveunet_loss import (
    si_snr_loss    as waveunet_si_snr_loss,
    ms_stft_loss,
    waveunet_total,
)

