"""
DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement

    enc_ch      = [16, 32, 64, 128, 256, 256]
    kernel      = (5, 2), stride = (2, 1), causal padding
    ComplexConv = shared weight (1 Conv2d dùng chung)
    LSTM        = ComplexLSTM, hidden=256, 2 layers 
    Skip        = U-Net skip connections
    Masking     = DCCRN-E (polar, tanh) / DCCRN-R / DCCRN-CL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# ComplexBatchNorm2d
# ---------------------------------------------------------------------------

class ComplexBatchNorm2d(nn.Module):
    """BN riêng cho real và imag."""
    def __init__(self, num_features: int):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, r: Tensor, i: Tensor) -> Tuple[Tensor, Tensor]:
        return self.bn_r(r), self.bn_i(i)


# ---------------------------------------------------------------------------
# ComplexConv2d — SHARED weight (Eq. 1)
#
# Dùng 1 bộ weight W chia sẻ:
#   Y_r = W(X_r) - W(X_i)
#   Y_i = W(X_i) + W(X_r)
# → params = 1× Conv thay vì 2×  ← lý do paper đạt ~3.7M
# ---------------------------------------------------------------------------

class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Tuple[int, int] = (5, 2),
        stride      : Tuple[int, int] = (2, 1),
        freq_pad    : int = 2,
        time_pad    : int = 1,
        causal      : bool = True,
    ):
        super().__init__()
        self.time_pad = time_pad
        self.causal   = causal
        self.W = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(freq_pad, 0),  # time padding thủ công
            bias=False,
        )

    def _pad(self, x: Tensor) -> Tensor:
        p = self.time_pad
        return F.pad(x, (p, 0)) if self.causal else F.pad(x, (p // 2, p - p // 2))

    def forward(self, r: Tensor, i: Tensor) -> Tuple[Tensor, Tensor]:
        rp, ip = self._pad(r), self._pad(i)
        return self.W(rp) - self.W(ip),  self.W(ip) + self.W(rp)


# ---------------------------------------------------------------------------
# ComplexConvTranspose2d — SHARED weight
# ---------------------------------------------------------------------------

class ComplexConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        kernel_size   : Tuple[int, int] = (5, 2),
        stride        : Tuple[int, int] = (2, 1),
        padding       : Tuple[int, int] = (2, 0),
        output_padding: Tuple[int, int] = (1, 0),
    ):
        super().__init__()
        self.W = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding,
            bias=False,
        )

    def forward(self, r: Tensor, i: Tensor) -> Tuple[Tensor, Tensor]:
        return self.W(r) - self.W(i),  self.W(i) + self.W(r)


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """ComplexConv2d → ComplexBN → PReLU"""
    def __init__(self, in_ch: int, out_ch: int, causal: bool = True):
        super().__init__()
        self.conv = ComplexConv2d(in_ch, out_ch, causal=causal)
        self.bn   = ComplexBatchNorm2d(out_ch)
        self.pr   = nn.PReLU()
        self.pi   = nn.PReLU()

    def forward(self, r: Tensor, i: Tensor) -> Tuple[Tensor, Tensor]:
        r, i = self.conv(r, i)
        r, i = self.bn(r, i)
        return self.pr(r), self.pi(i)


class DecoderBlock(nn.Module):
    """ComplexConvTranspose2d → ComplexBN → PReLU (bỏ PReLU ở layer cuối)"""
    def __init__(self, in_ch: int, out_ch: int, is_last: bool = False):
        super().__init__()
        self.deconv  = ComplexConvTranspose2d(in_ch, out_ch)
        self.bn      = ComplexBatchNorm2d(out_ch)
        self.is_last = is_last
        if not is_last:
            self.pr = nn.PReLU()
            self.pi = nn.PReLU()

    def forward(self, r: Tensor, i: Tensor) -> Tuple[Tensor, Tensor]:
        r, i = self.deconv(r, i)
        r, i = self.bn(r, i)
        if not self.is_last:
            r, i = self.pr(r), self.pi(i)
        return r, i


# ---------------------------------------------------------------------------
# ComplexLSTM (Eq. 2-4) — 4 LSTM cross-mixing, 2 shared LSTMs
# ---------------------------------------------------------------------------

class ComplexLSTM(nn.Module):
    """
    Complex LSTM theo Eq. 2-4:
        F_rr = LSTM_r(X_r),  F_ir = LSTM_r(X_i)   ← dùng chung lstm_r
        F_ri = LSTM_i(X_r),  F_ii = LSTM_i(X_i)   ← dùng chung lstm_i
        out_r = F_rr - F_ii
        out_i = F_ri + F_ir

    Dùng 2 LSTM (lstm_r, lstm_i) thay vì 4, gọi mỗi LSTM 2 lần riêng biệt
    để tránh hidden state bị nhiễm giữa các samples.

    Input/Output shape: (B, C, F, T)
    """
    def __init__(
        self,
        in_ch     : int,
        in_freq   : int,
        hidden    : int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.in_ch   = in_ch
        self.in_freq = in_freq
        D = in_ch * in_freq

        kw = dict(hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.lstm_r = nn.LSTM(D, **kw)   # xử lý real path
        self.lstm_i = nn.LSTM(D, **kw)   # xử lý imag path

        # Projection về D (không có dense layer trung gian)
        self.proj_r = nn.Linear(hidden, D)
        self.proj_i = nn.Linear(hidden, D)

    def forward(self, r: Tensor, i: Tensor) -> Tuple[Tensor, Tensor]:
        B, C, F, T = r.shape
        rv = r.permute(0, 3, 1, 2).reshape(B, T, C * F)   # (B, T, D)
        iv = i.permute(0, 3, 1, 2).reshape(B, T, C * F)

        # FIX: gọi riêng từng lần để tránh hidden state bị nhiễm
        # lstm_r xử lý rv → Frr, xử lý iv → Fir
        # lstm_i xử lý rv → Fri, xử lý iv → Fii
        Frr, _ = self.lstm_r(rv)   # (B, T, hidden)
        Fir, _ = self.lstm_r(iv)   # dùng chung lstm_r, gọi riêng
        Fri, _ = self.lstm_i(rv)   # dùng chung lstm_i, gọi riêng
        Fii, _ = self.lstm_i(iv)

        # Công thức nhân phức (Eq. 4)
        out_r = Frr - Fii
        out_i = Fri + Fir

        # Project về (B, C, F, T)
        out_r = self.proj_r(out_r).reshape(B, T, C, F).permute(0, 2, 3, 1)
        out_i = self.proj_i(out_i).reshape(B, T, C, F).permute(0, 2, 3, 1)

        return out_r, out_i


# ---------------------------------------------------------------------------
# DCCRN — main model
# ---------------------------------------------------------------------------

class DCCRN(nn.Module):
    """
    Deep Complex Convolution Recurrent Network.

    Modes
    -----
    'DCCRN-E'   Polar mask + tanh  →  DNS real-time winner
    'DCCRN-R'   Independent real/imag mask
    'DCCRN-CL'  Polar mask + ComplexLSTM (hidden=128)

    Parameters
    ----------
    n_fft           : FFT size (paper: 512)
    hop_length      : hop (paper: 100 → 6.25 ms @ 16 kHz)
    win_length      : window (paper: 400 → 25 ms @ 16 kHz)
    encoder_channels: [16, 32, 64, 128, 256, 256]  (6 layers)
    lstm_hidden     : 256 for E/R, 128 for CL
    lstm_layers     : 2
    causal          : True → causal (left-pad only)
    """

    _DEFAULTS = {
        #               enc_ch                    hidden
        'DCCRN-E' : ([16, 32, 64, 128, 256, 256], 256),
        'DCCRN-R' : ([16, 32, 64, 128, 256, 256], 256),
        'DCCRN-CL': ([16, 32, 64, 128, 256, 256], 128),
    }

    def __init__(
        self,
        mode            : str = 'DCCRN-E',
        n_fft           : int = 512,
        hop_length      : int = 100,
        win_length      : int = 400,
        encoder_channels: Optional[List[int]] = None,
        lstm_hidden     : Optional[int] = None,
        lstm_layers     : int = 2,
        causal          : bool = True,
    ):
        super().__init__()
        assert mode in self._DEFAULTS, f"mode must be one of {list(self._DEFAULTS)}"
        self.mode = mode

        enc_ch, hid = self._DEFAULTS[mode]
        enc = encoder_channels if encoder_channels is not None else enc_ch
        hid = lstm_hidden      if lstm_hidden      is not None else hid

        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        # ---- Encoder ----
        in_ch = [1] + enc[:-1]
        self.encoder = nn.ModuleList([
            EncoderBlock(in_ch[k], enc[k], causal=causal)
            for k in range(len(enc))
        ])

        # ---- LSTM (lazy build — cần freq dim từ forward pass đầu tiên) ----
        self._hid        = hid
        self._layers     = lstm_layers
        self._last_ch    = enc[-1]
        self._lstm_ready = False

        # ---- Decoder ----
        dec   = list(reversed(enc))
        d_in  = [dec[k] * 2 for k in range(len(dec))]   # *2 vì skip concat
        d_out = dec[1:] + [1]
        self.decoder = nn.ModuleList([
            DecoderBlock(d_in[k], d_out[k], is_last=(k == len(dec) - 1))
            for k in range(len(dec))
        ])

        # ---- Mask activation ----
        self.mask_act = nn.Tanh() if mode in ('DCCRN-E', 'DCCRN-CL') else nn.Sigmoid()

    # ------------------------------------------------------------------
    def _build_lstm(self, ch: int, freq: int, device: torch.device):
        self.lstm = ComplexLSTM(
            in_ch=ch,
            in_freq=freq,
            hidden=self._hid,
            num_layers=self._layers,
        ).to(device)
        self._lstm_ready = True

    # ------------------------------------------------------------------
    def stft(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """(B, L) → real: (B,1,F,T'), imag: (B,1,F,T') — bỏ DC bin"""
        spec = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            return_complex=True,
        )                           # (B, n_fft//2+1, T')
        spec = spec[:, 1:, :]      # bỏ DC component
        return spec.real.unsqueeze(1), spec.imag.unsqueeze(1)

    def istft(self, real: Tensor, imag: Tensor, length: int) -> Tensor:
        """real/imag: (B,1,F,T') → (B, L) — thêm lại DC bin = 0"""
        B, _, F, T = real.shape
        z    = torch.zeros(B, 1, 1, T, device=real.device)
        real = torch.cat([z, real], dim=2).squeeze(1)
        imag = torch.cat([z, imag], dim=2).squeeze(1)
        return torch.istft(
            torch.complex(real, imag),
            n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            length=length,
        )

    # ------------------------------------------------------------------
    def _apply_mask(
        self, mr: Tensor, mi: Tensor,
        yr: Tensor, yi: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        mr = self.mask_act(mr)
        mi = self.mask_act(mi)

        if self.mode in ('DCCRN-E', 'DCCRN-CL'):
            # Polar coordinate (Eq. 9)
            m_mag = (mr**2 + mi**2 + 1e-8).sqrt()
            m_pha = torch.atan2(mi, mr)
            y_mag = (yr**2 + yi**2 + 1e-8).sqrt()
            y_pha = torch.atan2(yi, yr)
            e = y_mag * m_mag
            p = y_pha + m_pha
            return e * p.cos(), e * p.sin()

        else:  # DCCRN-R (Eq. 7)
            return yr * mr, yi * mi

    # ------------------------------------------------------------------
    def forward(
        self, noisy: Tensor
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]], List[Tuple[Tensor, Tensor]]]:
        """
        Parameters
        ----------
        noisy : (B, L)  noisy waveform

        Returns
        -------
        enhanced  : (B, L)
        enc_feats : list[(real, imag)]  per encoder layer
        dec_feats : list[(real, imag)]  per decoder layer
        """
        L = noisy.shape[-1]
        inp_r, inp_i = self.stft(noisy)    # (B, 1, F, T')

        r, i = inp_r, inp_i
        skips    : List[Tuple[Tensor, Tensor]] = []
        enc_feats: List[Tuple[Tensor, Tensor]] = []

        # ---- Encoder ----
        for layer in self.encoder:
            r, i = layer(r, i)
            skips.append((r, i))
            enc_feats.append((r, i))

        # ---- Build LSTM (lazy) ----
        if not self._lstm_ready:
            self._build_lstm(r.shape[1], r.shape[2], r.device)

        # ---- LSTM bottleneck ----
        r, i = self.lstm(r, i)

        # ---- Decoder ----
        dec_feats: List[Tuple[Tensor, Tensor]] = []

        for idx, layer in enumerate(self.decoder):
            sr, si = skips[-(idx + 1)]

            # Căn chỉnh spatial dims (off-by-one từ transposed conv)
            if r.shape[2] != sr.shape[2]:
                r = r[..., :sr.shape[2], :]
                i = i[..., :sr.shape[2], :]
            if r.shape[3] != sr.shape[3]:
                r = r[..., :sr.shape[3]]
                i = i[..., :sr.shape[3]]

            r = torch.cat([r, sr], dim=1)
            i = torch.cat([i, si], dim=1)
            r, i = layer(r, i)
            dec_feats.append((r, i))

        # Trim về kích thước spectrum đầu vào
        if r.shape[2] != inp_r.shape[2]:
            r = r[..., :inp_r.shape[2], :]
            i = i[..., :inp_r.shape[2], :]
        if r.shape[3] != inp_r.shape[3]:
            r = r[..., :inp_r.shape[3]]
            i = i[..., :inp_r.shape[3]]

        enh_r, enh_i = self._apply_mask(r, i, inp_r, inp_i)
        enhanced = self.istft(enh_r, enh_i, L)

        return enhanced, enc_feats, dec_feats


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_breakdown(model: DCCRN) -> None:
    enc_p  = sum(count_parameters(l) for l in model.encoder)
    dec_p  = sum(count_parameters(l) for l in model.decoder)
    lstm_p = count_parameters(model.lstm) if model._lstm_ready else 0
    total  = count_parameters(model)
    print(f"  {'Encoder':<10}: {enc_p/1e6:.3f} M")
    print(f"  {'LSTM':<10}: {lstm_p/1e6:.3f} M")
    print(f"  {'Decoder':<10}: {dec_p/1e6:.3f} M")
    print(f"  {'Total':<10}: {total/1e6:.3f} M ")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy  = torch.randn(2, 16_000 * 4).to(device)

    for mode in ('DCCRN-E', 'DCCRN-R', 'DCCRN-CL'):
        model = DCCRN(mode=mode).to(device)
        with torch.no_grad():
            enh, ef, df = model(dummy)

        print(f"\n{'='*45}")
        print(f"Mode: {mode}")
        param_breakdown(model)
        t0 = time.time()
        with torch.no_grad():
            enh, ef, df = model(dummy)
        print(f"  {'Output':<10}: {tuple(enh.shape)}")
        print(f"  {'Time':<10}: {(time.time()-t0)*1000:.1f} ms")