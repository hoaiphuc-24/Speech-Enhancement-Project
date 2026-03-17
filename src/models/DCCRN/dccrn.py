
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# ComplexBatchNorm2d
# ---------------------------------------------------------------------------
class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, r: Tensor, i: Tensor) -> Tuple[Tensor, Tensor]:
        return self.bn_r(r), self.bn_i(i)


# ---------------------------------------------------------------------------
# ComplexConv2d — Shared weight
#   Y_r = W(X_r) - W(X_i)
#   Y_i = W(X_i) + W(X_r)
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
            kernel_size=kernel_size, stride=stride,
            padding=(freq_pad, 0), bias=False,
        )

    def _pad(self, x: Tensor) -> Tensor:
        p = self.time_pad
        return F.pad(x, (p, 0)) if self.causal else F.pad(x, (p // 2, p - p // 2))

    def forward(self, r: Tensor, i: Tensor) -> Tuple[Tensor, Tensor]:
        rp, ip = self._pad(r), self._pad(i)
        return self.W(rp) - self.W(ip),  self.W(ip) + self.W(rp)


# ---------------------------------------------------------------------------
# ComplexConvTranspose2d — Shared weight
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
            padding=padding, output_padding=output_padding, bias=False,
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
# ComplexLSTM — 2 LSTMs, gọi riêng 4 lần (tránh hidden state nhiễm)
#
# Eq. 2-4:
#   F_rr = lstm_r(X_r),  F_ir = lstm_r(X_i)
#   F_ri = lstm_i(X_r),  F_ii = lstm_i(X_i)
#   out_r = F_rr - F_ii
#   out_i = F_ri + F_ir
# ---------------------------------------------------------------------------
class ComplexLSTM(nn.Module):
    def __init__(
        self,
        in_ch     : int,
        in_freq   : int,
        hidden    : int = 128,   # giảm 50%: 256 → 128
        num_layers: int = 2,
    ):
        super().__init__()
        D = in_ch * in_freq

        kw = dict(hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.lstm_r = nn.LSTM(D, **kw)
        self.lstm_i = nn.LSTM(D, **kw)

        self.proj_r = nn.Linear(hidden, D)
        self.proj_i = nn.Linear(hidden, D)

    def forward(self, r: Tensor, i: Tensor) -> Tuple[Tensor, Tensor]:
        B, C, F, T = r.shape
        rv = r.permute(0, 3, 1, 2).reshape(B, T, C * F)
        iv = i.permute(0, 3, 1, 2).reshape(B, T, C * F)

        # Gọi riêng từng lần — tránh hidden state của rv lan sang iv
        Frr, _ = self.lstm_r(rv)
        Fir, _ = self.lstm_r(iv)
        Fri, _ = self.lstm_i(rv)
        Fii, _ = self.lstm_i(iv)

        out_r = Frr - Fii   # Eq. 4
        out_i = Fri + Fir

        out_r = self.proj_r(out_r).reshape(B, T, C, F).permute(0, 2, 3, 1)
        out_i = self.proj_i(out_i).reshape(B, T, C, F).permute(0, 2, 3, 1)
        return out_r, out_i


# ---------------------------------------------------------------------------
# DCCRN Lightweight
# ---------------------------------------------------------------------------
class DCCRN(nn.Module):
    """
    DCCRN Lightweight (~1.9M params).

    Parameters
    ----------
    mode            : 'DCCRN-E' hoặc 'DCCRN-CL'
    n_fft           : FFT size (default 512)
    hop_length      : hop size (default 100 → 6.25 ms @ 16 kHz)
    win_length      : window   (default 400 → 25 ms  @ 16 kHz)
    encoder_channels: override channel list
    lstm_hidden     : override LSTM hidden size
    lstm_layers     : số lớp LSTM (default 2)
    causal          : True → real-time causal
    """

    _DEFAULTS = {
        #              enc_ch               hidden
        'DCCRN-E' : ([8, 16, 32, 64, 128, 128], 128),
        'DCCRN-CL': ([8, 16, 32, 64, 128, 128], 128),
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

        # ---- LSTM (lazy build) ----
        self._hid        = hid
        self._layers     = lstm_layers
        self._lstm_ready = False

        # ---- Decoder ----
        dec   = list(reversed(enc))
        d_in  = [dec[k] * 2 for k in range(len(dec))]   # *2 vì skip concat
        d_out = dec[1:] + [1]
        self.decoder = nn.ModuleList([
            DecoderBlock(d_in[k], d_out[k], is_last=(k == len(dec) - 1))
            for k in range(len(dec))
        ])

        # Cả DCCRN-E và DCCRN-CL đều dùng Tanh (polar mask)
        self.mask_act = nn.Tanh()

    def _build_lstm(self, ch: int, freq: int, device: torch.device):
        self.lstm = ComplexLSTM(
            in_ch=ch, in_freq=freq,
            hidden=self._hid,
            num_layers=self._layers,
        ).to(device)
        self._lstm_ready = True

    def stft(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """(B, L) → real: (B,1,F,T'), imag: (B,1,F,T') — bỏ DC bin"""
        spec = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            return_complex=True,
        )
        spec = spec[:, 1:, :]
        return spec.real.unsqueeze(1), spec.imag.unsqueeze(1)

    def istft(self, real: Tensor, imag: Tensor, length: int) -> Tensor:
        """real/imag: (B,1,F,T') → (B, L) — thêm lại DC=0"""
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

    def _apply_mask(
        self, mr: Tensor, mi: Tensor,
        yr: Tensor, yi: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Polar coordinate mask (Eq. 9) — dùng chung cho E và CL"""
        mr = self.mask_act(mr)
        mi = self.mask_act(mi)
        m_mag = (mr**2 + mi**2 + 1e-8).sqrt()
        m_pha = torch.atan2(mi, mr)
        y_mag = (yr**2 + yi**2 + 1e-8).sqrt()
        y_pha = torch.atan2(yi, yr)
        e = y_mag * m_mag
        p = y_pha + m_pha
        return e * p.cos(), e * p.sin()

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
        enc_feats : list[(real, imag)] per encoder layer
        dec_feats : list[(real, imag)] per decoder layer
        """
        L = noisy.shape[-1]
        inp_r, inp_i = self.stft(noisy)

        r, i = inp_r, inp_i
        skips    : List[Tuple[Tensor, Tensor]] = []
        enc_feats: List[Tuple[Tensor, Tensor]] = []

        # ---- Encoder ----
        for layer in self.encoder:
            r, i = layer(r, i)
            skips.append((r, i))
            enc_feats.append((r, i))

        # ---- LSTM bottleneck ----
        if not self._lstm_ready:
            self._build_lstm(r.shape[1], r.shape[2], r.device)
        r, i = self.lstm(r, i)

        # ---- Decoder ----
        dec_feats: List[Tuple[Tensor, Tensor]] = []
        for idx, layer in enumerate(self.decoder):
            sr, si = skips[-(idx + 1)]

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
    print(f"  {'Total':<10}: {total/1e6:.3f} M")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy  = torch.randn(2, 16_000 * 4).to(device)

    for mode in ('DCCRN-E', 'DCCRN-CL'):
        model = DCCRN(mode=mode).to(device)
        with torch.no_grad():
            enh, ef, df = model(dummy)

        print(f"\n{'='*45}")
        print(f"Mode: {mode}  (lightweight)")
        param_breakdown(model)
        t0 = time.time()
        with torch.no_grad():
            enh, ef, df = model(dummy)
        print(f"  {'Output':<10}: {tuple(enh.shape)}")
        print(f"  {'Time':<10}: {(time.time()-t0)*1000:.1f} ms")