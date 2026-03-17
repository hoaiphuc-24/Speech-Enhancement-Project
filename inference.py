"""
inference.py — Speech Enhancement Inference
==============================================
Load model từ experiments/WaveUnet/best_model.pt và chạy inference
trên từng file noisy WAV, kết quả lưu vào thư mục tests/.

Cách dùng:
  # Chạy trên 1 file:
  python inference.py --input data/noisy_testset_wav/p232_001.wav

  # Chạy trên toàn bộ thư mục noisy:
  python inference.py --input_dir data/noisy_testset_wav

  # Chỉ định model / output dir khác:
  python inference.py --input_dir data/noisy_testset_wav \\
                      --model experiments/WaveUnet/best_model.pt \\
                      --output_dir tests
"""

import os
import glob
import argparse
import numpy as np
import torch
import soundfile as sf
import librosa

EPS = 1e-8

# ─── Mặc định ────────────────────────────────────────────────────────────────
DEFAULT_MODEL     = "experiments/WaveUnet/best_model.pt"
DEFAULT_INPUT_DIR = "data/noisy_testset_wav"
DEFAULT_OUTPUT_DIR = "tests"
# ─────────────────────────────────────────────────────────────────────────────


def load_model(model_path: str, device: torch.device):
    """Load checkpoint và khởi tạo WaveUnet / MossFormerGAN tương ứng."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    args_dict = checkpoint["args"]
    network   = args_dict["network"]

    if network == "WaveUnet":
        from src.models.WaveUnet.WaveUNet import WaveUnet
        model = WaveUnet(
            in_channels       = args_dict.get("in_channels", 1),
            n_layers          = args_dict.get("num_layers", 12),
            channels_interval = args_dict.get("channels_interval", 24),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    elif network == "DCCRN":
        from src.models.DCCRN.dccrn import DCCRN
        model = DCCRN(
            mode       = args_dict.get("mode", "DCCRN-E"),
            causal     = args_dict.get("causal", True),
            n_fft      = args_dict.get("n_fft", 512),
            hop_length = args_dict.get("hop_length", 100),
            win_length = args_dict.get("win_length", 400),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError(f"Network không xác định: {network}")

    model.to(device).eval()

    sr = args_dict.get("sampling_rate", 16000)
    print(f"[✓] Đã load '{network}' — epoch {checkpoint.get('epoch', 'N/A')} "
          f"| best val SI-SNR: {checkpoint.get('si_snr_db', 0.0):.2f} dB "
          f"| SR: {sr} Hz")
    return model, args_dict


def audio_norm(x: np.ndarray):
    """Chuẩn hóa RMS về -25 dBFS (giống training)."""
    rms = (x ** 2).mean() ** 0.5
    scalar = 10 ** (-25 / 20) / (rms + EPS)
    x = x * scalar
    avg_pow = (x ** 2).mean()
    rmsx = ((x ** 2)[x ** 2 > avg_pow].mean()) ** 0.5
    scalarx = 10 ** (-25 / 20) / (rmsx + EPS)
    x = x * scalarx
    inv_scalar = 1.0 / (scalar * scalarx + EPS)
    return x, inv_scalar


def enhance_file(model, input_path: str, output_path: str,
                 sampling_rate: int, device: torch.device) -> None:
    """Đọc file noisy → model → lưu file enhanced."""
    # Đọc audio
    try:
        data, fs = sf.read(input_path)
    except Exception as e:
        print(f"  [!] Bỏ qua '{input_path}': {e}")
        return

    if data.ndim > 1:          # stereo → mono
        data = data[:, 0]
    if fs != sampling_rate:
        data = librosa.resample(data, orig_sr=fs, target_sr=sampling_rate)

    data = data.astype(np.float32)
    data_norm, inv_scalar = audio_norm(data)

    # Inference
    with torch.no_grad():
        x = torch.FloatTensor(data_norm).unsqueeze(0).to(device)  # [1, T]
        if model.__class__.__name__ == "DCCRN":
            out, _, _ = model(x)
        else:
            out = model(x)                                              # [1, T] hoặc [1, 1, T]
        out = out.squeeze().cpu().numpy()                           # [T]

    # Khôi phục biên độ gốc
    enhanced = out * inv_scalar

    # Lưu kết quả
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    sf.write(output_path, enhanced, sampling_rate)
    print(f"  [✓] {os.path.basename(input_path)}  →  {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Speech Enhancement Inference — WaveUnet / DCCRN")
    parser.add_argument("--model",      type=str, default=DEFAULT_MODEL,
                        help=f"Path đến best_model.pt (mặc định: {DEFAULT_MODEL})")
    parser.add_argument("--input",      type=str, default=None,
                        help="Path đến 1 file noisy WAV (không bắt buộc nếu dùng --input_dir)")
    parser.add_argument("--input_dir",  type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Thư mục chứa các file noisy WAV (mặc định: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Thư mục đầu ra (mặc định: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--sr",         type=int, default=16000,
                        help="Sampling rate fallback nếu checkpoint không lưu (mặc định: 16000)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, args_dict = load_model(args.model, device)
    sampling_rate = args_dict.get("sampling_rate", args.sr)

    # Xây dựng danh sách file cần xử lý
    if args.input:
        # chế độ đơn file
        wav_files = [args.input]
    else:
        # chế độ thư mục
        wav_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
        if not wav_files:
            print(f"[!] Không tìm thấy file WAV nào trong: {args.input_dir}")
            return

    print(f"\nXử lý {len(wav_files)} file(s) → '{args.output_dir}/' ...\n")

    for wav_path in wav_files:
        basename  = os.path.basename(wav_path)
        out_path  = os.path.join(args.output_dir, basename)
        enhance_file(model, wav_path, out_path, sampling_rate, device)

    print(f"\n[Done] Kết quả đã lưu vào '{args.output_dir}/'")


if __name__ == "__main__":
    main()
