import os
import argparse
import numpy as np
import torch
import soundfile as sf
from src.dataloader.dataloader import audioread

# ─── Mặc định ────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = "tests"
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device):
    """Load checkpoint và khởi tạo WaveUnet / DCCRN tương ứng."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Can't find model: {model_path}")

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
    elif network == "DCCRN":
        from src.models.DCCRN.dccrn import DCCRN
        model = DCCRN(
            mode       = args_dict.get("mode", "DCCRN-E"),
            causal     = args_dict.get("causal", True),
            n_fft      = args_dict.get("n_fft", 512),
            hop_length = args_dict.get("hop_length", 100),
            win_length = args_dict.get("win_length", 400),
        )
        # Chạy dummy forward để trigger _build_lstm()
        dummy_len = args_dict.get("n_fft", 512) * 10
        with torch.no_grad():
            dummy = torch.zeros(1, dummy_len).to(device)
            model(dummy)
    else:
        raise ValueError(f"Network is not defined: {network}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    sr = args_dict.get("sampling_rate", 16000)
    print(f"[✓] Đã load '{network}' — epoch {checkpoint.get('epoch', 'N/A')} "
          f"| best val SI-SNR: {checkpoint.get('si_snr_db', 0.0):.2f} dB "
          f"| SR: {sr} Hz")
    return model, args_dict


def enhance_file(model, input_path: str, output_path: str,
                 sampling_rate: int, device: torch.device) -> None:
    """Đọc file noisy → model → lưu file enhanced."""
    # Đọc audio bằng hàm audioread giống hệt lúc training để đảm bảo matching data distribution
    try:
        data_norm, inv_scalar = audioread(input_path, sampling_rate)
    except Exception as e:
        print(f"  [!] Error when reading '{input_path}': {e}")
        return

    data_norm = data_norm.astype(np.float32)

    # Inference
    with torch.no_grad():
        x = torch.FloatTensor(data_norm).unsqueeze(0).to(device)  # [1, T]

        if model.__class__.__name__ == "DCCRN":
            out, _, _ = model(x)
        else:
            out = model(x)                                        # [1, T] hoặc [1, 1, T]

        out = out.squeeze().cpu().numpy()                         # [T]

    # Khôi phục biên độ gốc (do audioread trả về inv_scalar trực tiếp theo cách norm của training)
    enhanced = out * inv_scalar

    # Lưu kết quả
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    sf.write(output_path, enhanced, sampling_rate)
    print(f"  [✓] {os.path.basename(input_path)}  →  {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Speech Enhancement Inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Path đến file model checkpoint (.pt)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path đến file âm thanh cần nhiễu (noisy WAV)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Thư mục chứa kết quả âm thanh đã xử lý (mặc định: {DEFAULT_OUTPUT_DIR})")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, args_dict = load_model(args.model, device)
    sampling_rate = args_dict.get("sampling_rate", 16000)

    # Xử lý 1 file
    if not os.path.isfile(args.input):
        print(f"[!] Error: Input file doesn't exist: {args.input}")
        return

    basename = os.path.basename(args.input+args.network)
    out_path = os.path.join(args.output_dir, basename)

    print(f"\nProcessing file: '{args.input}' ...")
    enhance_file(model, args.input, out_path, sampling_rate, device)
    print(f"\n[Done] '{out_path}'\n")


if __name__ == "__main__":
    main()
