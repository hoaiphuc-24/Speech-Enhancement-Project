import os
import argparse
import numpy as np
import torch
import soundfile as sf
import librosa

# ============================================================
# Inference script — Speech Enhancement
# Usage:
#   python inference.py --model experiments/WaveUnet/best_model.pt
#                       --input  path/to/noisy.wav
#                       --output path/to/enhanced.wav
# ============================================================

EPS = 1e-8

def load_model(model_path: str, device: torch.device):
    """Load best_model.pt và khởi tạo model tương ứng."""
    checkpoint = torch.load(model_path, map_location=device)

    args_dict = checkpoint['args']
    network   = args_dict['network']

    if network == 'WaveUnet':
        from src.models.WaveUnet.WaveUNet import WaveUnet
        model = WaveUnet(
            in_channels      = args_dict.get('in_channels', 1),
            n_layers         = args_dict.get('num_layers', 12),
            channels_interval= args_dict.get('channels_interval', 24),
        )
        model.load_state_dict(checkpoint['model_state_dict'])

    elif network == 'MossFormerGAN_SE_16K':
        from src.models.MossFormerGAN import MossFormerGAN_SE_16K
        model = MossFormerGAN_SE_16K(
            in_channels     = args_dict.get('in_channels', 1),
            out_channels    = args_dict.get('out_channels', 1),
            hidden_channels = args_dict.get('hidden_channels', 256),
            num_blocks      = args_dict.get('num_blocks', 4),
        )
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        # Use only generator for inference
        model = model.generator

    else:
        raise ValueError(f"Unknown network type: {network}")

    model.to(device)
    model.eval()

    print(f"Loaded '{network}' from epoch {checkpoint['epoch']} "
          f"(best val SI-SNR: {checkpoint['si_snr_db']:.2f} dB)")
    return model, args_dict


def audio_norm(x):
    """Normalize audio to a fixed RMS level (same as training)."""
    rms = (x ** 2).mean() ** 0.5
    scalar = 10 ** (-25 / 20) / (rms + EPS)
    x = x * scalar
    pow_x = x ** 2
    avg_pow_x = pow_x.mean()
    rmsx = pow_x[pow_x > avg_pow_x].mean() ** 0.5
    scalarx = 10 ** (-25 / 20) / (rmsx + EPS)
    x = x * scalarx
    return x, 1.0 / (scalar * scalarx + EPS)


def enhance(model, input_path: str, output_path: str,
            sampling_rate: int, device: torch.device):
    """
    Đọc file noisy, chạy qua model, lưu file enhanced.
    Hỗ trợ file dài tùy ý (không giới hạn độ dài).
    """
    # --- Load audio ---
    data, fs = sf.read(input_path)
    if len(data.shape) > 1:          # stereo → mono
        data = data[:, 0]
    if fs != sampling_rate:
        data = librosa.resample(data, orig_sr=fs, target_sr=sampling_rate)
        print(f"  Resampled {fs} Hz → {sampling_rate} Hz")

    data = data.astype(np.float32)
    data_norm, scalar = audio_norm(data)

    # --- Inference ---
    with torch.no_grad():
        x = torch.FloatTensor(data_norm).unsqueeze(0).to(device)  # [1, T]
        enhanced = model(x)                                         # [1, T]
        enhanced = enhanced.squeeze(0).cpu().numpy()                # [T]

    # --- Restore original scale ---
    enhanced = enhanced * scalar

    # --- Save output ---
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sf.write(output_path, enhanced, sampling_rate)
    print(f"  Saved enhanced audio → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Speech Enhancement Inference")
    parser.add_argument('--model',  type=str, required=True,
                        help="Path to best_model.pt (e.g. experiments/WaveUnet/best_model.pt)")
    parser.add_argument('--input',  type=str, required=True,
                        help="Path to noisy input .wav file")
    parser.add_argument('--output', type=str, default='enhanced.wav',
                        help="Path to save the enhanced .wav file (default: enhanced.wav)")
    parser.add_argument('--sr',     type=int, default=16000,
                        help="Sampling rate (default: 16000)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, args_dict = load_model(args.model, device)
    sampling_rate = args_dict.get('sampling_rate', args.sr)

    enhance(model, args.input, args.output, sampling_rate, device)
    print("Done!")


if __name__ == '__main__':
    main()
