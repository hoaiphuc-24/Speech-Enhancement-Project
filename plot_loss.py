"""
plot_loss.py — Vẽ biểu đồ train/val loss từ TensorBoard event files.
Usage:
    python plot_loss.py --logdir runs/WaveUnet
    python plot_loss.py --logdir runs/MossFormerGAN
    python plot_loss.py --logdir runs/WaveUnet --out waveunet_loss.png
"""

import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Tag definitions  (tag, label, color)  ────────────────────────────────────

WAVEUNET_TRAIN_STEP = [
    ('WaveUnet/Train/loss_total',  'train loss_total',  'tab:blue'),
    ('WaveUnet/Train/loss_sisnr',  'train loss_sisnr',  'tab:orange'),
    ('WaveUnet/Train/loss_msstft', 'train loss_msstft', 'tab:green'),
]
WAVEUNET_TRAIN_EPOCH = [
    ('WaveUnet/Train/avg_loss_total',  'avg train total',  'tab:blue'),
    ('WaveUnet/Train/avg_loss_sisnr',  'avg train sisnr',  'tab:orange'),
    ('WaveUnet/Train/avg_loss_msstft', 'avg train msstft', 'tab:green'),
]
WAVEUNET_VAL_EPOCH = [
    ('WaveUnet/Val/loss_total',  'val loss_total',  'tab:blue'),
    ('WaveUnet/Val/loss_sisnr',  'val loss_sisnr',  'tab:orange'),
    ('WaveUnet/Val/loss_msstft', 'val loss_msstft', 'tab:green'),
]
WAVEUNET_VAL_SISNR = [
    ('WaveUnet/Val/si_snr_db', 'Val SI-SNR (dB)', 'tab:purple'),
]

MOSSFORMERGAN_TRAIN_STEP = [
    ('MossFormerGAN/Train/loss_gan_g', 'train loss_g', 'tab:blue'),
    ('MossFormerGAN/Train/loss_gan_d', 'train loss_d', 'tab:red'),
]
MOSSFORMERGAN_TRAIN_EPOCH = [
    ('MossFormerGAN/Train/avg_loss_gan_g', 'avg train loss_g', 'tab:blue'),
    ('MossFormerGAN/Train/avg_loss_gan_d', 'avg train loss_d', 'tab:red'),
]
MOSSFORMERGAN_VAL_SISNR = [
    ('MossFormerGAN/Val/si_snr_db', 'Val SI-SNR (dB)', 'tab:purple'),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(ea: EventAccumulator, tag: str):
    if tag not in ea.Tags().get('scalars', []):
        return [], []
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def plot_panel(ax, ea, tags, title, ylabel='Loss'):
    plotted = False
    for tag, label, color in tags:
        steps, values = load(ea, tag)
        if steps:
            ax.plot(steps, values, label=label, color=color, linewidth=1.5)
            plotted = True
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Step / Epoch', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    return plotted


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--out',    type=str, default=None, help='Save as PNG')
    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        print(f'[!] Not found: {args.logdir}')
        return

    print(f'Reading: {args.logdir}')
    ea = EventAccumulator(args.logdir)
    ea.Reload()
    avail = ea.Tags().get('scalars', [])
    print(f'Tags found: {avail}')

    is_waveunet     = any('WaveUnet'      in t for t in avail)
    is_mossformergan = any('MossFormerGAN' in t for t in avail)

    if is_waveunet:
        # ── WaveUnet: 4 panels ────────────────────────────────────────────
        #  [0] step-level train loss  [1] epoch avg train vs val loss
        #  [2] val components         [3] val SI-SNR dB
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f'WaveUnet — {os.path.basename(args.logdir)}', fontsize=13, fontweight='bold')

        plot_panel(axes[0, 0], ea, WAVEUNET_TRAIN_STEP,  'Train Loss (per step)')
        plot_panel(axes[0, 1], ea, WAVEUNET_TRAIN_EPOCH, 'Avg Train Loss (per epoch)')

        # Val components panel: overlay avg_train and val on same axes
        for tag, label, color in WAVEUNET_TRAIN_EPOCH:
            steps, values = load(ea, tag)
            if steps:
                axes[1, 0].plot(steps, values, label=label, color=color,
                                linewidth=1.5, linestyle='--', alpha=0.6)
        for tag, label, color in WAVEUNET_VAL_EPOCH:
            steps, values = load(ea, tag)
            if steps:
                axes[1, 0].plot(steps, values, label=label, color=color, linewidth=1.5)
        axes[1, 0].set_title('Train vs Val Loss (per epoch)', fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=9)
        axes[1, 0].set_ylabel('Loss', fontsize=9)
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.25)

        plot_panel(axes[1, 1], ea, WAVEUNET_VAL_SISNR, 'Validation SI-SNR (dB)', ylabel='SI-SNR (dB)')

    elif is_mossformergan:
        # ── MossFormerGAN: 3 panels ───────────────────────────────────────
        fig, axes_flat = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'MossFormerGAN — {os.path.basename(args.logdir)}', fontsize=13, fontweight='bold')

        plot_panel(axes_flat[0], ea, MOSSFORMERGAN_TRAIN_STEP,  'Train Loss (per step)')
        plot_panel(axes_flat[1], ea, MOSSFORMERGAN_TRAIN_EPOCH, 'Avg Train Loss (per epoch)')
        plot_panel(axes_flat[2], ea, MOSSFORMERGAN_VAL_SISNR,   'Validation SI-SNR (dB)', ylabel='SI-SNR (dB)')
    else:
        print('[!] No WaveUnet or MossFormerGAN tags found.')
        return

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches='tight')
        print(f'Saved: {args.out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
