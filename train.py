import os
import argparse
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Import our custom models, dataloader, and loss functions
from src.dataloader.dataloader import get_dataloader
from src.dataloader.dccrn_dataloader import get_dccrn_dataloader
from src.loss.waveunet_loss import si_snr_loss as waveunet_si_snr_loss, waveunet_total
from src.loss.dccrn_loss import si_snr_loss as dccrn_si_snr_loss

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
def get_args():
    parser = argparse.ArgumentParser(description="Speech Enhancement Training")
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    return ConfigArgs(config_dict)

class ConfigArgs:
    def __init__(self, cfg):
        # Data Section
        self.input_path = cfg['data']['input_path']
        self.tr_list = cfg['data']['tr_list']
        self.cv_list = cfg['data']['cv_list']
        self.sampling_rate = cfg['data']['sampling_rate']
        self.max_length = cfg['data']['max_length']
        self.batch_size = cfg['data']['batch_size']
        self.num_workers = cfg['data']['num_workers']

        # Network Section
        self.network = cfg['network']['name']

        # Model Section
        self.in_channels = cfg['model']['in_channels']
        self.out_channels = cfg['model']['out_channels']
        self.num_layers = cfg['model'].get('num_layers', 12)
        self.channels_interval = cfg['model'].get('channels_interval', 24)
        
        # DCCRN Specific
        self.mode = cfg['model'].get('mode', 'DCCRN-E')
        self.causal = cfg['model'].get('causal', True)

        # Training Section
        self.epochs = cfg['training']['epochs']
        self.lr_g = cfg['training']['lr_g']
        self.lr_d = cfg['training']['lr_d']
        self.device = device
        self.log_dir = cfg['training']['log_dir']
        # Scheduler params (optional, with sensible defaults)
        self.scheduler_eta_min = cfg['training'].get('scheduler_eta_min', 1e-6)  # for Cosine
        self.scheduler_gamma   = cfg['training'].get('scheduler_gamma',   0.999) # for Exponential

        # Loss Section
        self.si_snr_weight = cfg['loss'].get('si_snr_weight', 0.0)
        self.stft_weight  = cfg['loss'].get('stft_weight', 0.0)

        # STFT params (for spectral loss)
        self.n_fft      = cfg['loss'].get('n_fft',      512)
        self.hop_length = cfg['loss'].get('hop_length', 100)
        self.win_length = cfg['loss'].get('win_length', 400)

        # Distributed Section
        self.distributed = cfg['distributed']['use_ddp']
        self.world_size = cfg['distributed']['world_size']
        self.local_rank = cfg['distributed']['local_rank']


def train(args):
    writer = SummaryWriter(log_dir=args.log_dir)

    print(f"Loading data using device: {device}")
    
    # Load Dataloader
    # Dataloader implementation requires a single 'args' namespace.
    if args.network == "DCCRN":
        _, train_loader = get_dccrn_dataloader(args, 'train')
        _, val_loader = get_dccrn_dataloader(args, 'val')
    else:
        _, train_loader = get_dataloader(args, 'train')
        _, val_loader = get_dataloader(args, 'val')
    # Initialize Model
    print(f"Initializing Network: {args.network}")
    if args.network == "WaveUnet":
        from src.models.WaveUnet.WaveUNet import WaveUnet
        model = WaveUnet(
            in_channels=args.in_channels,
            n_layers=args.num_layers,
            channels_interval=args.channels_interval
        )
    elif args.network == "DCCRN":
        from src.models.DCCRN.dccrn import DCCRN
        model = DCCRN(
            mode=args.mode,
            causal=args.causal,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length
        )
    else:
        raise ValueError(f"Unsupported network: {args.network}")

    model.to(device)
    # Optimizers
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_g, betas=(0.8, 0.99))

    print(f"  Loss weights — SI-SNR: {args.si_snr_weight}, STFT: {args.stft_weight}")

    # ---------------------
    # LR Schedulers
    # ---------------------
    # CosineAnnealingLR: smooth decay for regression models
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.scheduler_eta_min
    )
    print(f"  Scheduler: CosineAnnealingLR (T_max={args.epochs}, eta_min={args.scheduler_eta_min})")

    print("Starting Training Loop...")
    global_step = 0
    best_si_snr = -float('inf')  # Track best validation SI-SNR (dB)
    save_dir = os.path.join('experiments', args.network)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_model.pt')

    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()

        # ── Per-epoch running accumulators ─────────────────────────────────
        # WaveUnet/DCCRN : total loss + SI-SNR component + MS-STFT component
        running_g_loss     = 0.0
        running_sisnr_loss = 0.0
        running_stft_loss  = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            noisy_audio = batch_data[0].to(device)   # [B, T]
            clean_audio = batch_data[1].to(device)   # [B, T]

            # ===========================================================
            # WaveUnet / DCCRN training
            #   loss_sisnr  : Scale-Invariant SNR loss (primary metric)
            #   loss_msstft : Multi-Scale STFT loss (spectral convergence)
            #   loss_total  : si_snr_weight*loss_sisnr + stft_weight*loss_msstft
            # ===========================================================
            optimizer.zero_grad()
            if args.network == "DCCRN":
                enhanced_audio, _, _ = model(noisy_audio)
                loss_sisnr = dccrn_si_snr_loss(enhanced_audio, clean_audio)
                loss_msstft = torch.tensor(0.0, device=device)
                loss_total = loss_sisnr
            else:
                enhanced_audio = model(noisy_audio)                    # [B, T]
                loss_total, loss_sisnr, loss_msstft = waveunet_total(
                    enhanced_audio, clean_audio,
                    args.si_snr_weight, args.stft_weight
                )
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_g_loss     += loss_total.item()
            running_sisnr_loss += loss_sisnr.item()
            running_stft_loss  += loss_msstft.item()
            global_step += 1

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{batch_idx}/{len(train_loader)}] "
                    f"[{args.network}] "
                    f"loss_total={loss_total.item():.4f}  "
                    f"loss_sisnr={loss_sisnr.item():.4f}  "
                    f"loss_msstft={loss_msstft.item():.4f}"
                )
                writer.add_scalar(f'{args.network}/Train/loss_total',  loss_total.item(),  global_step)
                writer.add_scalar(f'{args.network}/Train/loss_sisnr',  loss_sisnr.item(),  global_step)
                writer.add_scalar(f'{args.network}/Train/loss_msstft', loss_msstft.item(), global_step)

        epoch_time = time.time() - start_time
        n_train = len(train_loader)
        avg_total  = running_g_loss     / n_train
        avg_sisnr  = running_sisnr_loss / n_train
        avg_msstft = running_stft_loss  / n_train
        print(
            f"Epoch {epoch} done in {epoch_time:.2f}s  [{args.network}]  "
            f"avg loss_total={avg_total:.4f}  "
            f"avg loss_sisnr={avg_sisnr:.4f}  "
            f"avg loss_msstft={avg_msstft:.4f}"
        )
        # Epoch-level average train losses
        writer.add_scalar(f'{args.network}/Train/avg_loss_total',  avg_total,  epoch)
        writer.add_scalar(f'{args.network}/Train/avg_loss_sisnr',  avg_sisnr,  epoch)
        writer.add_scalar(f'{args.network}/Train/avg_loss_msstft', avg_msstft, epoch)

        # ---------------------
        # Validation — per-component losses + SI-SNR dB
        # ---------------------
        model.eval()
        val_sisnr_sum  = 0.0
        val_msstft_sum = 0.0
        val_total_sum  = 0.0
        n_val = len(val_loader)

        with torch.no_grad():
            for val_batch in val_loader:
                v_noisy = val_batch[0].to(device)
                v_clean = val_batch[1].to(device)

                if args.network == "DCCRN":
                    v_enhanced, _, _ = model(v_noisy)
                    v_sisnr_t = dccrn_si_snr_loss(v_enhanced, v_clean)
                    v_msstft_t = torch.tensor(0.0, device=device)
                    v_total = args.si_snr_weight * v_sisnr_t
                else:
                    v_enhanced = model(v_noisy)
                    # WaveUnet val: full loss breakdown
                    v_total, v_sisnr_t, v_msstft_t = waveunet_total(
                        v_enhanced, v_clean,
                        args.si_snr_weight, args.stft_weight
                    )
                val_sisnr_sum  += v_sisnr_t.item()
                val_msstft_sum += v_msstft_t.item()
                val_total_sum  += v_total.item()

        val_avg_sisnr  = val_sisnr_sum  / n_val
        val_avg_msstft = val_msstft_sum / n_val
        val_avg_total  = val_total_sum  / n_val
        # loss_sisnr = -SI-SNR  →  actual SI-SNR dB = -loss
        current_si_snr_db = -val_avg_sisnr

        print(
            f"Validation [{args.network}] — SI-SNR: {current_si_snr_db:.2f} dB  "
            f"val_loss_total={val_avg_total:.4f}  "
            f"val_loss_sisnr={val_avg_sisnr:.4f}  "
            f"val_loss_msstft={val_avg_msstft:.4f}"
        )
        writer.add_scalar(f'{args.network}/Val/si_snr_db',    current_si_snr_db, epoch)
        writer.add_scalar(f'{args.network}/Val/loss_total',   val_avg_total,     epoch)
        writer.add_scalar(f'{args.network}/Val/loss_sisnr',   val_avg_sisnr,     epoch)
        writer.add_scalar(f'{args.network}/Val/loss_msstft',  val_avg_msstft,    epoch)

        # ---------------------
        # Save Best Model
        # ---------------------
        if current_si_snr_db > best_si_snr:
            best_si_snr = current_si_snr_db
            state = {
                'epoch':     epoch,
                'si_snr_db': best_si_snr,
                'args':      vars(args),
                'model_state_dict': model.state_dict()
            }
            torch.save(state, best_model_path)
            print(f"  ✅ Best model saved — SI-SNR: {best_si_snr:.2f} dB → {best_model_path}")
        else:
            print(f"  (No improvement, best so far: {best_si_snr:.2f} dB)\n")

        # ---------------------
        # Step LR Schedulers
        # ---------------------
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"LR updated — lr_g: {current_lr:.2e}")
        writer.add_scalar(f'{args.network}/LR', current_lr, epoch)



    writer.close()
    print("Training Complete!")

if __name__ == "__main__":
    args = get_args()
    
    # Create dummy scp files since user probably hasn't generated them yet. 
    # Just to prevent FileNotFound if they run it accidentally.
    os.makedirs('data', exist_ok=True)
    for scp_file in [args.tr_list, args.cv_list]:
        if not os.path.exists(scp_file):
            print(f"Warning: {scp_file} does not exist. Creating a dummy file.")
            with open(scp_file, 'w') as f:
                f.write("") # Dataloader needs file to not throw error

    # Note: the script will fail fast if the dummy files are empty. 
    # Provide helpful message here:
    if os.path.getsize(args.tr_list) == 0:
        print(f"\n[!] DATASET MISSING: The file '{args.tr_list}' is empty.")
        print("Please populate 'data/train.scp', 'data/test.scp' with data format:")
        print("noisy_audio_path clean_audio_path duration")
    else:
        train(args)
#python train.py --config config/waveunet.yaml
#python train.py --config config/dccrn.yaml
