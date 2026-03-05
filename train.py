import os
import argparse
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Import our custom models and dataloader
from src.dataloader.dataloader import get_dataloader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
        self.hidden_channels = cfg['model'].get('hidden_channels', 256)
        self.num_blocks = cfg['model'].get('num_blocks', 4)
        self.num_layers = cfg['model'].get('num_layers', 12)
        self.channels_interval = cfg['model'].get('channels_interval', 24)

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
        self.fm_weight    = cfg['loss'].get('fm_weight', 0.0)
        self.l1_weight    = cfg['loss'].get('l1_weight', 1.0)
        self.si_snr_weight = cfg['loss'].get('si_snr_weight', 0.0)
        self.stft_weight  = cfg['loss'].get('stft_weight', 0.0)

        # Distributed Section
        self.distributed = cfg['distributed']['use_ddp']
        self.world_size = cfg['distributed']['world_size']
        self.local_rank = cfg['distributed']['local_rank']

def feature_loss(fmap_r, fmap_g):
    """Calculates Feature Matching Loss between real and fake features derived from discriminator."""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2 # scale factor

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Calculates Discriminator Loss. It learns to output 1 for real and 0 for fake."""
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    """Calculates Generator Loss from Discriminator output. It wants Discriminator to output 1 for fake."""
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses


def si_snr_loss(estimated, target, eps=1e-8):
    """
    Scale-Invariant Signal-to-Noise Ratio Loss.
    Standard metric for speech enhancement — directly optimizes perceptual quality.
    Input: [B, T] float32 tensors.
    Returns: scalar loss (negative SI-SNR, so lower = better).
    """
    # Remove DC offset
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target    = target    - target.mean(dim=-1, keepdim=True)
    # Projection of estimated onto target
    dot          = (estimated * target).sum(dim=-1, keepdim=True)
    target_power = (target ** 2).sum(dim=-1, keepdim=True) + eps
    s_target     = dot * target / target_power          # scaled target
    e_noise      = estimated - s_target                 # noise residual
    si_snr = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps
    )
    return -si_snr.mean()  # minimise negative SI-SNR


def ms_stft_loss(estimated, target, fft_sizes=(512, 1024, 2048),
                 hop_sizes=(128, 256, 512), win_sizes=(512, 1024, 2048)):
    """
    Multi-Scale STFT Loss — evaluates reconstruction quality across multiple
    frequency resolutions.
    Input: [B, T] float32 tensors.
    Returns: scalar loss.
    """
    loss = 0.0
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        window = torch.hann_window(win_size, device=estimated.device)
        # Spectral convergence + log magnitude loss
        S_est = torch.stft(estimated.reshape(-1, estimated.shape[-1]),
                           n_fft=fft_size, hop_length=hop_size, win_length=win_size,
                           window=window, return_complex=True)
        S_tgt = torch.stft(target.reshape(-1, target.shape[-1]),
                           n_fft=fft_size, hop_length=hop_size, win_length=win_size,
                           window=window, return_complex=True)
        mag_est = S_est.abs() + 1e-8
        mag_tgt = S_tgt.abs() + 1e-8
        # Spectral convergence
        sc_loss  = torch.norm(mag_tgt - mag_est, 'fro') / (torch.norm(mag_tgt, 'fro') + 1e-8)
        # Log-STFT magnitude
        log_loss = F.l1_loss(torch.log(mag_est), torch.log(mag_tgt))
        loss += sc_loss + log_loss
    return loss / len(fft_sizes)


def train(args):
    writer = SummaryWriter(log_dir=args.log_dir)

    print(f"Loading data using device: {device}")
    
    # Load Dataloader
    # Dataloader implementation requires a single 'args' namespace.
    _, train_loader = get_dataloader(args, 'train')
    _, val_loader = get_dataloader(args, 'val')
    print(train_loader)
    print(val_loader)
    # Initialize Model
    print(f"Initializing Network: {args.network}")
    if args.network == "MossFormerGAN_SE_16K":
        from src.models.MossFormerGAN import MossFormerGAN_SE_16K
        model = MossFormerGAN_SE_16K(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks
        )
        is_gan = True
    elif args.network == "WaveUnet":
        from src.models.WaveUnet.WaveUNet import WaveUnet
        model = WaveUnet(
            in_channels=args.in_channels,
            n_layers=args.num_layers,
            channels_interval=args.channels_interval
        )
        is_gan = False
    else:
        raise ValueError(f"Unsupported network: {args.network}")

    model.to(device)

    # Optimizers
    if is_gan:
        optim_g = optim.AdamW(model.generator.parameters(), lr=args.lr_g, betas=(0.8, 0.99))
        optim_d = optim.AdamW(model.discriminator.parameters(), lr=args.lr_d, betas=(0.8, 0.99))
    else:
        optim_g = optim.AdamW(model.parameters(), lr=args.lr_g, betas=(0.8, 0.99))
        optim_d = None

    # Basic L1 Loss for Time-Domain Reconstruction
    l1_loss_fn = nn.L1Loss()
    print(f"  Loss weights — L1: {args.l1_weight}, SI-SNR: {args.si_snr_weight}, STFT: {args.stft_weight}")

    # ---------------------
    # LR Schedulers
    # ---------------------
    if is_gan:
        # ExponentialLR: stable decay for GAN training
        scheduler_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=args.scheduler_gamma)
        scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=args.scheduler_gamma)
        print(f"  Scheduler: ExponentialLR (gamma={args.scheduler_gamma})")
    else:
        # CosineAnnealingLR: smooth decay for regression models
        scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            optim_g, T_max=args.epochs, eta_min=args.scheduler_eta_min
        )
        scheduler_d = None
        print(f"  Scheduler: CosineAnnealingLR (T_max={args.epochs}, eta_min={args.scheduler_eta_min})")

    print("Starting Training Loop...")
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()
        
        running_g_loss    = 0.0
        running_d_loss    = 0.0
        running_l1_loss   = 0.0
        running_sisnr_loss = 0.0
        running_stft_loss  = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            # Dataloader currently yields (inputs, labels) or (inputs, labels, fbanks)
            noisy_audio = batch_data[0].to(device)   # [Batch, Time]
            clean_audio = batch_data[1].to(device)   # [Batch, Time]

            if is_gan:
                # ---------------------
                # Train Discriminator
                # ---------------------
                optim_d.zero_grad()
                enhanced_audio = model.generator(noisy_audio)
                y_d_rs, y_d_gs, _, _ = model.discriminator(clean_audio, enhanced_audio.detach())
                loss_d, _, _ = discriminator_loss(y_d_rs, y_d_gs)
                loss_d.backward()
                optim_d.step()
    
                # ---------------------
                # Train Generator — Adversarial + L1 + SI-SNR
                # ---------------------
                optim_g.zero_grad()
                y_d_rs, y_d_gs, _, _ = model.discriminator(clean_audio, enhanced_audio)
                loss_g_fake, _  = generator_loss(y_d_gs)
                loss_l1         = l1_loss_fn(enhanced_audio, clean_audio)
                loss_si_snr     = si_snr_loss(enhanced_audio, clean_audio)

                loss_g_total = (loss_g_fake
                                + args.l1_weight    * loss_l1
                                + args.si_snr_weight * loss_si_snr)

                loss_g_total.backward()
                optim_g.step()
                running_d_loss += loss_d.item()
                loss_d_item = loss_d.item()
                loss_stft_item = 0.0
            else:
                # ---------------------
                # Train WaveUnet — L1 + SI-SNR + Multi-Scale STFT
                # ---------------------
                optim_g.zero_grad()
                enhanced_audio = model(noisy_audio)       # [B, T]
                loss_l1         = l1_loss_fn(enhanced_audio, clean_audio)
                loss_si_snr     = si_snr_loss(enhanced_audio, clean_audio)
                loss_stft       = ms_stft_loss(enhanced_audio, clean_audio)

                loss_g_total = (args.l1_weight     * loss_l1
                                + args.si_snr_weight * loss_si_snr
                                + args.stft_weight   * loss_stft)

                loss_g_total.backward()
                optim_g.step()
                loss_d_item    = 0.0
                loss_stft_item = loss_stft.item()
                running_stft_loss += loss_stft_item

            # Logging
            running_g_loss    += loss_g_total.item()
            running_l1_loss   += loss_l1.item()
            running_sisnr_loss += loss_si_snr.item()
            global_step += 1

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Step [{batch_idx}/{len(train_loader)}], "
                      f"Loss G: {loss_g_total.item():.4f}, Loss D: {loss_d_item:.4f}, "
                      f"L1: {loss_l1.item():.4f}, SI-SNR: {loss_si_snr.item():.4f}, STFT: {loss_stft_item:.4f}")

                writer.add_scalar('Training/Loss_G',     loss_g_total.item(),   global_step)
                writer.add_scalar('Training/Loss_L1',    loss_l1.item(),        global_step)
                writer.add_scalar('Training/Loss_SI-SNR', loss_si_snr.item(),   global_step)
                if is_gan:
                    writer.add_scalar('Training/Loss_D', loss_d_item, global_step)
                else:
                    writer.add_scalar('Training/Loss_STFT', loss_stft_item, global_step)

        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} Completed in {epoch_time:.2f}s."
              f"\nAvg G Loss: {running_g_loss/len(train_loader):.4f}, "
              f"Avg D Loss: {running_d_loss/len(train_loader):.4f}, "
              f"Avg SI-SNR Loss: {running_sisnr_loss/len(train_loader):.4f}")

        # ---------------------
        # Validation — SI-SNR as primary metric
        # ---------------------
        model.eval()
        val_si_snr = 0.0
        val_l1     = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                v_noisy = val_batch[0].to(device)
                v_clean = val_batch[1].to(device)
                if is_gan:
                    v_enhanced = model.generator(v_noisy)
                else:
                    v_enhanced = model(v_noisy)
                val_si_snr += si_snr_loss(v_enhanced, v_clean).item()
                val_l1     += l1_loss_fn(v_enhanced, v_clean).item()

        val_si_snr /= len(val_loader)
        val_l1     /= len(val_loader)
        # SI-SNR loss is negative, so actual SI-SNR (dB) = -val_si_snr
        print(f"Validation — SI-SNR: {-val_si_snr:.2f} dB, L1: {val_l1:.4f}\n")
        writer.add_scalar('Validation/SI-SNR_dB', -val_si_snr, epoch)
        writer.add_scalar('Validation/Loss_L1',    val_l1,      epoch)

        # ---------------------
        # Step LR Schedulers
        # ---------------------
        scheduler_g.step()
        if scheduler_d is not None:
            scheduler_d.step()
        current_lr = scheduler_g.get_last_lr()[0]
        print(f"LR updated — lr_g: {current_lr:.2e}")
        writer.add_scalar('Training/LR_G', current_lr, epoch)



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
