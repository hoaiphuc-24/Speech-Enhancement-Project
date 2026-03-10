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
from src.loss.waveunet_loss      import si_snr_loss as waveunet_si_snr_loss, waveunet_total
from src.loss.mossformergan_loss import loss_mossformergan_se_16k, stft, power_compress

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

    if is_gan:
        print(f"  GAN Loss: spectral-domain (STFT n_fft={args.n_fft}, hop={args.hop_length}, win={args.win_length})")
    else:
        print(f"  Loss weights — SI-SNR: {args.si_snr_weight}, STFT: {args.stft_weight}")

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
    best_si_snr = -float('inf')  # Track best validation SI-SNR (dB)
    save_dir = os.path.join('experiments', args.network)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_model.pt')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()
        
        running_g_loss    = 0.0
        running_d_loss    = 0.0
        running_sisnr_loss = 0.0
        running_stft_loss  = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            # Dataloader currently yields (inputs, labels) or (inputs, labels, fbanks)
            noisy_audio = batch_data[0].to(device)   # [Batch, Time]
            clean_audio = batch_data[1].to(device)   # [Batch, Time]

            if is_gan:
                # -------------------------------------------------------
                # MossFormerGAN — spectral-domain loss (loss_mossformergan_se_16k)
                # Generator outputs waveform → convert to spectrogram here
                # -------------------------------------------------------

                # 1. Forward pass: generator produces enhanced waveform
                enhanced_waveform = model.generator(noisy_audio)  # [B, T]

                # 2. STFT → [B, 2, F, T], then unsqueeze channel → [B, 1, T, F] (permuted)
                #    loss fn expects out_list[i].permute(0,1,3,2) → [B,1,F,T]
                enhanced_spec = stft(enhanced_waveform, args, center=True).to(torch.float32)
                enhanced_spec = power_compress(enhanced_spec)        # [B, 2, F, T]
                pred_real = enhanced_spec[:, 0, :, :].unsqueeze(1)  # [B, 1, F, T]
                pred_imag = enhanced_spec[:, 1, :, :].unsqueeze(1)  # [B, 1, F, T]
                # Permute to match expected input [B, 1, T, F] (loss will permute back)
                out_list = [
                    pred_real.permute(0, 1, 3, 2),  # [B, 1, T, F]
                    pred_imag.permute(0, 1, 3, 2),  # [B, 1, T, F]
                ]

                # 3. amplitude scale factor c = 1 (no external scaling)
                c = torch.ones(clean_audio.shape[0]).to(device)

                # 4. Compute generator + discriminator loss
                optim_g.zero_grad()
                optim_d.zero_grad()
                loss_g_total, loss_d = loss_mossformergan_se_16k(
                    args, noisy_audio, clean_audio, out_list, c,
                    model.discriminator, device
                )

                if loss_g_total is None:
                    # Batch skipped (NaN or PESQ failure)
                    loss_d_item    = 0.0
                    loss_stft_item = 0.0
                    loss_g_total = torch.tensor(0.0, device=device, requires_grad=False)
                    loss_d       = torch.tensor(0.0, device=device, requires_grad=False)
                    loss_si_snr  = torch.tensor(0.0, device=device)
                else:
                    # 5. Backprop generator
                    loss_g_total.backward(retain_graph=True)
                    optim_g.step()

                    # 6. Backprop discriminator
                    loss_d.backward()
                    optim_d.step()

                    running_d_loss += loss_d.item()
                    loss_d_item    = loss_d.item()
                    loss_stft_item = 0.0
                    loss_si_snr = torch.tensor(0.0, device=device)
            else:
                # ---------------------
                # Train WaveUnet — via waveunet_total
                # ---------------------
                optim_g.zero_grad()
                enhanced_audio = model(noisy_audio)   # [B, T]
                loss_g_total, loss_si_snr, loss_stft = waveunet_total(
                    enhanced_audio, clean_audio,
                    args.si_snr_weight, args.stft_weight
                )
                loss_g_total.backward()
                optim_g.step()
                loss_d_item    = 0.0
                loss_stft_item = loss_stft.item()
                running_stft_loss += loss_stft_item

            # Logging
            running_g_loss    += loss_g_total.item()
            running_sisnr_loss += loss_si_snr.item()
            global_step += 1

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Step [{batch_idx}/{len(train_loader)}], "
                      f"Loss G: {loss_g_total.item():.4f}, Loss D: {loss_d_item:.4f}, "
                      f"SI-SNR: {loss_si_snr.item():.4f}, STFT: {loss_stft_item:.4f}")

                writer.add_scalar('Training/Loss_G',      loss_g_total.item(), global_step)
                writer.add_scalar('Training/Loss_SI-SNR', loss_si_snr.item(),  global_step)
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
        with torch.no_grad():
            for val_batch in val_loader:
                v_noisy = val_batch[0].to(device)
                v_clean = val_batch[1].to(device)
                if is_gan:
                    v_enhanced = model.generator(v_noisy)
                else:
                    v_enhanced = model(v_noisy)
                val_si_snr += waveunet_si_snr_loss(v_enhanced, v_clean).item()

        val_si_snr /= len(val_loader)
        # SI-SNR loss is negative, so actual SI-SNR (dB) = -val_si_snr
        current_si_snr_db = -val_si_snr
        print(f"Validation — SI-SNR: {current_si_snr_db:.2f} dB")
        writer.add_scalar('Validation/SI-SNR_dB', current_si_snr_db, epoch)

        # ---------------------
        # Save Best Model
        # ---------------------
        if current_si_snr_db > best_si_snr:
            best_si_snr = current_si_snr_db
            state = {
                'epoch':     epoch,
                'si_snr_db': best_si_snr,
                'args':      vars(args),
            }
            if is_gan:
                state['generator_state_dict']     = model.generator.state_dict()
                state['discriminator_state_dict'] = model.discriminator.state_dict()
            else:
                state['model_state_dict'] = model.state_dict()
            torch.save(state, best_model_path)
            print(f"  ✅ Best model saved — SI-SNR: {best_si_snr:.2f} dB → {best_model_path}")
        else:
            print(f"  (No improvement, best so far: {best_si_snr:.2f} dB)\n")

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
