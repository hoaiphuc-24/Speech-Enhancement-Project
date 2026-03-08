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
from src.loss.mossformergan_loss import (discriminator_loss, mossformergan_g_total)

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
    best_si_snr = -float('inf')  # Track best validation SI-SNR (dB)
    save_dir = os.path.join('experiments', args.network)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_model.pt')
    
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
                # Train Generator — via mossformergan_g_total
                # ---------------------
                optim_g.zero_grad()
                _, y_d_gs, _, _ = model.discriminator(clean_audio, enhanced_audio)
                loss_g_total, _, loss_l1, loss_si_snr = mossformergan_g_total(
                    enhanced_audio, clean_audio, y_d_gs,
                    l1_loss_fn, args.l1_weight, args.si_snr_weight
                )
                loss_g_total.backward()
                optim_g.step()
                running_d_loss += loss_d.item()
                loss_d_item    = loss_d.item()
                loss_stft_item = 0.0
            else:
                # ---------------------
                # Train WaveUnet — via waveunet_total
                # ---------------------
                optim_g.zero_grad()
                enhanced_audio = model(noisy_audio)   # [B, T]
                loss_g_total, loss_l1, loss_si_snr, loss_stft = waveunet_total(
                    enhanced_audio, clean_audio, l1_loss_fn,
                    args.l1_weight, args.si_snr_weight, args.stft_weight
                )
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
                    val_si_snr += waveunet_si_snr_loss(v_enhanced, v_clean).item()
                else:
                    v_enhanced = model(v_noisy)
                    val_si_snr += waveunet_si_snr_loss(v_enhanced, v_clean).item()
                val_l1 += l1_loss_fn(v_enhanced, v_clean).item()

        val_si_snr /= len(val_loader)
        val_l1     /= len(val_loader)
        # SI-SNR loss is negative, so actual SI-SNR (dB) = -val_si_snr
        current_si_snr_db = -val_si_snr
        print(f"Validation — SI-SNR: {current_si_snr_db:.2f} dB, L1: {val_l1:.4f}")
        writer.add_scalar('Validation/SI-SNR_dB', current_si_snr_db, epoch)
        writer.add_scalar('Validation/Loss_L1',   val_l1,            epoch)

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
