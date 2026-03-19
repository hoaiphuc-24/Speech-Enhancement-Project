# Speech Enhancement Project

Speech Enhancement Project for denoising noisy speech waveform using:
- `WaveUnet` (time-domain)
- `DCCRN` with 2 modes: `DCCRN-E` and `DCCRN-CL` (complex spectral domain)

The repository includes training, checkpoint-based inference, and a Streamlit demo app for quick listening comparison.

## 1. Main Features

- Train speech enhancement models from `.scp` pair lists (`noisy clean duration`)
- Validate by SI-SNR and save the best checkpoint automatically
- Support two model families:
  - `WaveUnet` with SI-SNR + Multi-Scale STFT loss
  - `DCCRN` with SI-SNR loss
- Offline inference from checkpoint `.pt`
- Streamlit demo: upload noisy audio, hear before/after, and download enhanced result

## 2. Project Structure

```text
SpeechEnhancementProject/
|-- app.py                             # Streamlit demo app
|-- train.py                           # Training entrypoint
|-- inference.py                       # Inference CLI
|-- plot_loss.py                       # Plot train/val loss from TensorBoard logs
|-- requirements.txt
|-- .gitignore
|-- README.md
|-- config/
|   |-- waveunet.yaml
|   `-- dccrn.yaml
|-- data/
|   |-- train.scp
|   |-- test.scp
|   `-- train_full.scp
|-- experiments/
|   |-- WaveUnet/
|   |   `-- best_model_w.pt
|   `-- DCCRN/
|       |-- best_model_e.pt
|       `-- best_model_cl.pt
|-- runs/                              # TensorBoard logs
|-- chart/                             # Saved loss charts
|-- notebooks/
|   `-- Train_On_Colab.ipynb            
|-- tests/                             # Save result after inference
`-- src/
    |-- dataloader/
    |   |-- dataloader.py
    |   `-- dccrn_dataloader.py
    |-- loss/
    |   |-- waveunet_loss.py
    |   `-- dccrn_loss.py
    `-- models/
        |-- WaveUnet/
        |   `-- WaveUNet.py
        `-- DCCRN/
            `-- dccrn.py
```

## 3. Installation

```bash
pip install -r requirements.txt
```

For Streamlit demo:

```bash
pip install streamlit
```

## 4. Dataset Format

Training/validation lists use `.scp` text files with each line:

```text
path/to/noisy.wav path/to/clean.wav duration_in_seconds
```

Example:

```text
data/noisy/p226_007.wav data/clean/p226_007.wav 3.84
```

Default config uses:
- Train list: `data/train.scp`
- Validation list: `data/test.scp`
- Sampling rate: `16000`

## 5. Training

### Train WaveUnet

```bash
python train.py --config config/waveunet.yaml
```

### Train DCCRN

```bash
python train.py --config config/dccrn.yaml
```

Training logs are written to:
- `runs/WaveUnet` (WaveUnet)
- `runs/DCCRN` (DCCRN)

Best model checkpoint is saved to:
- `experiments/<Network>/best_model.pt`

## 6. Inference (CLI)

Run enhancement on one audio file:

```bash
python inference.py --model experiments/DCCRN/best_model_cl.pt --input tests/p226_007.wav --output_dir tests
```

You can also use other checkpoints, e.g.:
- `experiments/DCCRN/best_model_e.pt`
- `experiments/WaveUnet/best_model_w.pt`

## 7. Streamlit Demo

Launch app:

```bash
streamlit run app.py
```

Demo flow:
1. Choose checkpoint path (default: `experiments/DCCRN/best_model_cl.pt`)
2. Upload audio file (`wav`, `flac`, `mp3`, `m4a`, `ogg`)
3. Click **Enhance audio**
4. Listen to original/enhanced side-by-side
5. Download enhanced `.wav`

## 8. Notes

- If no CUDA is available, training/inference automatically uses CPU.
- Ensure audio files are readable by `soundfile/librosa`.
- If `train.scp` is empty, training will not start (the script prints a dataset warning).

