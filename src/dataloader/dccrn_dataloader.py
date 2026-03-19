import os
import random
import numpy as np
import torch
import soundfile as sf
import librosa
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from src.dataloader.dataloader import read_and_config_file, audio_norm, audioread, DistributedSampler

# ---------------------------------------------------------------------------
# DCCRN Dataloader Helper Functions
# ---------------------------------------------------------------------------
def collate_fn_dccrn(batch):
    """
    Collate function to prepare raw input/target waveforms.
    batch is a list of tuples: (inputs, labels), where each is a numpy array.
    """
    inputs, labels = zip(*batch)
    x = torch.FloatTensor(np.array(inputs))
    y = torch.FloatTensor(np.array(labels))
    return x, y


class DCCRNAudioDataset(Dataset):
    """
    Dataloader dataset exclusively for models like DCCRN that directly process waveforms
    and don't need CPU-side Fbank extraction.
    """
    def __init__(self, args, data_type):
        self.args = args
        self.sampling_rate = args.sampling_rate
        if data_type == 'train':
            self.wav_list = read_and_config_file(args.tr_list)
        elif data_type == 'val':
            self.wav_list = read_and_config_file(args.cv_list)
        elif data_type == 'test':
            self.wav_list = read_and_config_file(args.tt_list)
        else:
            print(f'Data type: {data_type} is unknown!')

        self.segment_length = int(self.sampling_rate * self.args.max_length)
        print(f'[DCCRN] No. {data_type} files: {len(self.wav_list)}')

    def __len__(self):
        return len(self.wav_list)

    def process_wave(self, path_dict):
        """Read and align/truncate audio for DCCRN models"""
        wave_inputs, _ = audioread(path_dict['inputs'], self.sampling_rate)
        wave_labels, _ = audioread(path_dict['labels'], self.sampling_rate)
        
        len_wav = wave_labels.shape[0]
        if wave_inputs.shape[0] < self.segment_length:
            padded_inputs = np.zeros(self.segment_length, dtype=np.float32)
            padded_labels = np.zeros(self.segment_length, dtype=np.float32)
            padded_inputs[:wave_inputs.shape[0]] = wave_inputs
            padded_labels[:wave_labels.shape[0]] = wave_labels
        else:
            st_idx = random.randint(0, len_wav - self.segment_length)
            padded_inputs = wave_inputs[st_idx:st_idx + self.segment_length]
            padded_labels = wave_labels[st_idx:st_idx + self.segment_length]
            
        return padded_inputs, padded_labels

    def __getitem__(self, index):
        data_info = self.wav_list[index]
        inputs, labels = self.process_wave({
            'inputs': data_info['inputs'], 
            'labels': data_info['labels']
        })
        return inputs, labels


def get_dccrn_dataloader(args, data_type):
    """
    Instantiate the Dataloader explicitly for DCCRN with no excess processing.
    """
    datasets = DCCRNAudioDataset(args=args, data_type=data_type)

    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank
    ) if hasattr(args, 'distributed') and args.distributed else None

    generator = data.DataLoader(
        datasets,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        collate_fn=collate_fn_dccrn,
        num_workers=args.num_workers,
        sampler=sampler
    )
    return sampler, generator
