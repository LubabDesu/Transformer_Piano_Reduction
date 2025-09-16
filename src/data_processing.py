import pretty_midi 
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Helper functions

def pad_or_truncate_tensor(x: torch.Tensor, L: int, pad_value: float = 0) -> torch.Tensor:
    """
    Pads or truncates a tensor along the time dimension (first axis) to length L.
    Works for 1D label arrays [T] or 2D/3D piano-rolls [T, ...].
    """
    if x.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions to pad/truncate along dim=1.")

    T = x.shape[1]  # Get the current length of the second dimension

    if T >= L:
        # If the sequence is too long, truncate it
        return x[:, :L]
    else:
        # If the sequence is too short, pad it
        # 1. Determine the shape of the padding needed
        pad_shape = (x.shape[0], L - T) + x.shape[2:]
        
        # 2. Create the padding tensor
        pad = torch.full(pad_shape, pad_value, dtype=x.dtype, device=x.device)
        
        # 3. Concatenate the original tensor and the padding along the second dimension
        return torch.cat([x, pad], dim=1)


def load_piano_midi_file(filepath,fs=16) :
    """ 
    This function computes a piano roll matrix from the piano (presumably reudced) MIDI data
    
    Parameters : 
    - filepath : path to the Orchestral MIDI file to convert it into piano roll format
    - fs : Sampling rate, default = 16
    
    Returns :
    - A piano roll of the piano_reduced target file 
    - 
    """
    midi_data = pretty_midi.PrettyMIDI(filepath)
    piano_roll = midi_data.get_piano_roll(fs=fs)
    piano_roll = (piano_roll > 0).astype(np.float64) #Binarize the piano roll 
    onset_labels, frame_labels, offset_labels = get_piano_labels(piano_roll=piano_roll)
    fname = os.path.basename(filepath)

    return piano_roll, onset_labels, frame_labels, offset_labels, fname


def load_midi_file(filepath, fs=16):
    """
    This function computes a piano roll matrix from the orchestral MIDI data

    Parameters :
    - filepath : path to the Orchestral MIDI file to convert it into piano roll format
    - fs : Sampling rate, default = 16

    Returns :
    - A piano roll of the orchestral input
    """
    # print(f"--- Starting to load MIDI file: {filepath} ---")
    midi_data = pretty_midi.PrettyMIDI(filepath)
    piano_roll = midi_data.get_piano_roll(fs=fs)
    piano_roll = (piano_roll > 0).astype(np.float64)
    fname = os.path.basename(filepath)

    return piano_roll, fname

def get_piano_labels(piano_roll):
    # piano_roll has shape [pitches, time]
    onset_labels = np.zeros_like(piano_roll)
    offset_labels = np.zeros_like(piano_roll)

    # For onset labels (note attacks)
    # A note is on at the first time step if it's active
    onset_labels[:, 0] = piano_roll[:, 0] > 0
    # A note is on at other time steps if it's active now but was not active before
    onset_labels[:, 1:] = (piano_roll[:, 1:] > 0) & (piano_roll[:, :-1] == 0)

    # For offset labels (note releases)
    # A note is off if it's inactive now but was active before
    offset_labels[:, 1:] = (piano_roll[:, 1:] == 0) & (piano_roll[:, :-1] > 0)

    # For frame labels (notes being held down)
    frame_labels = piano_roll.copy()

    return onset_labels.astype(np.int64), frame_labels.astype(np.int64), offset_labels.astype(np.int64)
    


class PianoReductionDataset(Dataset):
    def __init__(self, input_files, target_files, max_len, fs=16):
        # print(f"length of input files is {len(input_files)} and length of target files is {len(target_files)}\n\n")
        assert len(input_files) == len(target_files), "Input and target file counts must match"
        self.pairs = list(zip(input_files, target_files))
        print(f"length of pairs is {len(self.pairs)}")
        self.fs = fs
        self.max_len = max_len


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        orch_path, piano_path = self.pairs[idx]

        orch_roll, fname = load_midi_file(orch_path)
        piano_roll, onset_labels, frame_labels, offset_labels, fname = load_piano_midi_file(piano_path)

        orch_roll = torch.tensor(orch_roll, dtype=torch.float32)
        piano_roll = torch.tensor(piano_roll, dtype=torch.float32)
        onset_labels = torch.tensor(onset_labels, dtype=torch.long)
        frame_labels = torch.tensor(frame_labels, dtype=torch.long)
        offset_labels = torch.tensor(offset_labels, dtype=torch.long)

        # Pad or truncate to fixed length
        orch_roll = pad_or_truncate_tensor(orch_roll, self.max_len)
        piano_roll = pad_or_truncate_tensor(piano_roll, self.max_len)
        onset_labels = pad_or_truncate_tensor(onset_labels, self.max_len)
        frame_labels = pad_or_truncate_tensor(frame_labels, self.max_len)
        offset_labels = pad_or_truncate_tensor(offset_labels, self.max_len)
        
        


        return {'input': torch.tensor(orch_roll), 
                'target': torch.tensor(piano_roll),
                'onset': torch.tensor(onset_labels), 
                'frame': torch.tensor(frame_labels),
                'offset': torch.tensor(offset_labels),
                'filename' : fname
               } 

