import pretty_midi 
import os
import numpy as np
import torch
from torch.utils.data import Dataset
# print(os.path.dirname(pretty_midi.__file__))

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
    print(f"--- Starting to load MIDI file: {filepath} ---")
    midi_data = pretty_midi.PrettyMIDI(filepath)
    piano_roll = midi_data.get_piano_roll(fs=fs)
    piano_roll = (piano_roll > 0).astype(np.float64)
    fname = os.path.basename(filepath)

    return piano_roll, fname

def get_piano_labels(piano_roll) :
    onset_labels = np.zeros_like(piano_roll)
    offset_labels = np.zeros_like(piano_roll)

    #for onset labels
    onset_labels[0] = piano_roll[0] > 0 
    onset_labels[1:] = (piano_roll[1:] == 1) & (piano_roll[:-1] == 0)

    #for offset labels 
    offset_labels[1:] = (piano_roll[1:] == 0) & (piano_roll[:-1] == 1)

    #for frame labels
    frame_labels = piano_roll.copy()

    return onset_labels, frame_labels, offset_labels
    


class PianoReductionDataset(Dataset):
    def __init__(self, input_files, target_files, fs):
        assert len(input_files) == len(target_files), "Input and target file counts must match"
        self.pairs = list(zip(input_files, target_files))
        self.fs = fs


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        orch_path, piano_path = self.pairs[idx]

        orch_roll, fname = load_midi_file(orch_path)
        piano_roll, onset_labels, frame_labels, offset_labels, fname = load_piano_midi_file(piano_path)
        


        return {'input': torch.tensor(orch_roll), 
                'target': torch.tensor(piano_roll),
                'onset': torch.tensor(onset_labels), 
                'frame': torch.tensor(frame_labels),
                'offset': torch.tensor(offset_labels),
                'filename' : fname} 

