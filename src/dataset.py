import pretty_midi
import numpy as np

def midi_to_pianoroll(midi_path, fs=100):
    """
    Converts a MIDI file into a piano-roll.
    
    Args:
        midi_path (str): Path to the .mid file.
        fs (int): Frame rate (frames per second). Determines time resolution.

    Returns:
        np.ndarray: Piano-roll matrix of shape (T, 128) with velocities (0–127).
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    midi = pretty_midi.PrettyMIDI(midi_path)
    
    # Get the piano-roll (velocity values) summed across all instruments
    piano_roll = midi_data.get_piano_roll(fs=fs).T  # shape: (T, 128)
    for i, inst in enumerate(midi.instruments):
        print(f"Instrument {i + 1}: {inst.name or 'Unnamed'}, Program: {inst.program}, "
            f"Drum: {inst.is_drum}, Notes: {len(inst.notes)}")

    # Optional: binarize or clip the velocities if needed
    # piano_roll = (piano_roll > 0).astype(np.uint8)  # For binary roll

    return piano_roll

pianoroll = midi_to_pianoroll("/Users/lucasyan/Winter - Spring 25 research project/Transformer_Piano_Reduction/aligned/bouliane-0/orchestra.mid", fs=80)

print("Shape:", pianoroll.shape)  # e.g., (3000, 128)
print("Note active at time 0:", np.where(pianoroll[0] > 0))
print("First 10 time steps:\n", pianoroll[:10])
active_times = np.where(pianoroll.sum(axis=1) > 0)[0]
print("Active time steps:", active_times)
print(len(active_times))