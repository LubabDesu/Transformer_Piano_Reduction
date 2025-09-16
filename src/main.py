import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import BaseTransformerConfig
from model import PianoReductionTransformer
from data_processing import PianoReductionDataset
from train import train_model

# Global variables 

BASE_DATA_DIR = "Transformer_Piano_reduction/Transformer_Piano_Reduction/aligned" # Base directory containing piece subfolders
VAL_RATIO = 0.15
TRAIN_RATIO = 0.70 # Implicitly defined by 1 - VAL_RATIO - TEST_RATIO
TEST_RATIO = 0.15
RANDOM_STATE = 42 # For reproducible splits
BATCH_SIZE = 8
NUM_WORKERS = 0 # Set higher if you have multiple CPU cores and data loading is a bottleneck
NUM_EPOCHS = 100
MODEL_SAVE_PATH = "transformer-weighted-v3.pth"

# --- Function Definitions ---

def find_file_pairs(base_dir):
    """
    Scans subdirectories of base_dir for 'orchestra.mid' and 'piano.mid' files
    and returns lists of corresponding input (orchestra) and target (piano) file paths.
    """
    file_pairs = {} # Dictionary to store pairs: {piece_name: {'orch': path, 'piano': path}}
    found_orchestra = 0
    found_piano = 0

    print(f"Searching for MIDI pairs in subfolders of: {base_dir}")

    if not os.path.isdir(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return [], []

    for subfolder in os.listdir(base_dir):
        if subfolder == ".DS_Store": # Ignore system files
            continue
        subfolder_path = os.path.join(base_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        piece_name = subfolder  # The subfolder name is the piece name
        file_pairs[piece_name] = {} # Initialize an entry for this piece

        for filename in os.listdir(subfolder_path):
            full_path = os.path.join(subfolder_path, filename)
            if filename == "orchestra.mid":
                file_pairs[piece_name]['orch'] = full_path
                found_orchestra += 1
            elif filename == "piano.mid":
                file_pairs[piece_name]['piano'] = full_path
                found_piano += 1

    # Filter for complete pairs and create the final lists
    all_input_files = []
    all_target_files = []
    paired_count = 0
    for piece_name, paths in file_pairs.items():
        if 'orch' in paths and 'piano' in paths:
            all_input_files.append(paths['orch'])
            all_target_files.append(paths['piano'])
            paired_count += 1

    print(f"Found {found_orchestra} orchestra files and {found_piano} piano files.")
    print(f"Successfully paired {paired_count} files.")

    if paired_count == 0:
        print("Warning: No complete pairs ('orchestra.mid', 'piano.mid') found.")

    return all_input_files, all_target_files



def split_data(all_input_files, all_target_files, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Splits the input and target file lists into training, validation, and test sets.
    """
    if not all_input_files or not all_target_files:
        print("Error: Cannot split data - input file lists are empty.")
        return None, None, None, None, None, None # Return None for all sets

    # Ensure ratios are valid
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and (val_ratio + test_ratio) < 1):
         print(f"Error: Invalid split ratios. val={val_ratio}, test={test_ratio}")
         return None, None, None, None, None, None

    # Split into training + validation AND Test
    input_train_val, input_test, target_train_val, target_test = train_test_split(
        all_input_files,
        all_target_files,
        test_size=test_ratio,
        random_state=random_state
    )

    # Calculate the validation split size relative to the remaining train_val set
    # E.g., if train_ratio=0.7, test_ratio=0.15, val_ratio=0.15
    # train_val size is 0.85 of total. We want val to be 0.15 of total.
    # So, val_split_ratio = 0.15 / 0.85
    val_split_ratio = val_ratio / (1.0 - test_ratio)

    # Split training + validation -> training AND validation
    input_train, input_val, target_train, target_val = train_test_split(
        input_train_val,
        target_train_val,
        test_size=val_split_ratio,
        random_state=random_state # Use same random state for reproducibility if desired, or different for more randomness
    )

    return input_train, input_val, input_test, target_train, target_val, target_test


def create_dataloaders(input_train, target_train, input_val, target_val, input_test, target_test, batch_size=16, num_workers=0):
    """
    Creates PyTorch DataLoaders for the training, validation, and test sets.
    """
    if not all([input_train, target_train, input_val, target_val, input_test, target_test]):
         print("Error: Cannot create DataLoaders - one or more input lists are missing.")
         return None, None, None, None

    train_dataset = PianoReductionDataset(input_files=input_train, target_files=target_train, max_len=512)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = PianoReductionDataset(input_files=input_val, target_files=target_val, max_len=512)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_dataset = PianoReductionDataset(input_files=input_test, target_files=target_test, max_len=512)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"DataLoaders created with batch size: {batch_size}")
    return train_loader, val_loader, test_loader, test_dataset

def setup_and_train_model(train_loader, val_loader, num_epochs=50, model_save_path="transformer-v1.pth"):
    """
    Initializes the CNN model, determines the device, runs the training loop,
    and saves the trained model's state dictionary.
    """

    if train_loader is None:
        print("Exiting script due to DataLoader creation error. I am in the setup and train model function itself")
        return

    
    config = BaseTransformerConfig()
    # Initialize the model
    model = PianoReductionTransformer(
        config=config,
        num_pitches=128
    )

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Run the actual training loop (imported function)
    print(f"Starting training for {num_epochs} epochs...")
    # Pass the device to the training loop function
    trained_model = train_model(model=model,
                                      train_loader=train_loader,
                                      val_loader=val_loader,
                                      num_epochs=num_epochs,
                                      device=device)

    # Save the model's state dictionary
    print(f"Training complete. Saving model state_dict to {model_save_path}...")
    torch.save(model.state_dict(), model_save_path)
    print("Model saved successfully.")
    
def count_pos_neg(loader, key):
    pos = neg = 0
    for b in loader:
        y = b[key].float()   # shape [B, P, T] or [B, T, P]
        pos += (y == 1).sum().item()
        neg += (y == 0).sum().item()
    return pos, neg
    
# --- Main Execution ---

def main():
    all_input_files, all_target_files = find_file_pairs(BASE_DATA_DIR)
    
    # 2. Split data into train, validation, test sets    
    input_train, input_val, input_test, target_train, target_val, target_test = split_data(
        all_input_files, all_target_files,
        val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, random_state=RANDOM_STATE
    )


    ds = PianoReductionDataset(input_files=input_test, target_files=target_test, max_len=512)
    
    if input_train is None:
         print("Exiting script due to data splitting error.")
         return

    print(f"Data split completed:")
    print(f"  Training samples: {len(input_train)}")
    print(f"  Validation samples: {len(input_val)}")
    print(f"  Test samples: {len(input_test)}")

    # 3. Create DataLoaders
    train_loader, val_loader, test_loader, test_dataset = create_dataloaders(
        input_train, target_train, input_val, target_val, input_test, target_test,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # Exit if DataLoader creation failed
    if train_loader is None:
        print("Exiting script due to DataLoader creation error.")
        return

    # 4. Initialize, Train, and Save the Model
    # Note: test_loader is created but not used in the training setup function provided.
    setup_and_train_model(
        train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        model_save_path=MODEL_SAVE_PATH
    )

    print("\nScript execution finished.")

if __name__ == "__main__" :
    main()
 
