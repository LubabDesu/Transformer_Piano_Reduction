import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Assuming these modules exist and contain the necessary classes/functions
from model import PianoReductionTransformer
from data_processing import PianoReductionDataset
from train import train_model as run_training_loop # Import the actual training loop function




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


def create_dataloaders(input_train, target_train, input_val, target_val, input_test, target_test, batch_size=64, num_workers=0):
    """
    Creates PyTorch DataLoaders for the training, validation, and test sets.
    """
    if not all([input_train, target_train, input_val, target_val, input_test, target_test]):
         print("Error: Cannot create DataLoaders - one or more input lists are missing.")
         return None, None, None, None

    train_dataset = PianoReductionDataset(input_files=input_train, target_files=target_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = PianoReductionDataset(input_files=input_val, target_files=target_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_dataset = PianoReductionDataset(input_files=input_test, target_files=target_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"DataLoaders created with batch size: {batch_size}")
    return train_loader, val_loader, test_loader, test_dataset

def setup_and_train_model(train_loader, val_loader, num_epochs=50, model_save_path="bce-loss-model.pth"):
    """
    Initializes the CNN model, determines the device, runs the training loop,
    and saves the trained model's state dictionary.
    """
    # Initialize the model
    model = PianoReductionCNN()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Run the actual training loop (imported function)
    print(f"Starting training for {num_epochs} epochs...")
    # Pass the device to the training loop function
    trained_model = run_training_loop(model=model,
                                      train_loader=train_loader,
                                      val_loader=val_loader,
                                      num_epochs=num_epochs,
                                      device=device)

    # Save the model's state dictionary
    # Note: It might be slightly better practice to save trained_model.state_dict()
    # if run_training_loop returns the trained model instance.
    # Sticking to the original pattern of saving the 'model' object's state dict.
    print(f"Training complete. Saving model state_dict to {model_save_path}...")
    torch.save(model.state_dict(), model_save_path)
    print("Model saved successfully.")

# --- Main Execution ---

def main():
 