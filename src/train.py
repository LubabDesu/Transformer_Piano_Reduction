# Training loop
from model import PianoReductionTransformer, 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from onsetsandframes import OnsetsAndFrames

def train_model(model, train_loader, val_loader, num_epochs = 100, device='cpu') :
    pos_weight_value = 0.4
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current device being used is .... : " + str(device))
    model = model.to(device)

    loss_calc = OnsetsAndFrames()
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
 
        for batch in train_loader:
            for key in batch: # Move the tensors to the correct device
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            batch_inputs   = batch['input']      # (B, T, 128)
            batch_targets = batch['target']
            
            # Forward pass
            optimizer.zero_grad()  # Clear previous gradients
            predictions = model(batch_inputs)  # Get predictions
            losses = loss_calc.onsetFrameLoss(batch, predictions=predictions)
            loss = sum(losses.values())

            # Backward pass and optimize
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        #Validation phase
        model.eval() #Set model to evaluation mode 
        running_val_loss = 0.0
        
        with torch.no_grad() : 
            for batch in val_loader:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                batch_inputs = batch['input']
                batch_targets = batch['target']

                outputs = model(batch_inputs)
                val_losses = loss_calc.onsetFrameLoss(outputs, batch_targets)
                val_loss = sum(val_losses.values())
                running_val_loss += val_loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

    return model