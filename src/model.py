import torch.nn as nn 
import torch
import transformers
import math
import sys
from dataclasses import dataclass
from config import BaseTransformerConfig

class PositionalEncoding(nn.Module) :
    """
    Applies sinusoidal positional encoding to the input embeddings.
    This helps the Transformer understand the order of time steps (sequence_length).
    """
    def __init__ (self, d_model : int, dropout : float, max_len : int = 5000):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)


    def forward(self, x : torch.Tensor):

        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

@dataclass
class PianoReductionTransformerConfig(BaseTransformerConfig) :
    pass

class PianoReductionTransformer(nn.Module) :
    '''Defines the main Piano Reduction Transformer class'''
    def __init__(self, config, num_pitches):
        super().__init__()
        self.input_proj  = nn.Linear(num_pitches, config.MODEL_DIM)
        self.pos_enc     = PositionalEncoding(config.MODEL_DIM, config.DROPOUT_RATE, config.MAX_SEQUENCE_LENGTH)
        self.transformer = nn.Transformer(
            d_model=config.MODEL_DIM,
            nhead=config.NUM_HEADS,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            dim_feedforward=config.FF_DIM,
            dropout=config.DROPOUT_RATE,
            batch_first=True
        )
        self.onset_proj = nn.Linear(config.MODEL_DIM, num_pitches)
        self.offset_proj = nn.Linear(config.MODEL_DIM, num_pitches)
        self.frame_proj = nn.Linear(config.MODEL_DIM, num_pitches)

    def forward(self, x):
        
        x = x.permute(0, 2, 1)
        # Now x is [Batch, Time, Pitches], e.g., [16, 512, 128]

        x = self.input_proj(x)
        # Now x is [Batch, Time, d_model], e.g., [16, 512, 512]

        x = self.pos_enc(x)
        x = self.transformer(x, x)
        # The transformer output x is the model's "understanding" of the music
        # Shape is still [Batch, Time, d_model]
        
# # For sigmoid, normal BCE Loss
#         # Pass the transformer's output to each head to get separate predictions
#         onset_pred = torch.sigmoid(self.onset_proj(x))
#         offset_pred = torch.sigmoid(self.offset_proj(x))
#         frame_pred = torch.sigmoid(self.frame_proj(x))
#         # Each prediction now has the shape [Batch, Time, Pitches]

#         # Return a dictionary that the loss function expects
#         return {
#             'onset': onset_pred,
#             'offset': offset_pred,
#             'frame': frame_pred
#         }

#BCE With Logits loss 
        onset_logits  = self.onset_proj(x)   # [B, T, P]
        offset_logits = self.offset_proj(x)
        frame_logits  = self.frame_proj(x)
    
        return {'onset': onset_logits, 'offset': offset_logits, 'frame': frame_logits}


def in_venv():
    return sys.prefix != sys.base_prefix

def main() :
    print("in main method now!")

    class TempConfig(BaseTransformerConfig):
        PAD_TOKEN_ID = 0 # Assuming 0 is your pad token ID for testing

    config_instance = TempConfig()
    dummy_src_vocab_size = 100
    dummy_tgt_vocab_size = 50

    # Instantiate the model
    model = PianoReductionTransformer(config_instance, dummy_src_vocab_size, dummy_tgt_vocab_size)
    print("Model initialized successfully.")
    print(model)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create dummy input tensors
    dummy_src_ids = torch.randint(0, dummy_src_vocab_size, (2, 64), device=device) # Batch size 2, seq len 64
    dummy_tgt_input_ids = torch.randint(0, dummy_tgt_vocab_size, (2, 32), device=device) # Batch size 2, seq len 32

    # Apply some padding to test masks
    dummy_src_ids[0, 60:] = config_instance.PAD_TOKEN_ID
    dummy_tgt_input_ids[1, 20:] = config_instance.PAD_TOKEN_ID

    print("\nAttempting forward pass with dummy data...")
    try:
        output_logits = model(dummy_src_ids, dummy_tgt_input_ids)
        print(f"Forward pass successful! Output logits shape: {output_logits.shape}")
        assert output_logits.shape == (dummy_tgt_input_ids.shape[0], dummy_tgt_input_ids.shape[1], dummy_tgt_vocab_size), "Output shape mismatch!"
        print("Output shape matches expected.")
    except Exception as e:
        print(f"An error occurred during dummy forward pass: {e}")

if __name__ == "__main__" :
    main()
