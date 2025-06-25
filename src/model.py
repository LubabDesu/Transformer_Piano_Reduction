import torch.nn as nn 
import torch
import transformers
import math
import sys
from dataclasses import dataclass

from config import BaseTransformerConfig
#hi hello test
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


        print(f"DEBUG: PositionalEncoding __init__ finished. Type of self.pe: {type(self.pe)}")
        print(f"DEBUG: PositionalEncoding __init__ finished. Shape of self.pe: {self.pe.shape}")

    def forward(self, x : torch.Tensor):

        print(f"DEBUG: PositionalEncoding forward started. Type of self.pe: {type(self.pe)}")
        print(f"DEBUG: PositionalEncoding forward started. Shape of input x: {x.shape}")
        print(f"DEBUG: PositionalEncoding forward started. Expected slice for self.pe: {self.pe.shape[0]}, {x.size(1)}, {self.pe.shape[2]}")
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

@dataclass
class PianoReductionTransformerConfig(BaseTransformerConfig) :
    pass

class PianoReductionTransformer(nn.Module) :
    '''Defines the main Piano Reduction Transformer class'''
    def __init__(self, config : PianoReductionTransformerConfig, src_vocab_size: int, tgt_vocab_size: int) :
        super().__init__()

        #Store config
        self.config = config

        #Input embeddings 
        self.encoder_embedding = nn.Embedding(src_vocab_size, config.MODEL_DIM)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, config.MODEL_DIM)

        #Positional Encoding Layer
        self.positional_encodings = PositionalEncoding(config.MODEL_DIM, 0.5, config.MAX_SEQUENCE_LENGTH)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)

        # The encoder part.
        self.encoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.VOCAB_SIZE, config.N_EMBD),
            wpe = nn.Embedding(config.BLOCK_SIZE, config.N_EMBD),
            drop = nn.Dropout(config.DROPOUT),
        ))

        # The decoder part.
        self.decoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.VOCAB_SIZE, config.N_EMBD),
            wpe = nn.Embedding(config.BLOCK_SIZE, config.N_EMBD),
            drop = nn.Dropout(config.DROPOUT),
        ))

        self.transformer = nn.Transformer(
            d_model=config.MODEL_DIM,
            nhead=config.NUM_HEADS,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            dim_feedforward=config.FF_DIM,
            dropout=config.DROPOUT_RATE,
            #batch_first = True
        )

        self.onset_head = nn.Linear(config.MODEL_DIM, config.VOCAB_SIZE)
        self.offset_head = nn.Linear(config.MODEL_DIM, config.VOCAB_SIZE)
        self.frame_head = nn.Linear(config.MODEL_DIM, config.VOCAB_SIZE)
        self.velo_head = nn.Linear(config.MODEL_DIM, config.VOCAB_SIZE)

    def forward(self, encoder_ids, decoder_ids, target_ids=None, padding_mask=None):
        device = encoder_ids.device
        b, t = encoder_ids.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        #Embed inputs
        tok_emb_encoder = self.encoder_embedding(encoder_ids)
        tok_emb_encoder += self.positional_encodings(tok_emb_encoder)
        tok_emb_encoder = self.dropout(tok_emb_encoder)


        pos_emb_decoder = self.decoder_embedding(decoder_ids)
        pos_emb_decoder += self.positional_encodings(pos_emb_decoder)
        pos_emb_decoder = self.dropout(pos_emb_decoder)

        tgt_seq_len = pos_emb_decoder.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)

        x = self.transformer(src=tok_emb_encoder,
                              tgt=pos_emb_decoder,
                              tgt_mask=tgt_mask)

        return {
            'onset': self.onset_head(x),
            'frame': self.frame_head(x),
            'offset': self.offset_head(x),
            'velocity': self.velo_head(x)
        }

    










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