# config.py
from dataclasses import dataclass
@dataclass
class BaseTransformerConfig:
    # Data paths
    ORCHESTRAL_MIDI_DIR = "data/orchestral_midis"
    PIANO_REDUCTION_MIDI_DIR = "data/piano_reductions_midis"
    VOCAB_DIR = "data/vocabularies"
    SRC_VOCAB_PATH = f"{VOCAB_DIR}/src_vocab.json"
    TGT_VOCAB_PATH = f"{VOCAB_DIR}/tgt_vocab.json"

    # Data processing parameters
    MIDI_QUANTIZATION_STEPS_PER_BEAT = 16 # e.g., 16th notes
    MAX_SEQUENCE_LENGTH = 512 # Max tokens for input/output sequence. Adjust based on avg piece length & GPU memory.

    # Special Tokens (ensure these are consistent with your tokenization)
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>' # Start of Sequence
    EOS_TOKEN = '<eos>' # End of Sequence
    UNK_TOKEN = '<unk>' # Unknown token

    # Model Hyperparameters
    MODEL_DIM = 512       # Embedding dimension for all transformer layers
    NUM_HEADS = 8         # Number of attention heads
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    FF_DIM = MODEL_DIM * 4 # Dimension of the feed-forward network in each transformer layer
    DROPOUT_RATE = 0.1

    # Training parameters
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 100
    BATCH_SIZE = 16
    CLIP_GRAD_NORM = 1.0 # Gradient clipping to prevent exploding gradients
    SAVE_INTERVAL_EPOCHS = 5 # Save model every X epochs
    CHECKPOINT_DIR = "checkpoints" # Directory to save model weights
    LOG_INTERVAL_BATCHES = 10 # Print loss every X batches

    # Inference parameters
    INFERENCE_TEMPERATURE = 1.0 # Controls randomness of generation (1.0 = normal, <1.0 = more deterministic, >1.0 = more random)
    INFERENCE_MAX_LEN = 1024 # Max length for generated sequence during inference

    VOCAB_SIZE = 512  # or whatever your vocabulary size is (notes, tokens, etc.)
    BLOCK_SIZE = 512  # max sequence length
    N_EMBD = 512      # embedding dimension
    DROPOUT = 0.1
    NUM_LAYERS = 8       # number of encoder/decoder layers
    BIAS = True 