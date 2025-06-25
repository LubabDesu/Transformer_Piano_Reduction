# Piano Reduction using Transformers
This project investigates using transformer models to perform piano reduction. The goal is to automatically convert multi-instrumental music into a simplified version playable on a single piano.

Our approach takes orchestral pieces, transforms them into piano roll format, and feeds them into a transformer network. The model learns to generate a piano-playable reduction, also in piano roll format, by mapping the complex orchestral input to a simpler piano output.

We're currently using the onsets and frames loss function to train the model, which helps it accurately predict note beginnings and durations. The entire system is built with PyTorch, leveraging a dataset of aligned orchestral and piano reduction MIDI file pairs.
