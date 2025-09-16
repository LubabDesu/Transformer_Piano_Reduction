# 🎹 Piano Reduction with Transformers

This project investigates whether Transformer-based models can automatically reduce full orchestral scores into playable piano arrangements. While the models did not generate plausible reductions, the process offered valuable lessons about the challenges of applying sequence models to symbolic music.

## ✨ Overview
  •	Goal: Translate dense, multi-instrument orchestral input into stylistically faithful piano reductions.\
	•	Approach: Train a Transformer on aligned orchestral–piano MIDI pairs (LOP dataset).\
	•	Outcome: Models converged poorly, often producing sparse or incoherent outputs, highlighting dataset and architectural limitations.

## ⚙️ Technical Highlights
  •	Built with PyTorch and Transformers for sequence modeling.\
	•	Implemented positional encoding + multi-head self-attention for long-range structure.\
	•	Applied loss masking and negative subsampling to address sparsity.

## 📚 Lessons Learned
  •	Data representation matters: Piano reductions require structure (melody, harmony, voice distribution) that tokenized sequences struggled to capture.\
	•	Model limitations: Pure Transformers without hierarchical context often collapse to trivial outputs in highly sparse domains.\
	•	Evaluation gap: Note-level metrics (precision/recall/F1) did not capture musical plausibility, suggesting a need for human-in-the-loop or music-theoretic evaluation.

## 🚀 Future Directions
  •	Explore hierarchical VAEs (e.g., PianoTree-VAE) to model structured latent representations.\
	•	Incorporate music-theoretic priors (e.g., pitch ranges, chordal grouping).\
	•	Extend to multimodal inputs (audio + score) for richer grounding.
