# Project Structure: Transliteration with and without Attention

This repository is structured into two core implementations:

1. `attn_encoder_decoder` — Encoder-Decoder model **with Attention**
2. `vanilla_encoder_decoder` — Encoder-Decoder model **without Attention**

---

## 📂 Folder Contents

Each folder (`attn_encoder_decoder`, `vanilla_encoder_decoder`) contains the following core scripts:

### 1. `dataload.py`
- Loads the dataset from TSV files.
- Prepares train/validation/test splits.
- Tokenizes inputs and targets.
- Returns vocabularies and PyTorch DataLoaders.

### 2. `model_attn.py`
- Defines the encoder-decoder architecture.
- `attn_encoder_decoder`: includes an attention module (Luong-style).
- `vanilla_encoder_decoder`: uses basic encoder-decoder without attention.
- Both models support configurable embedding size, number of layers, and RNN type (RNN/GRU/LSTM).

### 3. `train_attn.py`
- Handles training with or without Weights & Biases (wandb) sweeps.
- Supports:
  - Hyperparameter configuration
  - Layer freezing options (optional)
  - Training loop with validation and model checkpointing
- Logs metrics (accuracy, loss) and saves the best model.

### 4. `evaluate_attn.py`
- Loads the best saved model and runs inference on the **test set**.
- Computes:
  - Character-level and word-level accuracy
  - Beam search decoding
  - Optionally logs attention visualizations (for `attn_encoder_decoder`)
- Saves predictions to a `.tsv` or `.csv` file with columns:  
  `source`, `target`, `prediction`, `status`.

---

## 📁 Output Folder

Each model saves its predictions in a **dedicated output folder**, structured as:
predictions/
└── test_predictions.tsv # Contains predictions + correctness labels

## To Execute

python3 train_attn.py --dataset_path <path_to_the_dataset>

As the attention logic is implemented with the best sweep config of vanilla model the hyperparameters mentioned in the report are mentioned under default config

