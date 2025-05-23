# Project Structure: Transliteration with and without Attention

This repository is structured into two core implementations:

1. `attn_encoder_decoder` — Encoder-Decoder model **with Attention**
2. `vanilla_encoder_decoder` — Encoder-Decoder model **without Attention**

---

## 📂 Folder Contents

Each folder (`attn_encoder_decoder`, `vanilla_encoder_decoder`) contains the following core scripts:

### 1. `dataload.py`
- Loads the dataset from TSV files.
- train/validation/test splits of Dakshina Dataset is handled.
- Tokenizes inputs and targets.
- Returns vocabularies and PyTorch DataLoaders.

### 2. `model_attn.py`
- Defines the encoder-decoder architecture.
- `attn_encoder_decoder`: includes an attention module.
- Both models support configurable embedding size, hidden size , number of layers, and RNN type (RNN/GRU/LSTM).

### 3. `train_attn.py`
- Handles training with or without Weights & Biases (wandb) sweeps.
- Supports:
  - Hyperparameter configuration
  - Training loop with validation and model checkpointing
- Logs metrics (accuracy, loss) and saves the best model.

### 4. `evaluate_attn.py`
- Loads the best saved model and runs inference on the **test set**.
- Computes:
  - Character-level and word-level accuracy
  - Beam search decoding
  - Attention heatmaps for the test split
  - logs attention visualizations (for `attn_encoder_decoder`)
- Saves predictions to a `.tsv` or `.csv` file with columns:  
  `source`, `target`, `prediction`, `status`.
- ## python3 evaluate_attn.py --model_path <path_to_bestmodel> --dataset_path <path_to_dataset>

---

## 📁 Output Folder

Each model saves its predictions in a **dedicated output folder**, structured as:
- predictions_attn folder
- └── decoded_outputs.tsv ## Contains Source + Ground Truth + predictions + correctness of labels

## To Execute

## python3 train_attn.py --dataset_path <path_to_the_dataset>

As the attention logic is implemented with the best sweep config of vanilla model the hyperparameters mentioned in the report are mentioned under default config.


## Viss2.py and visualize.py are used to plot the attention connectivity.
## python3 visualize.py --model_path <path_to_bestmodel> --dataset_path <path_to_dataset>
